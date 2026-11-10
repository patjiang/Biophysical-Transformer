#!/usr/bin/env python
"""FLIP Meltome (mixed split) evaluation for the SE3PLM encoder.

Three subcommands::

    python flip_meltome.py embed --data flip_data/meltome/mixed_split.csv \\
        --ckpt chkpts/se3plm.ckpt --out meltome_emb --max-tokens 1024
    python flip_meltome.py run --emb meltome_emb --out meltome_results
    python flip_meltome.py postmortem --results meltome_results

Controlled ESM comparison (same head, splits, seeds)::

    python flip_meltome.py embed --data flip_data/meltome/mixed_split.csv \\
        --esm esm2_t33_650M_UR50D --project 64 --no-ctx \\
        --out meltome_emb_esm650 --batch-size 4
    python flip_meltome.py run --emb meltome_emb_esm650 --seeds 3 \\
        --arms aa_comp dipep ridge_bos ridge_mean ridge_stat cnn attn \\
        --out meltome_results_esm650

Heads: ridge_mean, ridge_bos, ridge_stat, cnn, attn.
Controls: aa_comp (20 AA fractions + length), dipep (400 dipeptide fractions).
Context arms (``--ctx``): ridge_stat_ctx, cnn_ctx, attn_ctx, cnn_both, attn_both.
"""

import argparse
import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

AAS = "ARNDCQEGHILKMFPSTWYV"
AA_IDX = {a: i for i, a in enumerate(AAS)}
IVYWREL = set("IVYWREL")
CHARGED = set("DEKR")
ALPHAS = np.logspace(-2, 5, 15)

FLIP_TABLE6_MIXED = {
    "ESM-1b (per AA)": 0.68, "ESM-1b (mean)": 0.68,
    "ESM-1v (per AA)": 0.65, "ESM-1v (mean)": 0.67,
    "ESM-untrained (per AA)": 0.44, "ESM-untrained (mean)": 0.36,
    "Ridge (one-hot)": 0.17, "CNN (one-hot)": 0.34,
}

CTX_ARMS = ("ridge_stat_ctx", "cnn_ctx", "attn_ctx", "cnn_both", "attn_both")
CTX_PAIRS = (("attn_both", "attn"), ("cnn_both", "cnn"),
             ("attn_ctx", "attn"), ("cnn_ctx", "cnn"),
             ("ridge_stat_ctx", "ridge_stat"))


# ============================================================ 1. EMBED

PAD_ID, MASK_ID, BOS_ID = 8000, 8001, 8002
CTX_SPAN = 2


@torch.no_grad()
def se3_positions(model, x, tensor):
    """Per-position features ``(B, L, C)`` for body positions.

    *tensor*: ``"forward_int"`` (decoder input, the default) or ``"z"``
    (encoder output).  Use ``"z"`` for the voxel ablation -- in the
    no-voxel arm self.i and the identity skip are untrained.
    """
    if tensor == "z":
        return model.forward(x, pred=True)[:, 1:, :]
    if tensor == "forward_int":
        return model.forward_int(x, pred=True)
    raise ValueError(f"unknown embed tensor {tensor!r}")


@torch.no_grad()
def context_embed(model, ids, stride, device, token_budget=1 << 16,
                  tensor="forward_int"):
    """``(L, C)`` float32: each position's feature with its own span hidden.

    Pass *o* hides a 5-token span centred on every position
    ``c = o (mod stride)`` and reads the feature at those centres.  All
    passes for a protein go in one batch, split only if the token budget
    is exceeded.
    """
    L = int(ids.numel())
    if L == 0:
        return None
    base = torch.cat([torch.tensor([BOS_ID], dtype=torch.long),
                      ids.long().cpu()]).to(device)
    offsets = list(range(min(stride, L)))
    per = max(1, min(len(offsets), token_budget // (L + 1)))
    width = torch.arange(-CTX_SPAN, CTX_SPAN + 1, device=device)
    out = None
    for s in range(0, len(offsets), per):
        offs = offsets[s:s + per]
        x = base.repeat(len(offs), 1)
        cents = []
        for r, o in enumerate(offs):
            c = torch.arange(o, L, stride, device=device)
            hid = (c.unsqueeze(1) + width).flatten()
            hid = hid[(hid >= 0) & (hid < L)]
            x[r, 1 + hid] = MASK_ID
            cents.append(c)
        zr = se3_positions(model, x, tensor).float()
        if out is None:
            out = torch.zeros(L, zr.shape[-1])
        for r, c in enumerate(cents):
            out[c.cpu()] = zr[r, c].cpu()
    return out


# ------------------------------------------------------- encoder backends

class SE3Backend:
    """SE3PLM encoder, producing the same cache format as ``ESMBackend``."""

    name = "se3plm"

    def __init__(self, args):
        from src.proteingym_eval import load_model, sanitize
        from src.data import seqs_to_tens
        self.sanitize, self.seqs_to_tens = sanitize, seqs_to_tens
        self.model = load_model(args.ckpt, args.device)
        self.device, self.max_tokens = args.device, args.max_tokens
        self.embed_tensor = getattr(args, "embed_tensor", "forward_int")
        self.tag = f"{args.ckpt}[{self.embed_tensor}]"
        if self.embed_tensor == "forward_int":
            nv = bool(getattr(self.model, "hparams", {}).get("no_voxel", False)) \
                if hasattr(self.model, "hparams") else False
            if nv:
                raise SystemExit(
                    "This checkpoint was trained with no_voxel=True (the "
                    "ablation control arm), where self.i and the identity skip "
                    "never receive gradient.\n"
                    "Re-run with --embed-tensor z, and embed the voxel arm with "
                    "--embed-tensor z as well so the two caches match.")

    def tokenize(self, seq):
        clean, _ = self.sanitize(seq)
        kmers = [clean[i:i + 3] for i in range(len(clean) - 2)]
        ids = self.seqs_to_tens(kmers).long().clamp(max=7999)
        return ids[: self.max_tokens]

    @torch.no_grad()
    def embed(self, block):
        """Embed a list of tokenised 1-D LongTensors -> ``(tok, bos)``."""
        bl = max(max(r.numel() for r in block), 1)
        padded = torch.full((len(block), bl), PAD_ID, dtype=torch.long)
        for j, r in enumerate(block):
            if r.numel():
                padded[j, : r.numel()] = r
        bos = torch.full((len(block), 1), BOS_ID, dtype=torch.long)
        toks = torch.cat([bos, padded], dim=1).to(self.device)
        zr = se3_positions(self.model, toks, self.embed_tensor)
        z = self.model.forward(toks, pred=True)
        return zr.float().cpu(), z[:, 0, :].float().cpu()


class ESMBackend:
    """fair-esm, producing the same cache format as ``SE3Backend``.

    Positions are residues (not 3-mer tokens), so a length-L sequence
    gives L rows instead of L-2.  The mask handles it and every head
    pools over the position axis.
    """

    name = "esm"

    def __init__(self, args):
        try:
            import esm
        except ImportError as e:
            raise SystemExit("--esm needs fair-esm:  pip install fair-esm") from e
        from src.proteingym_eval import sanitize
        self.sanitize = sanitize
        if not hasattr(esm.pretrained, args.esm):
            raise SystemExit(
                f"esm.pretrained has no {args.esm!r}. Common choices: "
                "esm2_t6_8M_UR50D, esm2_t12_35M_UR50D, esm2_t30_150M_UR50D, "
                "esm2_t33_650M_UR50D, esm1b_t33_650M_UR50S")
        model, alphabet = getattr(esm.pretrained, args.esm)()
        self.model = model.to(args.device).eval()
        self.bc = alphabet.get_batch_converter()
        self.layer = args.esm_layer if args.esm_layer is not None else model.num_layers
        if not 0 <= self.layer <= model.num_layers:
            raise SystemExit(f"--esm-layer {self.layer} outside 0..{model.num_layers}")
        self.device, self.max_tokens = args.device, args.max_tokens
        self.tag = f"{args.esm}:layer{self.layer}"
        print(f"[esm] {args.esm}, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M "
              f"parameters, representation layer {self.layer}/{model.num_layers}")

    def tokenize(self, seq):
        """Residue string as a 1-D uint8 tensor (truncated to max_tokens)."""
        clean, _ = self.sanitize(seq)
        clean = clean[: self.max_tokens]
        return torch.tensor(bytearray(clean.encode()), dtype=torch.uint8)

    @torch.no_grad()
    def embed(self, block):
        seqs = [bytes(r.tolist()).decode() for r in block]
        _, _, toks = self.bc([(f"s{i}", s) for i, s in enumerate(seqs)])
        out = self.model(toks.to(self.device), repr_layers=[self.layer],
                         return_contacts=False)
        rep = out["representations"][self.layer].float().cpu()
        bl = max(max(len(s) for s in seqs), 1)
        tok = torch.zeros(len(seqs), bl, rep.shape[-1])
        for j, s in enumerate(seqs):
            tok[j, : len(s)] = rep[j, 1 : 1 + len(s)]
        return tok, rep[:, 0, :]


def fit_projection(backend, df, args):
    """PCA fit on train positions only, for ``--project``."""
    from sklearn.decomposition import PCA
    tr = df.index[df.get("set", pd.Series(["train"] * len(df))) == "train"]
    if len(tr) == 0:
        tr = df.index
    rng = np.random.default_rng(args.seed)
    pick = rng.permutation(np.asarray(tr))[: args.project_seqs]
    print(f"  --project {args.project}: fitting PCA on {len(pick)} train "
          f"sequences", flush=True)
    rows, bs = [], max(1, args.batch_size)
    seqs = [backend.tokenize(s) for s in df["sequence"].iloc[pick]]
    seqs = [s for s in seqs if s.numel()]
    for i in range(0, len(seqs), bs):
        tok, _ = backend.embed(seqs[i:i + bs])
        for j, r in enumerate(seqs[i:i + bs]):
            rows.append(tok[j, : r.numel()])
        if sum(r.shape[0] for r in rows) >= args.project_sample:
            break
    X = torch.cat(rows).numpy()
    if len(X) > args.project_sample:
        X = X[rng.permutation(len(X))[: args.project_sample]]
    if args.project > X.shape[1]:
        raise SystemExit(f"--project {args.project} exceeds the backend's "
                         f"{X.shape[1]} dims")
    p = PCA(n_components=args.project, random_state=args.seed).fit(X)
    var = float(p.explained_variance_ratio_.sum())
    print(f"  PCA {X.shape[1]} -> {args.project} on {len(X)} positions; "
          f"keeps {100 * var:.1f}% of the variance", flush=True)
    return (torch.tensor(p.mean_, dtype=torch.float32),
            torch.tensor(p.components_.T.copy(), dtype=torch.float32), var)


def cmd_embed(args):
    if args.device == "cpu":
        n_threads = args.threads or (os.cpu_count() or 1)
        torch.set_num_threads(n_threads)
        print(f"CPU run: {n_threads} torch threads")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    df = df.dropna(subset=["sequence", "target"]).reset_index(drop=True)
    print(f"{len(df)} sequences from {args.data}")
    trunc = (df["sequence"].str.len() - 2 > args.max_tokens).sum()
    if trunc:
        print(f"  warning: {trunc} sequences ({100 * trunc / len(df):.1f}%) exceed "
              f"--max-tokens {args.max_tokens} and will be truncated")

    backend = ESMBackend(args) if args.esm else SE3Backend(args)
    if args.esm and args.ctx:
        raise SystemExit("--ctx is an SE3PLM masking trick with no ESM "
                         "counterpart; pass --no-ctx")
    print(f"backend {backend.name} ({backend.tag}) on {args.device}")

    proj = None
    if args.project:
        proj = fit_projection(backend, df, args)
        torch.save({"mean": proj[0], "components": proj[1],
                    "explained_variance": proj[2]}, out_dir / "projection.pt")

    def apply_proj(t):
        return t if proj is None else (t - proj[0]) @ proj[1]

    n = len(df)
    chunk = args.chunk_size
    meta = {"n": n, "dim": None, "max_tokens": args.max_tokens,
            "chunks": [], "source": str(args.data),
            "backend": backend.name, "model": backend.tag,
            "ckpt": str(args.ckpt) if not args.esm else None,
            "esm": args.esm, "project": args.project,
            "embed_tensor": getattr(args, "embed_tensor", "forward_int"),
            "ctx": bool(args.ctx),
            "ctx_tensor": (getattr(args, "embed_tensor", "forward_int")
                           if args.ctx else None),
            "ctx_stride": args.ctx_stride if args.ctx else None}
    if args.ctx:
        print(f"context embedding ON: one {2 * CTX_SPAN + 1}-token span hidden "
              f"every {args.ctx_stride} tokens, {args.ctx_stride} passes per protein")

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        rows, lengths = [], []
        for s in df["sequence"].iloc[start:end]:
            ids = backend.tokenize(s)
            rows.append(ids)
            lengths.append(max(ids.numel(), 1))
        L = max(lengths)

        embs = []
        bs = args.batch_size
        for i in range(0, len(rows), bs):
            block = rows[i:i + bs]
            e, b = backend.embed(block)
            e, b = apply_proj(e), apply_proj(b)
            if e.shape[1] < L:
                e = F.pad(e, (0, 0, 0, L - e.shape[1]))
            embs.append((e.half(), b.half()))
            if (i // bs) % 20 == 0:
                print(f"  {start + i + len(block)}/{n}", flush=True)

        tok = torch.cat([a for a, _ in embs], dim=0)
        bosv = torch.cat([b for _, b in embs], dim=0)
        mask = (torch.arange(L)[None, :] < torch.tensor(lengths)[:, None])

        extra = {}
        if args.ctx:
            ctx = torch.zeros(len(rows), L, tok.shape[-1], dtype=torch.float16)
            for j, r in enumerate(rows):
                c = context_embed(backend.model, r, args.ctx_stride, args.device,
                                  tensor=getattr(args, "embed_tensor",
                                                 "forward_int"))
                if c is not None:
                    ctx[j, :c.shape[0]] = apply_proj(c).half()
                if j % 500 == 0:
                    print(f"  ctx {start + j + 1}/{n}", flush=True)
            extra["ctx"] = ctx

        meta["dim"] = tok.shape[-1]
        path = out_dir / f"chunk_{start // chunk:04d}.pt"
        torch.save({"tokens": tok, "bos": bosv, "mask": mask, **extra,
                    "target": torch.tensor(df["target"].iloc[start:end].values,
                                           dtype=torch.float32),
                    "set": df["set"].iloc[start:end].tolist(),
                    "validation": df["validation"].iloc[start:end].tolist()
                    if "validation" in df else [False] * (end - start),
                    "seq_len": torch.tensor(
                        df["sequence"].iloc[start:end].str.len().values),
                    "sequence": df["sequence"].iloc[start:end].tolist()},
                   path)
        meta["chunks"].append(path.name)
        print(f"  wrote {path.name}  tokens={tuple(tok.shape)}"
              f"{'  +ctx' if args.ctx else ''}")
        del embs, tok, bosv, extra

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\ndone -> {out_dir}")


# ============================================================ loading

def load_cache(emb_dir):
    """Load and concatenate the chunked embedding cache."""
    emb_dir = Path(emb_dir)
    meta = json.loads((emb_dir / "meta.json").read_text())
    parts = [torch.load(emb_dir / c, weights_only=False) for c in meta["chunks"]]
    Lmax = max(p["tokens"].shape[1] for p in parts)
    has_ctx = all("ctx" in p for p in parts)
    toks, masks, ctxs = [], [], []
    for p in parts:
        t, m = p["tokens"], p["mask"]
        c = p.get("ctx")
        if t.shape[1] < Lmax:
            t = F.pad(t, (0, 0, 0, Lmax - t.shape[1]))
            m = F.pad(m, (0, Lmax - m.shape[1]))
            if c is not None:
                c = F.pad(c, (0, 0, 0, Lmax - c.shape[1]))
        toks.append(t)
        masks.append(m)
        ctxs.append(c)
    return {
        "tokens": torch.cat(toks), "mask": torch.cat(masks),
        **({"ctx": torch.cat(ctxs)} if has_ctx else {}),
        "bos": torch.cat([p["bos"] for p in parts]).float(),
        "target": torch.cat([p["target"] for p in parts]),
        "seq_len": torch.cat([p["seq_len"] for p in parts]),
        "set": sum([p["set"] for p in parts], []),
        "validation": sum([p["validation"] for p in parts], []),
        "sequence": sum([p["sequence"] for p in parts], []),
    }, meta


def _as_bool(v):
    """Parse FLIP's ``validation`` column, which can be bool, str, or NaN."""
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "t")
    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    return bool(v)


def split_idx(d, verbose=True):
    """Return ``(train, val, test)`` index arrays from the cache."""
    s = np.array(d["set"])
    v = np.array([_as_bool(x) for x in d["validation"]], dtype=bool)
    tr = np.where((s == "train") & ~v)[0]
    va = np.where((s == "train") & v)[0]
    te = np.where(s == "test")[0]
    if len(tr) == 0 or len(va) == 0:
        pool = np.where(s == "train")[0]
        perm = np.random.default_rng(0).permutation(pool)
        cut = int(0.9 * len(perm))
        tr, va = perm[:cut], perm[cut:]
        if verbose:
            print("  note: `validation` flag unusable; carved 90/10 out of train")
    if not (len(tr) and len(va) and len(te)):
        raise ValueError(f"bad split: train={len(tr)} val={len(va)} test={len(te)}")
    return tr, va, te


# ============================================================ features

def feat_mean(d):
    t, m = d["tokens"].float(), d["mask"].float().unsqueeze(-1)
    return ((t * m).sum(1) / m.sum(1).clamp(min=1)).numpy()


def feat_stat(d, key="tokens"):
    """mean|std|max|min over the token axis."""
    t, m = d[key].float(), d["mask"].float().unsqueeze(-1)
    cnt = m.sum(1).clamp(min=1)
    mean = (t * m).sum(1) / cnt
    var = ((t - mean.unsqueeze(1)) ** 2 * m).sum(1) / cnt
    neg = torch.finfo(torch.float32).min
    mx = (t.masked_fill(m == 0, neg)).max(1).values
    mn = (t.masked_fill(m == 0, -neg)).min(1).values
    return torch.cat([mean, var.sqrt(), mx, mn], dim=-1).numpy()


def feat_aa_comp(d):
    """20 AA fractions + log length + charged/hydrophobic fractions."""
    rows = []
    for s in d["sequence"]:
        s = s.upper()
        n = max(len(s), 1)
        c = np.zeros(20, dtype=np.float32)
        for ch in s:
            j = AA_IDX.get(ch)
            if j is not None:
                c[j] += 1
        c /= n
        rows.append(np.concatenate([
            c,
            [math.log(n),
             sum(s.count(a) for a in IVYWREL) / n,
             sum(s.count(a) for a in CHARGED) / n],
        ]))
    return np.stack(rows).astype(np.float32)


def feat_dipep(d):
    """400 dipeptide fractions."""
    rows = []
    for s in d["sequence"]:
        s = s.upper()
        c = np.zeros(400, dtype=np.float32)
        tot = 0
        for a, b in zip(s, s[1:]):
            ia, ib = AA_IDX.get(a), AA_IDX.get(b)
            if ia is not None and ib is not None:
                c[ia * 20 + ib] += 1
                tot += 1
        rows.append(c / max(tot, 1))
    return np.stack(rows).astype(np.float32)


# ============================================================ neural heads

class CNNHead(nn.Module):
    """Multi-kernel Conv1d over the token axis, masked pooling, MLP."""

    def __init__(self, dim, kernels=(3, 5, 7, 9), width=128, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, width, k, padding=k // 2),
                nn.BatchNorm1d(width),
                nn.GELU(),
            ) for k in kernels
        ])
        h = width * len(kernels) * 2
        self.mlp = nn.Sequential(
            nn.Linear(h, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x, mask):
        x = x.transpose(1, 2)
        m = mask.unsqueeze(1).float()
        outs = []
        for conv in self.convs:
            h = conv(x)
            h = h[..., : m.shape[-1]] * m
            mx = h.masked_fill(m == 0, torch.finfo(h.dtype).min).max(-1).values
            mn = h.sum(-1) / m.sum(-1).clamp(min=1)
            outs += [mx, mn]
        return self.mlp(torch.cat(outs, dim=-1)).squeeze(-1)


class AttnHead(nn.Module):
    """Gated attention pooling (ABMIL-style)."""

    def __init__(self, dim, hidden=128, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.att_v = nn.Linear(hidden, hidden)
        self.att_u = nn.Linear(hidden, hidden)
        self.att_w = nn.Linear(hidden, 1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x, mask):
        h = self.proj(x)
        a = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h)))
        a = a.masked_fill(~mask.unsqueeze(-1), torch.finfo(a.dtype).min)
        a = torch.softmax(a, dim=1)
        return self.mlp((a * h).sum(1)).squeeze(-1)


def train_head(head, d, tr, va, te, device, epochs=40, bs=64, lr=1e-3,
               patience=8, verbose=True, keys=("tokens",)):
    """Train a neural head with early stopping on validation Spearman.

    *keys*: which cached per-token arrays to concatenate as input.
    Returns ``(test_predictions, best_val_rho)``.
    """
    if len(tr) == 0 or len(va) == 0:
        raise ValueError(f"empty split (train={len(tr)}, val={len(va)})")
    head = head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    y = d["target"]
    mu, sd = y[tr].mean(), y[tr].std().clamp(min=1e-6)

    def batches(idx, shuffle):
        order = np.random.permutation(idx) if shuffle else idx
        for i in range(0, len(order), bs):
            sel = torch.as_tensor(order[i:i + bs])
            yield (torch.cat([d[k][sel].float() for k in keys], -1).to(device),
                   d["mask"][sel].to(device),
                   ((y[sel] - mu) / sd).to(device))

    best, best_state, bad = -np.inf, None, 0
    for ep in range(epochs):
        head.train()
        n_steps = 0
        for x, m, t in batches(tr, True):
            opt.zero_grad()
            F.smooth_l1_loss(head(x, m), t).backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            n_steps += 1
        if ep == 0 and n_steps == 0:
            raise ValueError("zero training steps in epoch 0")
        sched.step()

        head.eval()
        with torch.no_grad():
            pv = np.concatenate([head(x, m).cpu().numpy()
                                 for x, m, _ in batches(va, False)])
        rho = spearmanr(pv, y[va].numpy()).correlation
        if rho is not None and rho > best:
            best, bad = rho, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in head.state_dict().items()}
        else:
            bad += 1
        if verbose and ep % 5 == 0:
            print(f"      ep {ep:3d}  steps {n_steps}  val rho {rho:.4f} "
                  f"(best {best:.4f})")
        if bad >= patience:
            break

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        pt = np.concatenate([head(x, m).cpu().numpy()
                             for x, m, _ in batches(te, False)])
    return pt * sd.item() + mu.item(), best


def fit_ridge(X, y, fit_idx, te):
    sc = StandardScaler().fit(X[fit_idx])
    Xs = sc.transform(X)
    m = RidgeCV(alphas=ALPHAS).fit(Xs[fit_idx], y[fit_idx])
    return m.predict(Xs[te])


# ============================================================ 2. RUN

def cmd_run(args):
    d, meta = load_cache(args.emb)
    tr, va, te = split_idx(d)
    y = d["target"].numpy()
    print(f"train={len(tr)} val={len(va)} test={len(te)}  dim={meta['dim']}  "
          f"max_tokens={meta['max_tokens']}")
    print(f"cache: backend={meta.get('backend', 'se3plm')} "
          f"model={meta.get('model', meta.get('ckpt'))} "
          f"project={meta.get('project')}  "
          f"embed_tensor={meta.get('embed_tensor', 'forward_int')}  "
          f"ctx={meta.get('ctx')} ctx_tensor={meta.get('ctx_tensor')}")

    # Ridge fits on train+val by default (no early stopping needed);
    # --strict-splits restricts it to train only.
    ridge_fit = tr if args.strict_splits else np.concatenate([tr, va])
    print(f"ridge arms fit on {len(ridge_fit)} rows"
          f"{' (strict)' if args.strict_splits else ' (train+val)'}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds, rows = {}, []
    seed_preds = {}

    has_ctx = "ctx" in d
    want_ctx = [a for a in args.arms if a in CTX_ARMS]
    if want_ctx and not has_ctx:
        print(f"  note: no `ctx` in {args.emb}; skipping {want_ctx}")

    ridge_arms = {
        "aa_comp": feat_aa_comp, "dipep": feat_dipep,
        "ridge_bos": lambda dd: dd["bos"].numpy(),
        "ridge_mean": feat_mean, "ridge_stat": feat_stat,
        "ridge_stat_ctx": lambda dd: feat_stat(dd, "ctx"),
    }
    for name, fn in ridge_arms.items():
        if name not in args.arms or (name in CTX_ARMS and not has_ctx):
            continue
        X = fn(d)
        p = fit_ridge(X, y, ridge_fit, te)
        preds[name] = p
        rho = spearmanr(p, y[te]).correlation
        r = pearsonr(p, y[te])[0]
        rows.append({"arm": name, "spearman": rho, "pearson": r,
                     "dim": X.shape[1], "trained": False})
        print(f"  {name:<12} SCC={rho:.4f}  PCC={r:.4f}  dim={X.shape[1]}")

    neural_arms = (("cnn", CNNHead, ("tokens",)), ("attn", AttnHead, ("tokens",)),
                   ("cnn_ctx", CNNHead, ("ctx",)), ("attn_ctx", AttnHead, ("ctx",)),
                   ("cnn_both", CNNHead, ("tokens", "ctx")),
                   ("attn_both", AttnHead, ("tokens", "ctx")))
    for name, cls, keys in neural_arms:
        if name not in args.arms or (name in CTX_ARMS and not has_ctx):
            continue
        print(f"  {name} training...")
        dim = meta["dim"] * len(keys)
        runs = []
        for s in range(args.seeds):
            torch.manual_seed(s)
            np.random.seed(s)
            p_s, val_s = train_head(cls(dim), d, tr, va, te, args.device,
                                    epochs=args.epochs, bs=args.batch_size,
                                    lr=args.lr, keys=keys,
                                    verbose=(args.seeds == 1))
            rho_s = spearmanr(p_s, y[te]).correlation
            runs.append((val_s, rho_s, p_s))
            if args.seeds > 1:
                print(f"      seed {s}: val {val_s:.4f}  test {rho_s:.4f}",
                      flush=True)
        val_rho, rho, p = max(runs, key=lambda q: q[0])
        sd = float(np.std([q[1] for q in runs], ddof=1)) if len(runs) > 1 else np.nan
        preds[name] = p
        for s_i, (_, _, p_s) in enumerate(runs):
            seed_preds[f"{name}_s{s_i}"] = p_s
        r = pearsonr(p, y[te])[0]
        rows.append({"arm": name, "spearman": rho, "pearson": r,
                     "dim": dim, "trained": True, "val_spearman": val_rho,
                     "seeds": args.seeds, "test_sd_over_seeds": sd})
        print(f"  {name:<12} SCC={rho:.4f}  PCC={r:.4f}  (val {val_rho:.4f}"
              + (f", sd over {args.seeds} seeds {sd:.4f})" if len(runs) > 1 else ")"))

    res = pd.DataFrame(rows).sort_values("spearman", ascending=False)
    res.to_csv(out_dir / "arm_scores.csv", index=False)
    pd.DataFrame({**{"y_true": y[te]}, **preds,
                  "seq_len": d["seq_len"].numpy()[te]}).to_csv(
        out_dir / "test_predictions.csv", index=False)
    if seed_preds:
        pd.DataFrame({**{"y_true": y[te]}, **seed_preds,
                      "seq_len": d["seq_len"].numpy()[te]}).to_csv(
            out_dir / "test_predictions_seeds.csv", index=False)
        print(f"  per-seed predictions -> "
              f"{out_dir / 'test_predictions_seeds.csv'} "
              f"({len(seed_preds)} columns)")

    print("\n" + "=" * 66)
    print("FLIP MELTOME (mixed split) -- test set")
    print("=" * 66)
    print(f"{'arm':<14}{'SCC':>9}{'PCC':>9}{'dim':>8}{'trained':>9}")
    for _, r in res.iterrows():
        print(f"{r.arm:<14}{r.spearman:>9.4f}{r.pearson:>9.4f}"
              f"{int(r['dim']):>8}{str(r.trained):>9}")
    print("=" * 66)
    print(f"  -> {out_dir}/   then: python flip_meltome.py postmortem "
          f"--results {out_dir}")


# ============================================================ 3. POSTMORTEM

def cmd_postmortem(args):
    res_dir = Path(args.results)
    pred = pd.read_csv(res_dir / "test_predictions.csv")
    scores = pd.read_csv(res_dir / "arm_scores.csv")
    y = pred["y_true"].values
    arms = [c for c in pred.columns if c not in ("y_true", "seq_len")]
    arms = [a for a in arms if np.isfinite(pred[a].values).all()]
    plm_arms = [a for a in arms if a not in ("aa_comp", "dipep")]
    ctrl_arms = [a for a in arms if a in ("aa_comp", "dipep")]

    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(y), len(y)) for _ in range(500)]

    print("=" * 74)
    print("POSTMORTEM")
    print("=" * 74)

    print("\n[1] test-set Spearman, bootstrapped over test sequences")
    for a in arms:
        v = pred[a].values
        b = [spearmanr(v[i], y[i]).correlation for i in boots]
        lo, hi = np.nanpercentile(b, [2.5, 97.5])
        print(f"    {a:<14}{spearmanr(v, y).correlation:>8.4f}  [{lo:.3f}, {hi:.3f}]")

    if ctrl_arms:
        base = ctrl_arms[0]
        print(f"\n[2] paired delta vs `{base}` (same test sequences)")
        for a in plm_arms:
            dd = np.array([spearmanr(pred[a].values[i], y[i]).correlation -
                           spearmanr(pred[base].values[i], y[i]).correlation
                           for i in boots])
            lo, hi = np.nanpercentile(dd, [2.5, 97.5])
            verdict = ("PLM better" if lo > 0 else
                       "CONTROL better" if hi < 0 else "indistinguishable")
            print(f"    {a:<14}{np.nanmean(dd):>+8.4f}  [{lo:+.3f}, {hi:+.3f}]  {verdict}")

        print(f"\n[3] incremental signal over `{base}`")
        print("    Residualise the PLM prediction against the control, then")
        print("    correlate the residual with the residual of the truth.")
        c = pred[base].values
        for a in plm_arms:
            p = pred[a].values
            rp = p - np.polyval(np.polyfit(c, p, 1), c)
            ry = y - np.polyval(np.polyfit(c, y, 1), c)
            print(f"    {a:<14} partial SCC = {spearmanr(rp, ry).correlation:>7.4f}")

        print(f"\n[4] prediction agreement with `{base}`")
        for a in plm_arms:
            print(f"    {a:<14} SCC(pred, control_pred) = "
                  f"{spearmanr(pred[a].values, c).correlation:>7.4f}")

    print("\n[5] length stratification (quartiles of sequence length)")
    q = pd.qcut(pred["seq_len"], 4, labels=False, duplicates="drop")
    print("    " + "quartile".ljust(12) + "".join(a[:11].rjust(12) for a in arms))
    for k in sorted(pd.unique(q.dropna())):
        sel = (q == k).values
        line = f"    Q{int(k) + 1} n={sel.sum():<7}"
        for a in arms:
            line += f"{spearmanr(pred[a].values[sel], y[sel]).correlation:>12.4f}"
        print(line)
    print(f"\n    SCC(seq_len, y_true) = {spearmanr(pred['seq_len'], y).correlation:.4f}")

    if ctrl_arms and plm_arms:
        best_plm = scores[scores.arm.isin(plm_arms)].iloc[0]["arm"]
        X = np.column_stack([pred[ctrl_arms[0]].values, pred[best_plm].values])
        sc = StandardScaler().fit(X)
        comb = RidgeCV(alphas=ALPHAS).fit(sc.transform(X), y).predict(sc.transform(X))
        print(f"\n[6] stacked {ctrl_arms[0]} + {best_plm} (in-sample, optimistic)")
        print(f"    stacked SCC = {spearmanr(comb, y).correlation:.4f}  vs "
              f"control {spearmanr(pred[ctrl_arms[0]], y).correlation:.4f}, "
              f"{best_plm} {spearmanr(pred[best_plm], y).correlation:.4f}")

    print("\n[7] published FLIP Table 6 baselines (Mixed split, Spearman)")
    mine = {r.arm: r.spearman for _, r in scores.iterrows()
            if np.isfinite(r.spearman)}
    merged = [(k, v, False) for k, v in FLIP_TABLE6_MIXED.items()]
    merged += [(k, v, True) for k, v in mine.items()]
    for name, val, is_mine in sorted(merged, key=lambda x: -x[1]):
        print(f"    {'*' if is_mine else ' '} {name:<26}{val:>7.3f}")
    print("    * = this work")

    pairs = [(a, b) for a, b in CTX_PAIRS if a in arms and b in arms]
    if pairs:
        print("\n[8] context embedding delta (paired bootstrap)")
        for a, b in pairs:
            dd = np.array([spearmanr(pred[a].values[i], y[i]).correlation -
                           spearmanr(pred[b].values[i], y[i]).correlation
                           for i in boots])
            lo, hi = np.nanpercentile(dd, [2.5, 97.5])
            verdict = ("context adds" if lo > 0 else
                       "context hurts" if hi < 0 else "no detectable difference")
            print(f"    {a:<15} - {b:<11}{np.nanmean(dd):>+8.4f}  "
                  f"[{lo:+.3f}, {hi:+.3f}]  {verdict}")

    print("\n" + "=" * 74)


# ============================================================ main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed")
    e.add_argument("--data", required=True)
    e.add_argument("--ckpt", default="chkpts/se3plm.ckpt",
                   help="SE3PLM checkpoint; ignored when --esm is given")
    e.add_argument("--esm", default=None, metavar="MODEL",
                   help="embed with fair-esm instead, e.g. esm2_t33_650M_UR50D")
    e.add_argument("--esm-layer", type=int, default=None,
                   help="representation layer (default: the last)")
    e.add_argument("--project", type=int, default=None, metavar="K",
                   help="PCA per-position features to K dims, fit on train only")
    e.add_argument("--project-seqs", type=int, default=2000)
    e.add_argument("--project-sample", type=int, default=200_000)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", default="meltome_emb")
    e.add_argument("--max-tokens", type=int, default=1024)
    e.add_argument("--chunk-size", type=int, default=4000)
    e.add_argument("--batch-size", type=int, default=16)
    e.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    e.add_argument("--threads", type=int, default=None,
                   help="CPU torch threads (default: all cores)")
    e.add_argument("--no-ctx", dest="ctx", action="store_false",
                   help="skip the context embedding")
    e.add_argument("--embed-tensor", choices=("forward_int", "z"),
                   default="forward_int",
                   help="which SE3PLM tensor to cache. Use z for the voxel "
                        "ablation (both arms must match).")
    e.add_argument("--ctx-stride", type=int, default=33,
                   help="one 5-token span hidden every this many tokens")
    e.set_defaults(func=cmd_embed)

    r = sub.add_parser("run")
    r.add_argument("--emb", default="meltome_emb")
    r.add_argument("--out", default="meltome_results")
    r.add_argument("--arms", nargs="+",
                   default=["aa_comp", "dipep", "ridge_bos", "ridge_mean",
                            "ridge_stat", "cnn", "attn", *CTX_ARMS])
    r.add_argument("--strict-splits", action="store_true",
                   help="ridge fits on train only, matching the neural heads")
    r.add_argument("--seeds", type=int, default=1,
                   help="seeds per neural head; use >=3 when comparing caches")
    r.add_argument("--epochs", type=int, default=40)
    r.add_argument("--batch-size", type=int, default=64)
    r.add_argument("--lr", type=float, default=1e-3)
    r.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    r.set_defaults(func=cmd_run)

    m = sub.add_parser("postmortem")
    m.add_argument("--results", default="meltome_results")
    m.set_defaults(func=cmd_postmortem)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
