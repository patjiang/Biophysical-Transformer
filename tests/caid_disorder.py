#!/usr/bin/env python
"""CAID intrinsic disorder prediction using BioT or ESM2 per-residue embeddings.

Setup (run once from repo root)::

    bash tests/idr_bench/setup_caid.sh

BioT pipeline::

    cd tests/idr_bench

    python ../caid_disorder.py train \\
        --reference CAID/demo-data/references/disorder_pdb.fasta \\
        --ckpt ../../checkpoints/BioT_voxel_s0.safetensors \\
        --out probes/disorder_biot.pkl

    python ../caid_disorder.py predict \\
        --fasta CAID/demo-data/references/disorder_pdb.fasta \\
        --ckpt ../../checkpoints/BioT_voxel_s0.safetensors \\
        --probe probes/disorder_biot.pkl \\
        --out predictions/BioT.caid

ESM2 pipeline (same probe, different embedder)::

    python ../caid_disorder.py train \\
        --esm esm2_t33_650M_UR50D \\
        --reference CAID/demo-data/references/disorder_pdb.fasta \\
        --out probes/disorder_esm2.pkl

    python ../caid_disorder.py predict \\
        --esm esm2_t33_650M_UR50D \\
        --fasta CAID/demo-data/references/disorder_pdb.fasta \\
        --probe probes/disorder_esm2.pkl \\
        --out predictions/ESM2-650M.caid

Evaluate all together::

    python CAID/caid.py \\
        CAID/demo-data/references/disorder_pdb.fasta \\
        predictions/ -o results/
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------ constants

K = 3
BOS_ID = 8002
MAX_REAL_TOKEN = 7999
CKPT_DEFAULT = "checkpoints/BioT_voxel_s0.safetensors"

LABEL_MAP = {"0": 0.0, "1": 1.0, "-": np.nan}
AA20 = set("ACDEFGHIKLMNPQRSTVWY")


# ============================================================ BioT backend

def _biot_imports():
    from src.eval_utils import load_model, sanitize
    from src.data import seqs_to_tens
    return load_model, sanitize, seqs_to_tens


def _kmer_ids(seqs_to_tens, kmers):
    return seqs_to_tens(kmers).long().clamp(max=MAX_REAL_TOKEN)


@torch.no_grad()
def _biot_embed_window(model, tok_1d, device, layer="int"):
    bos = torch.tensor([BOS_ID], dtype=tok_1d.dtype)
    tokens = torch.cat([bos, tok_1d]).unsqueeze(0).to(device)
    if layer == "int":
        z = model.forward_int(tokens, pred=True)
    else:
        z = model.forward(tokens, pred=True)[:, 1:]
    return z[0].float().cpu()


def biot_per_residue(model, seq, device, sanitize_fn, seqs_to_tens_fn,
                     max_tokens=1024, stride=512, layer="int"):
    """BioT per-residue embeddings via k-mer averaging."""
    clean, _ = sanitize_fn(seq)
    L = len(clean)
    kmers = [clean[i:i + K] for i in range(L - (K - 1))]
    tok = _kmer_ids(seqs_to_tens_fn, kmers)
    n_kmers = tok.numel()

    if n_kmers <= 0:
        return None, clean

    if n_kmers <= max_tokens:
        z = _biot_embed_window(model, tok, device, layer)
    else:
        D = None
        accum, counts = None, None
        for start in range(0, n_kmers, stride):
            end = min(start + max_tokens, n_kmers)
            w = _biot_embed_window(model, tok[start:end], device, layer)
            if D is None:
                D = w.shape[1]
                accum = torch.zeros(n_kmers, D)
                counts = torch.zeros(n_kmers, 1)
            accum[start:end] += w
            counts[start:end] += 1
            if end == n_kmers:
                break
        z = accum / counts.clamp(min=1)

    D = z.shape[1]
    res_z = torch.zeros(L, D)
    res_c = torch.zeros(L)
    for dr in range(K):
        end = min(n_kmers, L - dr)
        res_z[dr:dr + end] += z[:end]
        res_c[dr:dr + end] += 1
    res_z /= res_c.unsqueeze(1).clamp(min=1)
    return res_z.numpy(), clean


# ============================================================ ESM2 backend

def _load_esm(model_name, device):
    """Load an ESM2 model by name. Returns (model, alphabet, repr_layer)."""
    try:
        import esm
    except ImportError:
        raise SystemExit("--esm requires fair-esm:  pip install fair-esm")
    if not hasattr(esm.pretrained, model_name):
        raise SystemExit(f"Unknown ESM model: {model_name}\n"
                         f"  Try: esm2_t33_650M_UR50D, esm2_t36_3B_UR50D, "
                         f"esm2_t6_8M_UR50D")
    model, alphabet = getattr(esm.pretrained, model_name)()
    model = model.eval().to(device)
    repr_layer = int(model_name.split("_")[1][1:])  # e.g. t33 -> 33
    return model, alphabet, repr_layer


@torch.no_grad()
def esm_per_residue(model, alphabet, repr_layer, seq, device,
                    max_tokens=1022):
    """ESM2 per-residue embeddings. Returns ``(L, D)`` numpy array."""
    clean = "".join(c if c in AA20 else "A" for c in seq.upper().strip())
    L = len(clean)
    if L == 0:
        return None, clean

    batch_converter = alphabet.get_batch_converter()

    if L <= max_tokens:
        _, _, tokens = batch_converter([("p", clean)])
        tokens = tokens.to(device)
        out = model(tokens, repr_layers=[repr_layer])
        emb = out["representations"][repr_layer][0, 1:L + 1].float().cpu().numpy()
    else:
        # Sliding window for long sequences
        stride = max_tokens // 2
        D = None
        accum, counts = None, None
        for start in range(0, L, stride):
            end = min(start + max_tokens, L)
            chunk = clean[start:end]
            _, _, tokens = batch_converter([("p", chunk)])
            tokens = tokens.to(device)
            out = model(tokens, repr_layers=[repr_layer])
            w = out["representations"][repr_layer][0, 1:len(chunk) + 1].float().cpu()
            if D is None:
                D = w.shape[1]
                accum = np.zeros((L, D), dtype=np.float32)
                counts = np.zeros(L, dtype=np.float32)
            accum[start:end] += w.numpy()
            counts[start:end] += 1
            if end == L:
                break
        emb = accum / counts[:, None].clip(min=1)

    return emb, clean


# ============================================================ unified API

class Embedder:
    """Wraps BioT or ESM2 behind a common per_residue() interface."""

    def __init__(self, args):
        self.device = args.device
        self.backend = "esm" if getattr(args, "esm", None) else "biot"

        if self.backend == "esm":
            print(f"loading ESM: {args.esm}")
            self.esm_model, self.alphabet, self.repr_layer = \
                _load_esm(args.esm, args.device)
            self.max_tokens = min(getattr(args, "max_tokens", 1022), 1022)
            self.name = args.esm
        else:
            load_model, sanitize, seqs_to_tens = _biot_imports()
            self.model = load_model(args.ckpt, args.device)
            self.sanitize = sanitize
            self.seqs_to_tens = seqs_to_tens
            self.max_tokens = getattr(args, "max_tokens", 1024)
            self.stride = getattr(args, "stride", 512)
            self.layer = getattr(args, "layer", "int")
            self.name = "BioT"

    def per_residue(self, seq):
        if self.backend == "esm":
            return esm_per_residue(
                self.esm_model, self.alphabet, self.repr_layer,
                seq, self.device, max_tokens=self.max_tokens)
        return biot_per_residue(
            self.model, seq, self.device, self.sanitize, self.seqs_to_tens,
            max_tokens=self.max_tokens, stride=self.stride, layer=self.layer)


# --------------------------------------------------------------- CAID I/O

def parse_reference_fasta(path):
    entries = []
    with open(path) as f:
        lines = [l.rstrip("\n\r") for l in f
                 if l.strip() and not l.startswith("#")]
    i = 0
    while i < len(lines):
        if lines[i].startswith(">"):
            acc = lines[i][1:].split()[0]
            seq = lines[i + 1]
            lab = np.array([LABEL_MAP.get(c, np.nan) for c in lines[i + 2]])
            if len(seq) != len(lab):
                print(f"  warning: {acc} length mismatch, skipping")
                i += 3
                continue
            entries.append((acc, seq, lab))
            i += 3
        else:
            i += 1
    return entries


def parse_fasta(path):
    entries = []
    with open(path) as f:
        acc, parts = None, []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if acc is not None:
                    entries.append((acc, "".join(parts)))
                acc = line[1:].split()[0]
                parts = []
            elif acc is not None and line:
                parts.append(line)
        if acc is not None:
            entries.append((acc, "".join(parts)))
    return entries


def write_caid(fh, acc, seq, scores, threshold=0.5, write_states=True):
    fh.write(f">{acc}\n")
    for i, (aa, sc) in enumerate(zip(seq, scores)):
        row = f"{i + 1}\t{aa}\t{sc:.3f}"
        if write_states:
            row += f"\t{1 if sc >= threshold else 0}"
        fh.write(row + "\n")


# ------------------------------------------------------------ embedding

def embed_labelled(embedder, entries):
    X_parts, y_parts, g_parts = [], [], []
    skipped = 0
    for idx, (acc, seq, labels) in enumerate(entries):
        emb, clean = embedder.per_residue(seq)
        if emb is None:
            skipped += 1
            continue
        L = min(len(clean), len(labels), emb.shape[0])
        lab = labels[:L]
        emb = emb[:L]
        mask = ~np.isnan(lab)
        if mask.sum() == 0:
            skipped += 1
            continue
        X_parts.append(emb[mask])
        y_parts.append(lab[mask])
        g_parts.append(np.full(int(mask.sum()), idx))
        if (idx + 1) % 25 == 0 or idx + 1 == len(entries):
            print(f"  embedded {idx + 1}/{len(entries)}", flush=True)
    if skipped:
        print(f"  skipped {skipped} proteins")
    return (np.concatenate(X_parts),
            np.concatenate(y_parts),
            np.concatenate(g_parts))


# ----------------------------------------------------------- subcommands

def cmd_train(args):
    embedder = Embedder(args)
    print(f"reference: {args.reference}")
    entries = parse_reference_fasta(args.reference)
    print(f"  {len(entries)} proteins  |  backend: {embedder.name}")

    print("embedding...")
    X, y, groups = embed_labelled(embedder, entries)
    n_pos = int(y.sum())
    print(f"  {len(y):,} residues ({n_pos:,} disordered, "
          f"{len(y) - n_pos:,} ordered, D={X.shape[1]})")

    scaler = StandardScaler().fit(X)
    print("fitting logistic regression...")
    clf = LogisticRegression(
        C=args.C, max_iter=2000, solver="lbfgs",
        class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(scaler.transform(X), y)

    auc = roc_auc_score(y, clf.predict_proba(scaler.transform(X))[:, 1])
    print(f"  training AUC-ROC: {auc:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({
            "scaler": scaler, "clf": clf,
            "backend": embedder.backend, "name": embedder.name,
            "layer": getattr(args, "layer", None),
            "max_tokens": getattr(args, "max_tokens", 1024),
            "stride": getattr(args, "stride", 512),
            "esm": getattr(args, "esm", None),
            "C": args.C, "n_proteins": len(entries),
            "n_residues": len(y), "train_auc": auc,
        }, f)
    print(f"  saved -> {out}")


def cmd_predict(args):
    with open(args.probe, "rb") as f:
        probe = pickle.load(f)
    scaler, clf = probe["scaler"], probe["clf"]

    # Override ESM flag from probe if not explicitly given
    if probe.get("esm") and not getattr(args, "esm", None):
        args.esm = probe["esm"]

    embedder = Embedder(args)
    print(f"probe: {args.probe}  ({probe.get('name', '?')}, "
          f"AUC {probe.get('train_auc', 0):.4f})")

    try:
        ref_entries = parse_reference_fasta(args.fasta)
        entries = [(acc, seq) for acc, seq, _ in ref_entries]
    except Exception:
        entries = parse_fasta(args.fasta)
    print(f"  {len(entries)} target sequences")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    threshold = args.threshold
    write_states = not args.no_states

    with open(out_path, "w") as fh:
        for i, (acc, seq) in enumerate(entries):
            emb, clean = embedder.per_residue(seq)
            if emb is None:
                scores = np.zeros(len(seq))
                clean = seq
            else:
                scores = clf.predict_proba(scaler.transform(emb))[:, 1]
            if len(scores) < len(clean):
                scores = np.pad(scores, (0, len(clean) - len(scores)))
            write_caid(fh, acc, clean, scores, threshold, write_states)
            if (i + 1) % 25 == 0 or i + 1 == len(entries):
                print(f"  predicted {i + 1}/{len(entries)}", flush=True)

    print(f"  wrote {len(entries)} predictions -> {out_path}")


def cmd_cv(args):
    embedder = Embedder(args)
    print(f"reference: {args.reference}  |  backend: {embedder.name}")
    entries = parse_reference_fasta(args.reference)
    print(f"  {len(entries)} proteins")

    print("embedding...")
    X, y, groups = embed_labelled(embedder, entries)
    n_pos = int(y.sum())
    print(f"  {len(y):,} residues ({n_pos:,} disordered, "
          f"{len(y) - n_pos:,} ordered)")

    n_folds = min(args.folds, len(set(groups)))
    gkf = GroupKFold(n_splits=n_folds)
    aucs, f1s, mccs = [], [], []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(
            C=args.C, max_iter=2000, solver="lbfgs",
            class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(sc.transform(X[tr]), y[tr])
        probs = clf.predict_proba(sc.transform(X[te]))[:, 1]
        preds = (probs >= 0.5).astype(float)
        auc = roc_auc_score(y[te], probs)
        f1 = f1_score(y[te], preds)
        mcc = matthews_corrcoef(y[te], preds)
        aucs.append(auc); f1s.append(f1); mccs.append(mcc)
        n_prot = len(set(groups[te]))
        print(f"  fold {fold + 1}/{n_folds}: AUC {auc:.4f}  "
              f"F1 {f1:.4f}  MCC {mcc:.4f}  "
              f"({n_prot} proteins, {len(te):,} residues)")

    print()
    print("=" * 56)
    print(f"  AUC : {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"  F1  : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"  MCC : {np.mean(mccs):.4f} +/- {np.std(mccs):.4f}")
    print("=" * 56)


# ------------------------------------------------------------------- CLI

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--ckpt", default=CKPT_DEFAULT)
        sp.add_argument("--esm", default=None, metavar="MODEL",
                        help="use ESM2 instead of BioT, e.g. esm2_t33_650M_UR50D")
        sp.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
        sp.add_argument("--max-tokens", type=int, default=1024)
        sp.add_argument("--stride", type=int, default=512)
        sp.add_argument("--layer", default="int", choices=["int", "enc"])

    s1 = sub.add_parser("train", help="train disorder probe")
    common(s1)
    s1.add_argument("--reference", required=True)
    s1.add_argument("--C", type=float, default=1.0)
    s1.add_argument("--out", default="probes/disorder.pkl")
    s1.set_defaults(func=cmd_train)

    s2 = sub.add_parser("predict", help="predict disorder -> .caid")
    common(s2)
    s2.add_argument("--fasta", required=True)
    s2.add_argument("--probe", required=True)
    s2.add_argument("--threshold", type=float, default=0.5)
    s2.add_argument("--no-states", action="store_true")
    s2.add_argument("--out", default="predictions/BioT.caid")
    s2.set_defaults(func=cmd_predict)

    s3 = sub.add_parser("cv", help="cross-validate probe")
    common(s3)
    s3.add_argument("--reference", required=True)
    s3.add_argument("--C", type=float, default=1.0)
    s3.add_argument("--folds", type=int, default=5)
    s3.set_defaults(func=cmd_cv)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
