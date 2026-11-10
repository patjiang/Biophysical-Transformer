#!/usr/bin/env python
"""Evaluation harness for the BioT k-mer protein language model.

Three subcommands::

    python eval_utils.py mlm-check  --assay-dir $D --n-assays 5
    python eval_utils.py zeroshot   --assay-dir $D --reference-file $R --out zs_scores/
    python eval_utils.py supervised --assay-dir $D --out sup_scores/

``mlm-check``
    Diagnoses k-mer overlap leakage: compares single-token masking (where
    neighbours reveal all three residues) against span masking (where they
    don't).  Run this first.

``zeroshot``
    Span-masked pseudo-log-likelihood ratio scoring.  For a substitution at
    residue *r*, the tokens containing *r* are masked and the score is the
    summed log-prob ratio (mutant - WT) over the masked span.

``supervised``
    Embed ``mutated_sequence``, fit ridge per CV fold, report Spearman.
"""

import argparse
import warnings
from collections import defaultdict
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import binomtest, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .model import *
from .data import *

torch.set_num_threads(1)

CKPT_PATH = "checkpoints/BioT_voxel_s0.safetensors"

VOCAB_SIZE = 8003
BOS_ID = 8002
MASK_ID = 8001
MAX_REAL_TOKEN = VOCAB_SIZE - 4   # 7999

AA20 = set("ACDEFGHIKLMNPQRSTVWY")
AA20_LIST = sorted(AA20)
K = 3


# ---------------------------------------------------------------- tokenization

def sanitize(seq):
    """Replace non-standard residues so ``seqs_to_tens`` can't emit
    out-of-range ids.  Returns ``(clean_seq, n_replaced)``."""
    out, n = [], 0
    for c in str(seq).upper().strip():
        if c in AA20:
            out.append(c)
        else:
            out.append("A")
            n += 1
    return "".join(out), n


def seq2kmer(seq, k=K):
    return [seq[i:i + k] for i in range(len(seq) - (k - 1))]


def kmer_ids(kmers):
    """List of k-mer strings -> clamped LongTensor of token ids."""
    ids = seqs_to_tens(kmers).long()
    return ids.clamp(max=MAX_REAL_TOKEN)


def tokenize(seq):
    """AA sequence -> LongTensor of k-mer token ids, length ``L-2``."""
    clean, n_bad = sanitize(seq)
    return kmer_ids(seq2kmer(clean)), n_bad


def add_bos(tok_2d):
    """``(B, Lk)`` -> ``(B, 1+Lk)`` with BOS prepended."""
    bos = torch.full((tok_2d.size(0), 1), BOS_ID, dtype=tok_2d.dtype,
                     device=tok_2d.device)
    return torch.cat([bos, tok_2d], dim=1)


def kmer_span(resid_idx, n_kmers):
    """K-mer token indices containing 0-indexed residue *resid_idx*.

    Token *j* spans residues ``[j, j+2]``, so masking ``[r-2, r]`` hides
    residue *r* completely.
    """
    lo = max(0, resid_idx - 2)
    hi = min(resid_idx, n_kmers - 1)
    return list(range(lo, hi + 1))


def kmer_hiding_window(j, n_kmers):
    """Token indices needed to hide all residues of token *j*: ``{j-2..j+2}``."""
    lo = max(0, j - 2)
    hi = min(j + 2, n_kmers - 1)
    return list(range(lo, hi + 1))


def parse_mutant(mutant_str):
    """``'A1P:D2N'`` -> ``[('A', 1, 'P'), ('D', 2, 'N')]`` (1-indexed)."""
    subs = []
    for part in str(mutant_str).split(":"):
        part = part.strip()
        if len(part) < 3:
            continue
        try:
            subs.append((part[0], int(part[1:-1]), part[-1]))
        except ValueError:
            continue
    return subs


def pad_positions(spans, device):
    """Pad span indices into a ``(B, P)`` LongTensor, +1 shifted past BOS."""
    spans = [list(s) for s in spans]
    max_p = max(len(s) for s in spans)
    rows = [[1 + j for j in s] + [0] * (max_p - len(s)) for s in spans]
    return torch.tensor(rows, dtype=torch.long, device=device)


# ---------------------------------------------------------------------- model

def load_model(ckpt_path, device, dtype=torch.bfloat16):
    """Load a BioT checkpoint (.ckpt or .safetensors)."""
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        model = BioTLightning()
        model.load_state_dict(load_file(ckpt_path), strict=False)
    else:
        model = BioTLightning.load_from_checkpoint(ckpt_path, map_location="cpu")
    model = model.to(dtype=dtype, device=device)
    model.eval()
    return model


@torch.no_grad()
def encoder_latents(model, tokens):
    """``tokens: (B, 1+Lk)`` -> ``z: (B, 1+Lk, C)``."""
    return model.forward(tokens, pred=True)


@torch.no_grad()
def mlm_logprobs_at(model, tokens, positions):
    """Log-softmax at specific positions only.

    Gathers before the vocab projection so we don't allocate a full
    ``(B, L, V)`` tensor for the 3 positions we actually need.
    """
    z = encoder_latents(model, tokens)
    b, p = positions.shape
    bidx = torch.arange(b, device=z.device).unsqueeze(1).expand(b, p)
    z_sel = z[bidx, positions]
    return F.log_softmax(model.mlm_head(z_sel).float(), dim=-1)


@torch.no_grad()
def embed_batch(model, tokens, pool="int_mean"):
    """``tokens: (B, 1+Lk)`` -> ``(B, D)`` float32 embeddings.

    Pooling modes: ``int_mean``, ``enc_mean``, ``enc_bos``, ``concat``.
    """
    if pool == "int_mean":
        zr = model.forward_int(tokens, pred=True)
        return zr.float().mean(dim=1)
    z = encoder_latents(model, tokens)
    if pool == "enc_mean":
        return z[:, 1:, :].float().mean(dim=1)
    if pool == "enc_bos":
        return z[:, 0, :].float()
    if pool == "concat":
        zr = model.forward_int(tokens, pred=True)
        return torch.cat([z[:, 0, :].float(), zr.float().mean(dim=1)], dim=-1)
    raise ValueError(f"unknown pool: {pool}")


# ------------------------------------------------------------------- batching

def tokenize_many(seqs, device):
    """Tokenize sequences that share a length (substitution benchmark)."""
    toks, bad = [], 0
    for s in seqs:
        ids, nb = tokenize(s)
        toks.append(ids)
        bad += nb
    lengths = {t.numel() for t in toks}
    if len(lengths) != 1:
        raise ValueError(
            f"Mixed sequence lengths {sorted(lengths)[:5]}... -- this harness "
            "assumes the substitution benchmark.")
    x = torch.stack(toks).to(device)
    return add_bos(x), bad


def embed_all(model, seqs, device, pool="int_mean", batch_size=32, verbose=True):
    """Embed all sequences, returning an ``(N, D)`` numpy array."""
    out = []
    total_bad = 0
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        tokens, bad = tokenize_many(chunk, device)
        total_bad += bad
        out.append(embed_batch(model, tokens, pool=pool).cpu().numpy())
        if verbose and (i // batch_size) % 25 == 0:
            print(f"    embedded {min(i + batch_size, len(seqs))}/{len(seqs)}", flush=True)
    if total_bad:
        print(f"    warning: {total_bad} non-standard residues coerced")
    return np.concatenate(out, axis=0)


# ------------------------------------------------------------ assay discovery

ASSAY_COLS = {"mutant", "mutated_sequence", "DMS_score"}


def _csv_header(path):
    """Read just the header row of a CSV."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            line = fh.readline()
    except OSError:
        return set()
    return {c.strip().strip('"').strip("'")
            for c in line.rstrip("\r\n").split(",")}


def _describe_dir(d, limit=12):
    """Summarise directory contents for debugging failed lookups."""
    try:
        entries = sorted(d.iterdir())
    except OSError as e:
        print(f"  cannot list {d}: {e}")
        return
    subdirs = [p for p in entries if p.is_dir()]
    exts = defaultdict(int)
    for p in entries:
        if p.is_file():
            exts[p.suffix.lower() or "(no extension)"] += 1
    print(f"  {d} holds {len(entries)} entries: {len(subdirs)} directories, "
          f"{sum(exts.values())} files")
    if subdirs:
        shown = ", ".join(p.name for p in subdirs[:limit])
        extra = f", ... (+{len(subdirs) - limit} more)" if len(subdirs) > limit else ""
        print(f"    directories: {shown}{extra}")
    if exts:
        print("    file types : " + ", ".join(
            f"{n}x {e}" for e, n in sorted(exts.items(), key=lambda x: -x[1])))


def find_assays(assay_dir, limit=None, require=ASSAY_COLS, validate=True):
    """Locate per-assay CSVs, searching recursively if needed."""
    d = Path(assay_dir).expanduser()

    if not d.exists():
        raise SystemExit(f"\n  --assay-dir does not exist: {d.resolve()}\n")

    if d.is_file():
        cands = [d]
    else:
        cands = sorted(d.glob("*.csv"))
        if not cands:
            cands = sorted(d.rglob("*.csv"))
            if cands:
                print(f"  note: no CSVs directly in {d}; found {len(cands)} "
                      f"nested one or more levels down")

    if not cands:
        print(f"\n  no .csv files anywhere under {d.resolve()}")
        _describe_dir(d)
        raise SystemExit("\n  Nothing to evaluate.\n")

    if not validate:
        return cands[:limit] if limit else cands

    good, bad = [], []
    for p in cands:
        (good if require <= _csv_header(p) else bad).append(p)

    if not good:
        print(f"\n  {len(cands)} CSV(s) found under {d.resolve()}, none carrying "
              f"the expected columns {sorted(require)}:")
        for p in cands[:3]:
            cols = sorted(_csv_header(p))
            print(f"    {p.name}: {cols[:10]}{' ...' if len(cols) > 10 else ''}")
        raise SystemExit(
            "\n  Re-run with --skip-header-check if columns are named differently.\n")

    if bad:
        print(f"  note: ignoring {len(bad)} CSV(s) without assay columns")
    sel = good[:limit] if limit else good
    tail = f"; using the first {len(sel)}" if len(sel) != len(good) else ""
    print(f"  {len(good)} assay CSVs{tail}")
    return sel


def find_reference(assay_dir, explicit=None):
    """Locate ``DMS_substitutions.csv`` for ``target_seq`` lookup."""
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            print(f"  reference: {p}")
            return p
        print(f"  warning: --reference-file {p} does not exist; ignoring")
        return None

    d = Path(assay_dir).expanduser()
    for root in (d, d.parent, d.parent.parent):
        for rel in ("DMS_substitutions.csv",
                    "reference/DMS_substitutions.csv",
                    "reference_files/DMS_substitutions.csv"):
            p = root / rel
            if p.exists():
                print(f"  reference: {p}")
                return p

    print("  no DMS_substitutions.csv nearby -- target_seq will be rebuilt "
          "from each assay's first variant.")
    return None


# --------------------------------------------------------- 1. MLM degeneracy

@torch.no_grad()
def cmd_mlm_check(args):
    device = args.device
    model = load_model(args.ckpt, device)

    assay_files = find_assays(args.assay_dir, limit=args.n_assays,
                              validate=not args.skip_header_check)

    rng = np.random.default_rng(0)
    stats = {m: {"correct": 0, "n": 0, "lp": []}
             for m in ("single", "window", "span")}

    for f in assay_files:
        df = pd.read_csv(f)
        if "mutated_sequence" not in df.columns:
            print(f"[{f.name}] skipped (no mutated_sequence column)")
            continue
        seqs = df["mutated_sequence"].dropna().unique().tolist()[: args.seqs_per_assay]
        print(f"[{f.name}] {len(seqs)} sequences")

        for s in seqs:
            clean, _ = sanitize(s)
            ids = kmer_ids(seq2kmer(clean))
            n_kmers = ids.numel()
            if n_kmers < 16:
                continue
            if n_kmers > args.max_kmers:
                ids = ids[: args.max_kmers]
                n_kmers = args.max_kmers
                clean = clean[: n_kmers + K - 1]

            targets = rng.choice(np.arange(4, n_kmers - 4),
                                 size=min(args.positions_per_seq, n_kmers - 8),
                                 replace=False)

            for mode in ("single", "window"):
                masked_rows, tgt_idx, tgt_ids = [], [], []
                for j in targets:
                    j = int(j)
                    row = ids.clone()
                    span = [j] if mode == "single" else kmer_hiding_window(j, n_kmers)
                    row[torch.tensor(span)] = args.mask_token_id
                    masked_rows.append(row)
                    tgt_idx.append(j)
                    tgt_ids.append(int(ids[j]))

                tokens = add_bos(torch.stack(masked_rows).to(device))
                pos = torch.tensor([[1 + j] for j in tgt_idx],
                                   dtype=torch.long, device=device)
                lp = mlm_logprobs_at(model, tokens, pos).cpu()
                for b, tid in enumerate(tgt_ids):
                    row_lp = lp[b, 0]
                    stats[mode]["lp"].append(float(row_lp[tid]))
                    stats[mode]["correct"] += int(row_lp.argmax().item() == tid)
                    stats[mode]["n"] += 1

            # 20-way span condition: the actual zero-shot scoring setup
            masked_rows, spans, res_idx = [], [], []
            for r in targets:
                r = int(r)
                if r + 2 >= len(clean):
                    continue
                sp = kmer_span(r, n_kmers)
                if len(sp) != K:
                    continue
                row = ids.clone()
                row[torch.tensor(sp)] = args.mask_token_id
                masked_rows.append(row)
                spans.append(sp)
                res_idx.append(r)

            if masked_rows:
                tokens = add_bos(torch.stack(masked_rows).to(device))
                pos = torch.tensor([[1 + j for j in sp] for sp in spans],
                                   dtype=torch.long, device=device)
                lp = mlm_logprobs_at(model, tokens, pos).cpu()

                for b, (sp, r) in enumerate(zip(spans, res_idx)):
                    cand = np.empty(len(AA20_LIST), dtype=np.float64)
                    for ai, aa in enumerate(AA20_LIST):
                        mseq = clean[:r] + aa + clean[r + 1:]
                        tok = kmer_ids([mseq[j:j + K] for j in sp])
                        cand[ai] = sum(float(lp[b, k, int(tok[k])])
                                       for k in range(len(sp)))
                    true_ai = AA20_LIST.index(clean[r])
                    post = cand - (cand.max() + np.log(np.exp(cand - cand.max()).sum()))
                    stats["span"]["lp"].append(float(post[true_ai]))
                    stats["span"]["correct"] += int(int(cand.argmax()) == true_ai)
                    stats["span"]["n"] += 1

    n_real = MAX_REAL_TOKEN + 1
    uni_tok = -np.log(n_real)
    uni_aa = -np.log(len(AA20_LIST))

    print("\n" + "=" * 70)
    print("MLM DEGENERACY CHECK")
    print("=" * 70)
    conditions = (
        ("single", "1-token mask  (neighbours leak all 3 residues)", uni_tok, n_real),
        ("window", "5-token mask  (residues actually hidden)", uni_tok, n_real),
        ("span", "3-token mask, 20-way  <-- THE ZERO-SHOT CONDITION", uni_aa, 20),
    )
    for mode, label, uni, k in conditions:
        s = stats[mode]
        acc = s["correct"] / max(s["n"], 1)
        mlp = float(np.mean(s["lp"])) if s["lp"] else float("nan")
        verdict = ("WORSE THAN CHANCE" if mlp < uni else
                   f"{np.exp(mlp - uni):.1f}x chance")
        print(f"  {label}")
        print(f"    top-1 accuracy : {acc:.4f}   (chance {1 / k:.4f}, n={s['n']})")
        print(f"    mean log-prob  : {mlp:+.4f}  (ppl {np.exp(-mlp):.1f}; "
              f"uniform {np.exp(-uni):.0f})  -> {verdict}")

    sa = stats["single"]["correct"] / max(stats["single"]["n"], 1)
    wa = stats["window"]["correct"] / max(stats["window"]["n"], 1)
    print("-" * 70)

    p_lp = (float(np.mean(stats["span"]["lp"])) if stats["span"]["lp"]
            else float("nan"))
    w_lp = float(np.mean(stats["window"]["lp"])) if stats["window"]["lp"] else 0.0

    if sa < 0.2:
        print("  TOKENIZER/MASK FAULT: even the leaky single-mask condition is")
        print("  near-chance. Check mask_token_id and the tokenizer.")
    else:
        print(f"  TOKENIZER: single {sa:.2f} vs window {wa:.2f}.")
        print("  This gap is a property of stride-1 k-mers, not the training scheme.")
        print()
        print("  RESIDUE PRIOR: the 20-way condition is what separates things.")
        if p_lp < uni_aa:
            deficit = np.exp(uni_aa - p_lp)
            print(f"    log-prob on truth {p_lp:+.2f} vs {uni_aa:+.2f} at chance "
                  f"-> {deficit:.1f}x LESS mass than chance.")
        else:
            print(f"    log-prob on truth {p_lp:+.2f} vs {uni_aa:+.2f} at chance "
                  f"-> {np.exp(p_lp - uni_aa):.2f}x MORE mass than chance.")
        if w_lp < uni_tok:
            print(f"    (5-token window still {uni_tok - w_lp:.2f} nats below uniform)")

    s = stats["span"]
    k_corr, n = s["correct"], max(s["n"], 1)
    bt = binomtest(k_corr, n, 0.05, alternative="greater")
    lift = (k_corr / n) / 0.05
    ci_lo, _ = bt.proportion_ci(confidence_level=0.95)

    print()
    print(f"  ZERO-SHOT VIABILITY   {k_corr}/{n} = {k_corr / n:.4f} vs 0.0500 chance")
    print(f"    binomial p = {bt.pvalue:.2e}   lift = {lift:.2f}x   "
          f"95% CI lower bound = {ci_lo:.4f}")
    print()
    if bt.pvalue >= 0.05:
        print("  -> INDISTINGUISHABLE FROM CHANCE. Skip the zeroshot pass;")
        print("     the supervised track is where the evidence has to come from.")
    elif lift < 2.0:
        print("  -> WEAK BUT REAL. Pilot ~20 assays before committing to all.")
    else:
        print("  -> VIABLE. Run the full pass.")
    print("=" * 70)


# ------------------------------------------------------ 2. zero-shot scoring

def check_wt_consistency(target_seq, df, n=200):
    """Check that the mutant string's WT residues match ``target_seq``."""
    checked = mismatched = 0
    for m in df["mutant"].head(n):
        for wt, pos, _ in parse_mutant(m):
            checked += 1
            if pos - 1 >= len(target_seq) or target_seq[pos - 1] != wt:
                mismatched += 1
    return checked, mismatched


@torch.no_grad()
def score_assay_zeroshot(model, df, target_seq, device, mask_id, batch_size=32):
    """Span-masked pseudo-log-likelihood ratio for one assay.

    After masking, mutant and WT inputs are identical (the differing tokens
    are all inside the masked span), so one forward pass per unique span
    serves all variants at that span.
    """
    clean_wt, _ = sanitize(target_seq)
    wt_ids = kmer_ids(seq2kmer(clean_wt))
    n_kmers = wt_ids.numel()

    by_span = defaultdict(list)
    n_skipped = 0
    for i, m in enumerate(df["mutant"].tolist()):
        subs = parse_mutant(m)
        if not subs or any(p < 1 or p > len(clean_wt) for _, p, _ in subs):
            n_skipped += 1
            continue
        span = tuple(sorted(set(chain.from_iterable(
            kmer_span(p - 1, n_kmers) for _, p, _ in subs))))
        if span:
            by_span[span].append(i)
        else:
            n_skipped += 1

    scores = np.full(len(df), np.nan)
    mut_seqs = df["mutated_sequence"].tolist()
    spans = list(by_span.keys())
    print(f"    {len(spans)} unique mask spans for {len(df)} variants"
          + (f" ({n_skipped} unparseable)" if n_skipped else ""))

    n_len_mismatch, len_examples = 0, []
    for i in range(0, len(spans), batch_size):
        chunk = spans[i:i + batch_size]
        rows = []
        for sp in chunk:
            r = wt_ids.clone()
            r[torch.tensor(sp)] = mask_id
            rows.append(r)
        tokens = add_bos(torch.stack(rows).to(device))
        pos = pad_positions(chunk, device)
        lp = mlm_logprobs_at(model, tokens, pos).cpu()

        for b, sp in enumerate(chunk):
            wt_ll = sum(float(lp[b, k, int(wt_ids[j])]) for k, j in enumerate(sp))
            for vi in by_span[sp]:
                mseq, _ = sanitize(mut_seqs[vi])
                if len(mseq) != len(clean_wt):
                    n_len_mismatch += 1
                    if len(len_examples) < 3:
                        len_examples.append(len(mseq))
                    continue
                mt_tok = kmer_ids([mseq[j:j + K] for j in sp])
                mt_ll = sum(float(lp[b, k, int(mt_tok[k])]) for k in range(len(sp)))
                scores[vi] = mt_ll - wt_ll

        if (i // batch_size) % 20 == 0:
            print(f"    span {min(i + batch_size, len(spans))}/{len(spans)}", flush=True)

    if n_len_mismatch:
        print(f"    WARNING: {n_len_mismatch}/{len(df)} variants dropped "
              f"(len mismatch vs target_seq={len(clean_wt)}). "
              f"Supply --reference-file to fix.")
    return scores


def cmd_zeroshot(args):
    device = args.device
    model = load_model(args.ckpt, device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = None
    ref_path = find_reference(args.assay_dir, args.reference_file)
    if ref_path is not None:
        ref = pd.read_csv(ref_path)
        if "DMS_id" in ref.columns:
            ref = ref.set_index("DMS_id")
        else:
            print(f"  warning: {ref_path.name} has no DMS_id column; ignoring it")
            ref = None

    assay_files = find_assays(args.assay_dir, limit=args.n_assays,
                              validate=not args.skip_header_check)

    results = []
    skipped = defaultdict(int)
    for f in assay_files:
        dms_id = f.stem
        out_csv = out_dir / f"{dms_id}.csv"
        if out_csv.exists() and not args.overwrite:
            print(f"[{dms_id}] skipped (already scored)")
            skipped["already scored"] += 1
            continue

        df = pd.read_csv(f)
        if "mutant" not in df.columns or "DMS_score" not in df.columns:
            print(f"[{dms_id}] skipped (missing columns)")
            skipped["missing columns"] += 1
            continue
        if args.limit and len(df) > args.limit:
            df = df.sample(args.limit, random_state=0).reset_index(drop=True)

        target_seq, seq_src = None, "reconstructed"
        if ref is not None and dms_id in ref.index and "target_seq" in ref.columns:
            target_seq = str(ref.loc[dms_id, "target_seq"])
            seq_src = "reference"
        if target_seq is None:
            row = df.iloc[0]
            s = list(str(row["mutated_sequence"]))
            for wt, pos, _ in parse_mutant(row["mutant"]):
                if pos - 1 < len(s):
                    s[pos - 1] = wt
            target_seq = "".join(s)

        if len(target_seq) > args.max_len:
            print(f"[{dms_id}] skipped (len {len(target_seq)} > --max-len "
                  f"{args.max_len})")
            skipped[f"longer than --max-len {args.max_len}"] += 1
            continue

        checked, mism = check_wt_consistency(target_seq, df)
        if mism:
            print(f"[{dms_id}] WARNING: {mism}/{checked} WT residues disagree "
                  f"with {seq_src} target_seq -- SKIPPING.")
            skipped[f"WT mismatch ({seq_src} target_seq)"] += 1
            continue

        print(f"[{dms_id}] {len(df)} variants, L={len(target_seq)}")
        try:
            scores = score_assay_zeroshot(model, df, target_seq, device,
                                          args.mask_token_id, args.batch_size)
        except Exception as e:
            print(f"[{dms_id}] FAILED: {e}")
            skipped["exception during scoring"] += 1
            continue

        ok = ~np.isnan(scores)
        if ok.sum() < 10:
            print(f"[{dms_id}] too few scored variants ({ok.sum()}/{len(df)})")
            skipped["fewer than 10 variants scored"] += 1
            continue
        rho = spearmanr(scores[ok], df["DMS_score"].values[ok]).correlation
        print(f"[{dms_id}] Spearman = {rho:.4f}  ({ok.sum()}/{len(df)} scored)")

        df_out = df.copy()
        df_out["model_score"] = scores
        df_out.to_csv(out_csv, index=False)
        results.append({"DMS_id": dms_id, "spearman": rho, "n": int(ok.sum()),
                        "n_variants": len(df), "target_seq": seq_src})

    summarize(results, out_dir / "summary_zeroshot.csv", "ZERO-SHOT",
              skipped=skipped, n_candidates=len(assay_files))


# ------------------------------------------------------- 3. supervised probe

def pick_fold_column(df, scheme):
    preferred = f"fold_{scheme}_5"
    if preferred in df.columns:
        return preferred
    candidates = [c for c in df.columns if c.startswith("fold_")]
    return candidates[0] if candidates else None


def cmd_supervised(args):
    device = args.device
    model = load_model(args.ckpt, device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    assay_files = find_assays(args.assay_dir, limit=args.n_assays,
                              validate=not args.skip_header_check)

    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    results = []
    skipped = defaultdict(int)

    for f in assay_files:
        dms_id = f.stem
        df = pd.read_csv(f)
        if "mutated_sequence" not in df.columns or "DMS_score" not in df.columns:
            print(f"[{dms_id}] skipped (missing columns)")
            skipped["missing columns"] += 1
            continue
        df = df.dropna(subset=["mutated_sequence", "DMS_score"]).reset_index(drop=True)
        if args.limit and len(df) > args.limit:
            df = df.sample(args.limit, random_state=0).reset_index(drop=True)
        if len(df) < 50:
            print(f"[{dms_id}] skipped (only {len(df)} variants)")
            skipped["fewer than 50 variants"] += 1
            continue
        seq_len = len(str(df.iloc[0]["mutated_sequence"]))
        if seq_len > args.max_len:
            print(f"[{dms_id}] skipped (L={seq_len} > --max-len {args.max_len})")
            skipped[f"longer than --max-len {args.max_len}"] += 1
            continue

        print(f"[{dms_id}] embedding {len(df)} variants (L={seq_len})...")
        try:
            X = embed_all(model, df["mutated_sequence"].tolist(), device,
                          pool=args.pool, batch_size=args.batch_size)
        except Exception as e:
            print(f"[{dms_id}] FAILED: {e}")
            skipped["exception during embedding"] += 1
            continue
        y = df["DMS_score"].values.astype(float)

        fold_col = pick_fold_column(df, args.fold_scheme)
        if fold_col:
            folds = df[fold_col].values
            splits = [(np.where(folds != k)[0], np.where(folds == k)[0])
                      for k in np.unique(folds)]
            print(f"    using folds: {fold_col}")
        else:
            print("    warning: no fold column found, using random 5-fold (seed 0)")
            splits = list(KFold(5, shuffle=True, random_state=0).split(X))

        rhos = []
        for tr, te in splits:
            if len(te) < 5:
                continue
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])

            n_in = max(int(0.8 * len(tr)), len(tr) - 1)
            best_a, best_r = alphas[0], -np.inf
            for a in alphas:
                m = Ridge(alpha=a).fit(Xtr[:n_in], y[tr][:n_in])
                p = m.predict(Xtr[n_in:])
                if len(p) < 5:
                    continue
                r = spearmanr(p, y[tr][n_in:]).correlation
                if r is not None and r > best_r:
                    best_r, best_a = r, a

            m = Ridge(alpha=best_a).fit(Xtr, y[tr])
            rho = spearmanr(m.predict(Xte), y[te]).correlation
            if rho is not None and not np.isnan(rho):
                rhos.append(rho)

        if not rhos:
            skipped["no usable CV folds"] += 1
            continue
        mean_rho = float(np.mean(rhos))
        print(f"[{dms_id}] Spearman = {mean_rho:.4f}  (folds: "
              + ", ".join(f"{r:.3f}" for r in rhos) + ")")
        results.append({"DMS_id": dms_id, "spearman": mean_rho,
                        "n": len(df), "dim": X.shape[1]})

    summarize(results, out_dir / "summary_supervised.csv", "SUPERVISED",
              skipped=skipped, n_candidates=len(assay_files))


# ------------------------------------------------------------------ reporting

def summarize(results, out_path, label, skipped=None, n_candidates=None):
    """Print summary statistics and write results CSV."""
    if not results:
        print("\n" + "=" * 62)
        print(f"{label}: no assays completed.")
        print("=" * 62)
        if skipped:
            print("  reasons:")
            for reason, n in sorted(skipped.items(), key=lambda x: -x[1]):
                print(f"    {n:>4}  {reason}")
        print("=" * 62)
        return
    res = pd.DataFrame(results).sort_values("spearman", ascending=False)
    res.to_csv(out_path, index=False)
    vals = res["spearman"].values

    rng = np.random.default_rng(0)
    boot = [np.mean(rng.choice(vals, size=len(vals), replace=True)) for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print("\n" + "=" * 62)
    print(f"{label} SUMMARY  ({len(res)} assays)")
    print("=" * 62)
    print(f"  mean Spearman   : {vals.mean():.4f}  [95% CI {lo:.4f}, {hi:.4f}]")
    print(f"  median Spearman : {np.median(vals):.4f}")
    print(f"  min / max       : {vals.min():.4f} / {vals.max():.4f}")
    print(f"  written to      : {out_path}")
    print("=" * 62)


# ----------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--ckpt", default=CKPT_PATH)
        sp.add_argument("--assay-dir", required=True)
        sp.add_argument("--skip-header-check", action="store_true")
        sp.add_argument("--n-assays", type=int, default=None)
        sp.add_argument("--max-len", type=int, default=1024)
        sp.add_argument("--batch-size", type=int, default=32)
        sp.add_argument("--mask-token-id", type=int, default=MASK_ID)
        sp.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    s1 = sub.add_parser("mlm-check", help="diagnose k-mer overlap leakage")
    common(s1)
    s1.add_argument("--seqs-per-assay", type=int, default=3)
    s1.add_argument("--positions-per-seq", type=int, default=16)
    s1.add_argument("--max-kmers", type=int, default=400)
    s1.set_defaults(func=cmd_mlm_check)

    s2 = sub.add_parser("zeroshot", help="span-masked zero-shot scoring")
    common(s2)
    s2.add_argument("--out", default="zs_scores")
    s2.add_argument("--limit", type=int, default=None)
    s2.add_argument("--reference-file", default=None)
    s2.add_argument("--overwrite", action="store_true")
    s2.set_defaults(func=cmd_zeroshot)

    s3 = sub.add_parser("supervised", help="embedding + ridge probe")
    common(s3)
    s3.add_argument("--out", default="sup_scores")
    s3.add_argument("--limit", type=int, default=None)
    s3.add_argument("--pool", default="int_mean",
                    choices=["int_mean", "enc_mean", "enc_bos", "concat"])
    s3.add_argument("--fold-scheme", default="random",
                    choices=["random", "modulo", "contiguous"])
    s3.set_defaults(func=cmd_supervised)

    args = p.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    print(f"device={args.device}  ckpt={args.ckpt}  mask_id={args.mask_token_id}")
    args.func(args)


if __name__ == "__main__":
    main()
