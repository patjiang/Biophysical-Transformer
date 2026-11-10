#!/usr/bin/env python
"""CAID intrinsic disorder prediction using BioT per-residue embeddings.

Subcommands::

    # Train a per-residue disorder probe on DisProt reference data
    python caid_disorder.py train \\
        --reference disprot_disorder_pdb.fasta \\
        --ckpt checkpoints/BioT_voxel_s0.safetensors \\
        --out probes/disorder.pkl

    # Predict disorder on CAID target sequences
    python caid_disorder.py predict \\
        --fasta caid_targets.fasta \\
        --ckpt checkpoints/BioT_voxel_s0.safetensors \\
        --probe probes/disorder.pkl \\
        --out predictions/BioT.caid

    # Cross-validate the probe (protein-level GroupKFold)
    python caid_disorder.py cv \\
        --reference disprot_disorder_pdb.fasta \\
        --ckpt checkpoints/BioT_voxel_s0.safetensors \\
        --folds 5

The reference FASTA follows the CAID format: three lines per protein
(``>accession``, sequence, per-residue labels ``0``/``1``/``-``).
Predictions are written in ``.caid`` format (tab-separated position,
residue, score, binary state).
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

from src.eval_utils import load_model, sanitize
from src.data import seqs_to_tens

# ------------------------------------------------------------------ constants

K = 3
BOS_ID = 8002
MAX_REAL_TOKEN = 7999

LABEL_MAP = {"0": 0.0, "1": 1.0, "-": np.nan}
AA20 = set("ACDEFGHIKLMNPQRSTVWY")


# ------------------------------------------------------------- tokenisation

def _kmer_ids(kmers):
    """k-mer strings -> clamped LongTensor of token ids."""
    return seqs_to_tens(kmers).long().clamp(max=MAX_REAL_TOKEN)


def _tokenize(seq):
    """AA string -> (1-D kmer_ids, sanitized sequence)."""
    clean, _ = sanitize(seq)
    kmers = [clean[i:i + K] for i in range(len(clean) - (K - 1))]
    return _kmer_ids(kmers), clean


def _add_bos_1d(tok):
    """(Lk,) -> (1, 1+Lk) with BOS prepended."""
    bos = torch.tensor([BOS_ID], dtype=tok.dtype)
    return torch.cat([bos, tok]).unsqueeze(0)


# ------------------------------------------------ per-residue embeddings

@torch.no_grad()
def _embed_window(model, tok_1d, device, layer="int"):
    """Embed one window of k-mer tokens.

    Returns ``(n_kmers, D)`` float32, BOS stripped.
    """
    tokens = _add_bos_1d(tok_1d).to(device)
    if layer == "int":
        # Interaction-block output: (B, n_kmers, D) — already sans BOS
        z = model.forward_int(tokens, pred=True)
    else:
        # Encoder output: (B, 1+n_kmers, D) — strip BOS
        z = model.forward(tokens, pred=True)
        z = z[:, 1:]
    return z[0].float().cpu()


def per_residue_embeddings(model, seq, device, max_tokens=1024,
                           stride=512, layer="int"):
    """BioT per-residue embeddings via k-mer averaging.

    For sequences longer than *max_tokens* k-mers, overlapping sliding
    windows are used and token embeddings from different windows are
    averaged before the k-mer-to-residue reduction.

    Returns ``(L, D)`` numpy array and sanitized sequence, or
    ``(None, clean)`` if the sequence is too short to tokenize.
    """
    tok, clean = _tokenize(seq)
    L = len(clean)
    n_kmers = tok.numel()

    if n_kmers <= 0:
        return None, clean

    # --- embed k-mer tokens (single pass or sliding window) ---
    if n_kmers <= max_tokens:
        z = _embed_window(model, tok, device, layer)
    else:
        D = None
        accum, counts = None, None
        for start in range(0, n_kmers, stride):
            end = min(start + max_tokens, n_kmers)
            w = _embed_window(model, tok[start:end], device, layer)
            if D is None:
                D = w.shape[1]
                accum = torch.zeros(n_kmers, D)
                counts = torch.zeros(n_kmers, 1)
            accum[start:end] += w
            counts[start:end] += 1
            if end == n_kmers:
                break
        z = accum / counts.clamp(min=1)

    # --- reduce k-mer embeddings to per-residue ---
    # Token j covers residues [j, j+K-1].  For dr in {0,1,2} the
    # slice ``z[:end]`` is added to ``res_z[dr:dr+end]``.
    D = z.shape[1]
    res_z = torch.zeros(L, D)
    res_c = torch.zeros(L)
    for dr in range(K):
        end = min(n_kmers, L - dr)
        res_z[dr:dr + end] += z[:end]
        res_c[dr:dr + end] += 1
    res_z /= res_c.unsqueeze(1).clamp(min=1)

    return res_z.numpy(), clean


# --------------------------------------------------------------- CAID I/O

def parse_reference_fasta(path):
    """Parse a CAID reference FASTA (>acc / sequence / labels).

    Returns list of ``(acc, seq, labels)`` where *labels* is a float
    array with NaN at excluded positions.
    """
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
                print(f"  warning: {acc} seq/label length mismatch "
                      f"({len(seq)} vs {len(lab)}), skipping")
                i += 3
                continue
            entries.append((acc, seq, lab))
            i += 3
        else:
            i += 1
    return entries


def parse_fasta(path):
    """Parse a plain multi-line FASTA. Returns list of ``(acc, seq)``."""
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
    """Write one protein block in ``.caid`` format to an open handle."""
    fh.write(f">{acc}\n")
    for i, (aa, sc) in enumerate(zip(seq, scores)):
        row = f"{i + 1}\t{aa}\t{sc:.3f}"
        if write_states:
            row += f"\t{1 if sc >= threshold else 0}"
        fh.write(row + "\n")


# ------------------------------------------------------------ embedding

def embed_labelled(model, entries, device, max_tokens=1024, stride=512,
                   layer="int"):
    """Embed reference entries, returning ``(X, y, groups)``.

    Only labelled positions (non-NaN) are kept.  *groups* holds the
    protein index for protein-level CV splits.
    """
    X_parts, y_parts, g_parts = [], [], []
    skipped = 0

    for idx, (acc, seq, labels) in enumerate(entries):
        emb, clean = per_residue_embeddings(
            model, seq, device, max_tokens=max_tokens,
            stride=stride, layer=layer)

        if emb is None:
            print(f"  {acc}: too short (L={len(seq)}), skipping")
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
        print(f"  skipped {skipped} proteins (too short or unlabelled)")

    return (np.concatenate(X_parts),
            np.concatenate(y_parts),
            np.concatenate(g_parts))


# ----------------------------------------------------------- subcommands

def cmd_train(args):
    """Train a per-residue logistic-regression disorder probe."""
    device = args.device
    model = load_model(args.ckpt, device)

    print(f"reference: {args.reference}")
    entries = parse_reference_fasta(args.reference)
    print(f"  {len(entries)} proteins with labels")

    print("embedding...")
    X, y, groups = embed_labelled(
        model, entries, device, max_tokens=args.max_tokens,
        stride=args.stride, layer=args.layer)

    n_pos = int(y.sum())
    print(f"  {len(y):,} labelled residues "
          f"({n_pos:,} disordered, {len(y) - n_pos:,} ordered)")

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    print("fitting logistic regression...")
    clf = LogisticRegression(
        C=args.C, max_iter=2000, solver="lbfgs",
        class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(Xs, y)

    auc = roc_auc_score(y, clf.predict_proba(Xs)[:, 1])
    print(f"  training AUC-ROC: {auc:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({
            "scaler": scaler, "clf": clf, "layer": args.layer,
            "max_tokens": args.max_tokens, "stride": args.stride,
            "C": args.C, "n_proteins": len(entries),
            "n_residues": len(y), "train_auc": auc,
        }, f)
    print(f"  saved -> {out}")


def cmd_predict(args):
    """Predict disorder on target sequences, write ``.caid`` output."""
    device = args.device
    model = load_model(args.ckpt, device)

    with open(args.probe, "rb") as f:
        probe = pickle.load(f)
    scaler, clf = probe["scaler"], probe["clf"]
    layer = probe.get("layer", "int")
    max_tokens = probe.get("max_tokens", 1024)
    stride = probe.get("stride", 512)
    print(f"probe: {args.probe}  (trained on {probe.get('n_proteins', '?')} "
          f"proteins, AUC {probe.get('train_auc', 0):.4f})")

    entries = parse_fasta(args.fasta)
    print(f"  {len(entries)} target sequences")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    threshold = args.threshold
    write_states = not args.no_states

    with open(out_path, "w") as fh:
        for i, (acc, seq) in enumerate(entries):
            emb, clean = per_residue_embeddings(
                model, seq, device, max_tokens=max_tokens,
                stride=stride, layer=layer)

            if emb is None:
                # Sequence too short — write zeros
                scores = np.zeros(len(clean))
            else:
                scores = clf.predict_proba(scaler.transform(emb))[:, 1]

            # Pad if sanitize truncated relative to original
            if len(scores) < len(clean):
                scores = np.pad(scores, (0, len(clean) - len(scores)))

            write_caid(fh, acc, clean, scores, threshold, write_states)

            if (i + 1) % 25 == 0 or i + 1 == len(entries):
                print(f"  predicted {i + 1}/{len(entries)}", flush=True)

    print(f"  wrote {len(entries)} predictions -> {out_path}")


def cmd_cv(args):
    """Protein-level cross-validation of the disorder probe."""
    device = args.device
    model = load_model(args.ckpt, device)

    print(f"reference: {args.reference}")
    entries = parse_reference_fasta(args.reference)
    print(f"  {len(entries)} proteins")

    print("embedding...")
    X, y, groups = embed_labelled(
        model, entries, device, max_tokens=args.max_tokens,
        stride=args.stride, layer=args.layer)

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
        aucs.append(auc)
        f1s.append(f1)
        mccs.append(mcc)

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
        sp.add_argument("--ckpt", default="checkpoints/BioT_voxel_s0.safetensors")
        sp.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
        sp.add_argument("--max-tokens", type=int, default=1024,
                        help="max k-mers per forward pass (default 1024)")
        sp.add_argument("--stride", type=int, default=512,
                        help="sliding-window stride for long sequences")
        sp.add_argument("--layer", default="int",
                        choices=["int", "enc"],
                        help="interaction block (default) or raw encoder")

    # train
    s1 = sub.add_parser("train", help="train disorder probe on DisProt data")
    common(s1)
    s1.add_argument("--reference", required=True,
                    help="CAID reference FASTA (>acc / seq / 0-1-dash labels)")
    s1.add_argument("--C", type=float, default=1.0,
                    help="logistic-regression inverse regularisation")
    s1.add_argument("--out", default="probes/disorder.pkl")
    s1.set_defaults(func=cmd_train)

    # predict
    s2 = sub.add_parser("predict", help="predict disorder -> .caid output")
    common(s2)
    s2.add_argument("--fasta", required=True,
                    help="target sequences (plain FASTA)")
    s2.add_argument("--probe", required=True,
                    help="trained probe (.pkl from `train`)")
    s2.add_argument("--threshold", type=float, default=0.5,
                    help="binary-state threshold (default 0.5; CAID "
                         "assessment can also find the optimal one)")
    s2.add_argument("--no-states", action="store_true",
                    help="omit the binary column (let CAID pick the "
                         "F1-optimal threshold)")
    s2.add_argument("--out", default="predictions/BioT.caid")
    s2.set_defaults(func=cmd_predict)

    # cv
    s3 = sub.add_parser("cv", help="cross-validate probe (protein-level)")
    common(s3)
    s3.add_argument("--reference", required=True)
    s3.add_argument("--C", type=float, default=1.0)
    s3.add_argument("--folds", type=int, default=5)
    s3.set_defaults(func=cmd_cv)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
