#!/usr/bin/env python
"""Calculate sequence overlap between model training sets and the CAID IDR dataset.

Usage::

    # Check overlap of a training FASTA against CAID reference
    python dataset_overlap.py \\
        --train training_sequences.fasta \\
        --caid CAID/demo-data/references/disorder_pdb.fasta \\
        --threshold 0.80 \\
        --out overlap_report.csv

    # Multiple training sets at once
    python dataset_overlap.py \\
        --train esm2_train.fasta biot_train.fasta \\
        --caid CAID/demo-data/references/disorder_pdb.fasta \\
        --out overlap_report.csv

For each training FASTA, reports how many CAID proteins appear in
(or are highly similar to) the training set. Overlap is measured by
normalised edit distance (1 - edit_distance / max_len) and by exact
substring containment.

The ``--hook`` flag accepts a Python script that defines a
``get_training_sequences(path) -> list[str]`` function, for training
sets stored in non-FASTA formats (HDF5, npz, etc.).
"""

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# --------------------------------------------------------------- parsing

def parse_fasta(path):
    """Parse a multi-line FASTA. Returns list of ``(acc, seq)``."""
    entries = []
    with open(path) as f:
        acc, parts = None, []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if acc is not None:
                    entries.append((acc, "".join(parts).upper()))
                acc = line[1:].split()[0]
                parts = []
            elif acc is not None and line:
                parts.append(line)
        if acc is not None:
            entries.append((acc, "".join(parts).upper()))
    return entries


def parse_caid_reference(path):
    """Parse CAID reference FASTA (>acc / seq / labels).

    Returns list of ``(acc, seq)``.
    """
    entries = []
    label_map = set("01-")
    with open(path) as f:
        lines = [l.rstrip("\n\r") for l in f
                 if l.strip() and not l.startswith("#")]

    i = 0
    while i < len(lines):
        if lines[i].startswith(">"):
            acc = lines[i][1:].split()[0]
            seq = lines[i + 1]
            # Skip the label line if present
            if i + 2 < len(lines) and set(lines[i + 2]) <= label_map:
                i += 3
            else:
                i += 2
            entries.append((acc, seq.upper()))
        else:
            i += 1
    return entries


# -------------------------------------------------------------- overlap

def seq_hash(seq):
    return hashlib.sha256(seq.encode()).hexdigest()


def exact_match_overlap(train_seqs, caid_entries):
    """Exact sequence match."""
    train_hashes = {seq_hash(s) for s in train_seqs}
    hits = []
    for acc, seq in caid_entries:
        if seq_hash(seq) in train_hashes:
            hits.append(acc)
    return set(hits)


def substring_overlap(train_seqs, caid_entries, min_len=30):
    """Check if any CAID protein is a substring of a training sequence
    or vice versa (above min_len)."""
    hits = defaultdict(str)
    # Build set of long training sequences for containment check
    for acc, seq in caid_entries:
        if len(seq) < min_len:
            continue
        for ts in train_seqs:
            if seq in ts:
                hits[acc] = "caid_in_train"
                break
            if ts in seq and len(ts) >= min_len:
                hits[acc] = "train_in_caid"
                break
    return hits


def identity_overlap(train_seqs, caid_entries, threshold=0.80):
    """Approximate identity by k-mer Jaccard as a fast pre-filter,
    then exact normalised identity for candidates above threshold.

    Returns ``{acc: best_identity}`` for CAID proteins exceeding
    the threshold.
    """
    K = 6

    # Index training sequences by k-mer sets
    print(f"  indexing {len(train_seqs)} training sequences (k={K})...")
    train_kmer_sets = []
    for seq in train_seqs:
        kmers = {seq[i:i + K] for i in range(len(seq) - K + 1)}
        train_kmer_sets.append(kmers)

    hits = {}
    for idx, (acc, cseq) in enumerate(caid_entries):
        if len(cseq) < K:
            continue
        c_kmers = {cseq[i:i + K] for i in range(len(cseq) - K + 1)}

        best_jaccard = 0
        best_train_idx = -1
        for ti, t_kmers in enumerate(train_kmer_sets):
            if not t_kmers:
                continue
            jacc = len(c_kmers & t_kmers) / len(c_kmers | t_kmers)
            if jacc > best_jaccard:
                best_jaccard = jacc
                best_train_idx = ti

        # Only compute expensive alignment for promising candidates
        if best_jaccard >= threshold * 0.5:
            tseq = train_seqs[best_train_idx]
            ident = _fast_identity(cseq, tseq)
            if ident >= threshold:
                hits[acc] = ident

        if (idx + 1) % 100 == 0:
            print(f"  checked {idx + 1}/{len(caid_entries)}", flush=True)

    return hits


def _fast_identity(s1, s2):
    """Normalised identity: matching positions / max(len(s1), len(s2)).

    For sequences of similar length, aligns by sliding the shorter
    across the longer and taking the best window. For very different
    lengths, returns the best local match fraction.
    """
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    if len(s1) == 0:
        return 0.0

    # If same length, just count matches
    if len(s1) == len(s2):
        matches = sum(a == b for a, b in zip(s1, s2))
        return matches / len(s1)

    # Slide shorter across longer
    best = 0
    for offset in range(len(s2) - len(s1) + 1):
        matches = sum(a == b for a, b in zip(s1, s2[offset:offset + len(s1)]))
        best = max(best, matches)
    return best / max(len(s1), len(s2))


# ----------------------------------------------------------------- main

def cmd_overlap(args):
    # Parse CAID reference
    caid = parse_caid_reference(args.caid)
    print(f"CAID reference: {len(caid)} proteins from {args.caid}")

    rows = []

    for train_path in args.train:
        train_path = Path(train_path)
        name = train_path.stem
        print(f"\n{'=' * 60}")
        print(f"Training set: {name} ({train_path})")
        print(f"{'=' * 60}")

        # Load training sequences
        if args.hook:
            import importlib.util
            spec = importlib.util.spec_from_file_location("hook", args.hook)
            hook = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hook)
            train_seqs = hook.get_training_sequences(str(train_path))
        else:
            train_entries = parse_fasta(str(train_path))
            train_seqs = [seq for _, seq in train_entries]

        print(f"  {len(train_seqs)} training sequences loaded")

        # Exact matches
        exact = exact_match_overlap(train_seqs, caid)
        print(f"  exact matches: {len(exact)}/{len(caid)} "
              f"({len(exact) / len(caid):.1%})")

        # Substring containment
        substr = substring_overlap(train_seqs, caid)
        print(f"  substring containment: {len(substr)}/{len(caid)} "
              f"({len(substr) / len(caid):.1%})")

        # Identity threshold
        ident = identity_overlap(train_seqs, caid, threshold=args.threshold)
        print(f"  identity >= {args.threshold:.0%}: {len(ident)}/{len(caid)} "
              f"({len(ident) / len(caid):.1%})")

        # Combined
        all_overlap = exact | set(substr.keys()) | set(ident.keys())
        print(f"  any overlap: {len(all_overlap)}/{len(caid)} "
              f"({len(all_overlap) / len(caid):.1%})")

        for acc, seq in caid:
            row = {
                "caid_acc": acc,
                "train_set": name,
                "seq_len": len(seq),
                "exact_match": acc in exact,
                "substring": substr.get(acc, ""),
                "best_identity": ident.get(acc, 0.0),
                "overlaps": acc in all_overlap,
            }
            rows.append(row)

    # Write report
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(out, index=False)
        print(f"\nwrote {out} ({len(df)} rows)")

    # Summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'training set':<25} {'exact':>6} {'substr':>7} "
          f"{'ident':>6} {'any':>6} {'total':>6}")
    print(f"  {'-' * 25} {'-' * 6} {'-' * 7} {'-' * 6} {'-' * 6} {'-' * 6}")
    for train_path in args.train:
        name = Path(train_path).stem
        subset = [r for r in rows if r["train_set"] == name]
        n_exact = sum(r["exact_match"] for r in subset)
        n_substr = sum(bool(r["substring"]) for r in subset)
        n_ident = sum(r["best_identity"] >= args.threshold for r in subset)
        n_any = sum(r["overlaps"] for r in subset)
        print(f"  {name:<25} {n_exact:>6} {n_substr:>7} "
              f"{n_ident:>6} {n_any:>6} {len(caid):>6}")


try:
    import pandas as pd
except ImportError:
    pd = None


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", nargs="+", required=True,
                   help="training set FASTA file(s)")
    p.add_argument("--caid", required=True,
                   help="CAID reference FASTA")
    p.add_argument("--threshold", type=float, default=0.80,
                   help="sequence identity threshold (default 0.80)")
    p.add_argument("--hook", default=None,
                   help="Python script defining get_training_sequences(path)")
    p.add_argument("--out", default="overlap_report.csv")
    args = p.parse_args()

    if pd is None and args.out:
        print("warning: pandas not available, skipping CSV output")
        args.out = None

    cmd_overlap(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
