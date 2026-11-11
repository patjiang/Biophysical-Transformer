#!/usr/bin/env python
"""PLM-Interact PPI benchmark using BioT or ESM2 embeddings.

Embeds each protein, concatenates pair embeddings, trains a logistic
regression classifier, and reports AUPRC, AUROC, F1, and accuracy on
the held-out test split.

Setup (run once from repo root)::

    bash tests/ppi_bench/setup_ppi.sh

Tasks::

    python ppi_bench.py eval --task cross_species \\
        --ckpt ../../checkpoints/BioT_voxel_s0.safetensors

    python ppi_bench.py eval --task bernett \\
        --esm esm2_t33_650M_UR50D

    python ppi_bench.py eval --task virus_human \\
        --ckpt ../../checkpoints/BioT_voxel_s0.safetensors

    python ppi_bench.py eval --task all \\
        --ckpt ../../checkpoints/BioT_voxel_s0.safetensors

Datasets are CSV files with columns ``query``, ``text``, ``label``
downloaded by ``setup_ppi.sh`` into ``data/<task>/``.
"""

import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------ constants

CKPT_DEFAULT = "checkpoints/BioT_voxel_s0.safetensors"
DATA_DIR = Path("data")

TASKS = {
    "cross_species": {
        "train": "data/cross_species/train.csv",
        "val":   "data/cross_species/validation.csv",
        "test": [
            "data/cross_species/test_mouse.csv",
            "data/cross_species/test_fly.csv",
            "data/cross_species/test_worm.csv",
            "data/cross_species/test_yeast.csv",
            "data/cross_species/test_ecoli.csv",
        ],
    },
    "bernett": {
        "train": "data/bernett/train.csv",
        "val":   "data/bernett/validation.csv",
        "test":  "data/bernett/test.csv",
    },
    "virus_human": {
        "train": "data/virus_human/train.csv",
        "val":   "data/virus_human/validation.csv",
        "test":  "data/virus_human/test.csv",
    },
}


def log(s=""):
    print(s, flush=True)


# --------------------------------------------------------------- data loading

def load_pairs(csv_path):
    """Load (seqA, seqB, label) from a PLM-Interact CSV."""
    pairs = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row["query"], row["text"], int(row["label"])))
    return pairs


def unique_sequences(pair_list):
    """Collect all unique sequences from a list of (seqA, seqB, label)."""
    seqs = set()
    for a, b, _ in pair_list:
        seqs.add(a)
        seqs.add(b)
    return sorted(seqs)


# ------------------------------------------------------------- embedding cache

def embed_sequences(embedder, seqs, cache_path=None):
    """Embed all unique sequences, returning {seq: vector}.

    Caches to disk so re-runs skip embedding.
    """
    if cache_path and Path(cache_path).exists():
        log(f"  loading cached embeddings from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    log(f"  embedding {len(seqs)} unique sequences...")
    seq2vec = {}
    t0 = time.time()
    for i, seq in enumerate(seqs):
        emb, _ = embedder.per_residue(seq)
        if emb is not None:
            seq2vec[seq] = emb.mean(axis=0)
        else:
            seq2vec[seq] = np.zeros(1)  # placeholder

        if (i + 1) % 100 == 0 or i + 1 == len(seqs):
            elapsed = time.time() - t0
            log(f"    {i + 1}/{len(seqs)}  ({elapsed:.0f}s)")

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(seq2vec, f)
        log(f"  cached -> {cache_path}")

    return seq2vec


def pairs_to_features(pairs, seq2vec):
    """Convert pair list to (X, y) using concatenated embeddings."""
    X_rows, y_rows = [], []
    skipped = 0
    for a, b, label in pairs:
        va, vb = seq2vec.get(a), seq2vec.get(b)
        if va is None or vb is None or len(va) < 2 or len(vb) < 2:
            skipped += 1
            continue
        X_rows.append(np.concatenate([va, vb]))
        y_rows.append(label)
    if skipped:
        log(f"    skipped {skipped} pairs (missing embeddings)")
    return np.array(X_rows), np.array(y_rows)


# --------------------------------------------------------------- evaluation

def evaluate(y_true, y_prob, name=""):
    """Compute AUPRC, AUROC, best-F1, MCC, accuracy."""
    auprc = average_precision_score(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)

    # Best F1 threshold search
    thresholds = np.linspace(0, 1, 201)
    best_f1, best_thr = 0, 0.5
    for thr in thresholds:
        preds = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    preds_best = (y_prob >= best_thr).astype(int)
    mcc = matthews_corrcoef(y_true, preds_best)
    acc = (preds_best == y_true).mean()

    prefix = f"  [{name}] " if name else "  "
    log(f"{prefix}AUPRC={auprc:.4f}  AUROC={auroc:.4f}  "
        f"F1={best_f1:.4f}  MCC={mcc:.4f}  Acc={acc:.4f}  "
        f"(thr={best_thr:.2f}, n={len(y_true)}, "
        f"pos={y_true.sum()}/{len(y_true)})")

    return {
        "name": name, "auprc": auprc, "auroc": auroc,
        "f1": best_f1, "mcc": mcc, "acc": acc,
        "threshold": best_thr, "n": len(y_true),
        "n_pos": int(y_true.sum()),
    }


# ----------------------------------------------------------- subcommands

def run_task(task_name, embedder, model_name, args):
    """Train + evaluate on one task."""
    task = TASKS[task_name]
    log(f"\n{'=' * 60}")
    log(f"TASK: {task_name}  |  model: {model_name}")
    log(f"{'=' * 60}")

    # Find all CSVs that exist
    train_path = task["train"]
    if not Path(train_path).exists():
        log(f"  {train_path} not found, skipping")
        return []

    train_pairs = load_pairs(train_path)
    log(f"  train: {len(train_pairs)} pairs from {train_path}")

    # Collect all sequences across train + test
    test_paths = task["test"]
    if isinstance(test_paths, str):
        test_paths = [test_paths]
    test_paths = [p for p in test_paths if Path(p).exists()]

    all_pairs = list(train_pairs)
    test_sets = {}
    for tp in test_paths:
        tp_pairs = load_pairs(tp)
        test_sets[tp] = tp_pairs
        all_pairs.extend(tp_pairs)
        log(f"  test:  {len(tp_pairs)} pairs from {tp}")

    if not test_sets:
        log("  no test files found, skipping")
        return []

    all_seqs = unique_sequences(all_pairs)
    log(f"  {len(all_seqs)} unique sequences total")

    # Embed
    cache = f"probes/{task_name}_{model_name}_emb.pkl"
    seq2vec = embed_sequences(embedder, all_seqs, cache_path=cache)

    # Train
    X_train, y_train = pairs_to_features(train_pairs, seq2vec)
    log(f"  training: {X_train.shape[0]} pairs, D={X_train.shape[1]}")

    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(
        C=args.C, max_iter=2000, solver="lbfgs",
        class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(scaler.transform(X_train), y_train)

    train_auprc = average_precision_score(
        y_train, clf.predict_proba(scaler.transform(X_train))[:, 1])
    log(f"  train AUPRC: {train_auprc:.4f}")

    # Evaluate each test split
    results = []
    for tp in test_paths:
        tp_pairs = test_sets[tp]
        X_test, y_test = pairs_to_features(tp_pairs, seq2vec)
        if len(X_test) == 0:
            log(f"  [{Path(tp).stem}] no valid pairs")
            continue
        y_prob = clf.predict_proba(scaler.transform(X_test))[:, 1]
        r = evaluate(y_test, y_prob, name=Path(tp).stem)
        r["task"] = task_name
        r["model"] = model_name
        results.append(r)

    # Save probe
    probe_path = f"probes/{task_name}_{model_name}.pkl"
    Path(probe_path).parent.mkdir(parents=True, exist_ok=True)
    with open(probe_path, "wb") as f:
        pickle.dump({"scaler": scaler, "clf": clf, "model": model_name}, f)

    return results


def cmd_eval(args):
    """Run the benchmark."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from caid_disorder import Embedder

    embedder = Embedder(args)
    model_name = args.esm if args.esm else "BioT"

    tasks_to_run = list(TASKS.keys()) if args.task == "all" else [args.task]
    all_results = []

    for task_name in tasks_to_run:
        if task_name not in TASKS:
            log(f"Unknown task: {task_name}")
            continue
        results = run_task(task_name, embedder, model_name, args)
        all_results.extend(results)

    # Summary
    if all_results:
        log(f"\n{'=' * 70}")
        log(f"SUMMARY — {model_name}")
        log(f"{'=' * 70}")
        log(f"  {'task':<16} {'split':<16} {'AUPRC':>7} {'AUROC':>7} "
            f"{'F1':>7} {'MCC':>7} {'n':>8}")
        log(f"  {'-' * 16} {'-' * 16} {'-' * 7} {'-' * 7} "
            f"{'-' * 7} {'-' * 7} {'-' * 8}")
        for r in all_results:
            log(f"  {r['task']:<16} {r['name']:<16} {r['auprc']:>7.4f} "
                f"{r['auroc']:>7.4f} {r['f1']:>7.4f} {r['mcc']:>7.4f} "
                f"{r['n']:>8}")
        log(f"{'=' * 70}")

        # Write CSV
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{model_name}_results.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_results[0].keys())
            w.writeheader()
            w.writerows(all_results)
        log(f"  wrote {csv_path}")


# ------------------------------------------------------------------- CLI

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("eval", help="run PPI benchmark")
    s.add_argument("--task", default="all",
                   choices=["cross_species", "bernett", "virus_human", "all"])
    s.add_argument("--ckpt", default=CKPT_DEFAULT)
    s.add_argument("--esm", default=None, metavar="MODEL",
                   help="use ESM2 instead of BioT")
    s.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    s.add_argument("--max-tokens", type=int, default=1024)
    s.add_argument("--stride", type=int, default=512)
    s.add_argument("--layer", default="int", choices=["int", "enc"])
    s.add_argument("--C", type=float, default=1.0)
    s.add_argument("--out", default="results/")
    s.set_defaults(func=cmd_eval)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
