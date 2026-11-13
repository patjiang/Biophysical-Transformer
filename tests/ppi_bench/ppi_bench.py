#!/usr/bin/env python
"""PLM-Interact PPI benchmark using BioT or ESM2 embeddings.

Uses a cross-attention probe over per-residue embeddings: residues in
protein A attend to residues in protein B (and vice versa), then the
attended representations are pooled and classified. This tests whether
the embedding space captures residue-level interaction signals, not
just protein-level identity.

Setup (run once from repo root)::

    bash tests/ppi_bench/setup_ppi.sh

Tasks::

    python ppi_bench.py eval --task all \\
        --ckpt ../../checkpoints/BioT_voxel_s0.safetensors

    python ppi_bench.py eval --task all \\
        --esm esm2_t33_650M_UR50D

Datasets are CSV files with columns ``query``, ``text``, ``label``
downloaded by ``setup_ppi.sh`` into ``data/<task>/``.
"""

import argparse
import csv
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (average_precision_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ------------------------------------------------------------------ constants

CKPT_DEFAULT = "../../checkpoints/BioT_voxel_s0.safetensors"
DATA_DIR = Path("data")
MAX_RESIDUES = 512  # truncate proteins longer than this

TASKS = {
    "cross_species": {
        "train": "data/cross_species/train.csv",
        "val":   "data/cross_species/validation.csv",
        "test":  "data/cross_species/",
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
    seqs = set()
    for a, b, _ in pair_list:
        seqs.add(a)
        seqs.add(b)
    return sorted(seqs)


# ------------------------------------------------- per-residue embedding cache

def embed_sequences_residue(embedder, seqs, max_res=MAX_RESIDUES,
                            cache_path=None):
    """Embed unique sequences, returning {seq: (L, D) ndarray}.

    Incrementally updates a global cache: loads existing entries, embeds
    only new sequences, and saves back.  This lets multiple tasks share
    one cache per model so proteins seen in an earlier task aren't
    re-embedded.
    """
    seq2emb = {}

    # Load existing cache
    if cache_path and Path(cache_path).exists():
        log(f"  loading embedding cache from {cache_path}")
        with open(cache_path, "rb") as f:
            seq2emb = pickle.load(f)

    # Find sequences not yet cached
    needed = [s for s in seqs if s not in seq2emb]
    if not needed:
        log(f"  all {len(seqs)} sequences already cached")
        return seq2emb

    log(f"  {len(seqs)} sequences requested, {len(needed)} new to embed "
        f"(per-residue, max {max_res} residues)...")
    for seq in tqdm(needed, desc="  embedding", unit="seq"):
        emb, _ = embedder.per_residue(seq)
        if emb is not None:
            seq2emb[seq] = emb[:max_res].astype(np.float16)
        else:
            seq2emb[seq] = None

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(seq2emb, f)
        log(f"  cache updated -> {cache_path} "
            f"({len(seq2emb)} total sequences)")

    return seq2emb


# --------------------------------------------------------- pair dataset

class PPIDataset(Dataset):
    """Pairs of per-residue embeddings with binary labels."""

    def __init__(self, pairs, seq2emb):
        self.items = []
        skipped = 0
        for a, b, label in pairs:
            ea, eb = seq2emb.get(a), seq2emb.get(b)
            if ea is None or eb is None:
                skipped += 1
                continue
            self.items.append((ea, eb, label))
        if skipped:
            log(f"    skipped {skipped} pairs (missing embeddings)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ea, eb, label = self.items[idx]
        return (torch.from_numpy(ea.astype(np.float32)),
                torch.from_numpy(eb.astype(np.float32)),
                torch.tensor(label, dtype=torch.float32))


def collate_ppi(batch):
    """Pad variable-length per-residue embeddings and build masks."""
    embs_a, embs_b, labels = zip(*batch)
    D = embs_a[0].shape[1]

    def pad_stack(tensors):
        max_len = max(t.shape[0] for t in tensors)
        padded = torch.zeros(len(tensors), max_len, D)
        mask = torch.zeros(len(tensors), max_len, dtype=torch.bool)
        for i, t in enumerate(tensors):
            L = t.shape[0]
            padded[i, :L] = t
            mask[i, :L] = True
        return padded, mask

    a_pad, a_mask = pad_stack(embs_a)
    b_pad, b_mask = pad_stack(embs_b)
    labels = torch.stack(labels)
    return a_pad, a_mask, b_pad, b_mask, labels


# --------------------------------------------------- cross-attention probe

class CrossAttentionBlock(nn.Module):
    """Single-head cross-attention: Q from one protein, K/V from the other."""

    def __init__(self, d_in, d_attn=64):
        super().__init__()
        self.d_attn = d_attn
        self.W_q = nn.Linear(d_in, d_attn, bias=False)
        self.W_k = nn.Linear(d_in, d_attn, bias=False)
        self.W_v = nn.Linear(d_in, d_attn, bias=False)

    def forward(self, query, key_value, kv_mask=None):
        """
        query:    (B, Lq, D)
        key_value: (B, Lk, D)
        kv_mask:  (B, Lk) bool — True = real token
        Returns:  (B, Lq, d_attn)
        """
        Q = self.W_q(query)            # (B, Lq, d_attn)
        K = self.W_k(key_value)        # (B, Lk, d_attn)
        V = self.W_v(key_value)        # (B, Lk, d_attn)

        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_attn)

        if kv_mask is not None:
            # Mask out padding positions in K/V
            scores = scores.masked_fill(~kv_mask.unsqueeze(1), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # all-pad rows
        return torch.bmm(attn, V)


class PPICrossAttnProbe(nn.Module):
    """Cross-attention probe for PPI prediction.

    A attends to B, B attends to A (bidirectional cross-attention).
    Attended residues are mean-pooled, concatenated, and classified.
    """

    def __init__(self, d_emb, d_attn=64, d_hidden=128, dropout=0.3):
        super().__init__()
        self.xattn_a2b = CrossAttentionBlock(d_emb, d_attn)
        self.xattn_b2a = CrossAttentionBlock(d_emb, d_attn)
        self.head = nn.Sequential(
            nn.Linear(d_attn * 2, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, a, a_mask, b, b_mask):
        """
        a: (B, La, D), a_mask: (B, La)
        b: (B, Lb, D), b_mask: (B, Lb)
        Returns: (B,) logits
        """
        # Cross-attention
        a2b = self.xattn_a2b(a, b, b_mask)   # (B, La, d_attn)
        b2a = self.xattn_b2a(b, a, a_mask)   # (B, Lb, d_attn)

        # Masked mean pool
        a2b = (a2b * a_mask.unsqueeze(-1)).sum(1) / a_mask.sum(1, keepdim=True).clamp(1)
        b2a = (b2a * b_mask.unsqueeze(-1)).sum(1) / b_mask.sum(1, keepdim=True).clamp(1)

        combined = torch.cat([a2b, b2a], dim=-1)  # (B, 2 * d_attn)
        return self.head(combined).squeeze(-1)


# ----------------------------------------------------------- training loop

def train_crossattn(train_ds, val_ds, d_emb, device,
                    d_attn=64, d_hidden=128, dropout=0.3,
                    lr=5e-4, epochs=30, batch_size=64,
                    patience=6, num_workers=0):
    """Train the cross-attention probe, return (model, best_val_auprc)."""
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_ppi, num_workers=num_workers,
                              drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_ppi, num_workers=num_workers,
                            pin_memory=True)

    model = PPICrossAttnProbe(d_emb, d_attn=d_attn, d_hidden=d_hidden,
                               dropout=dropout).to(device)

    # Compute pos_weight from training labels
    n_pos = sum(1 for _, _, l in train_ds.items if l == 1)
    n_neg = len(train_ds) - n_pos
    pw = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=2)

    best_auprc, best_state, wait = 0, None, 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"    epoch {epoch + 1:>2}",
                    leave=False, unit="batch")
        for a, am, b, bm, labels in pbar:
            a, am = a.to(device), am.to(device)
            b, bm = b.to(device), bm.to(device)
            labels = labels.to(device)

            opt.zero_grad()
            logits = model(a, am, b, bm)
            loss = loss_fn(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validate
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for a, am, b, bm, labels in val_loader:
                a, am = a.to(device), am.to(device)
                b, bm = b.to(device), bm.to(device)
                probs = torch.sigmoid(model(a, am, b, bm)).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())

        val_auprc = average_precision_score(all_labels, all_probs)
        scheduler.step(val_auprc)

        avg_loss = epoch_loss / max(len(train_loader), 1)
        log(f"    epoch {epoch + 1:>2}: loss={avg_loss:.4f}  "
            f"val_AUPRC={val_auprc:.4f}  "
            f"lr={opt.param_groups[0]['lr']:.1e}")

        if val_auprc > best_auprc:
            best_auprc = val_auprc
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                log(f"    early stop at epoch {epoch + 1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    log(f"  cross-attn trained: best val AUPRC={best_auprc:.4f}")
    return model, best_auprc


# --------------------------------------------------------------- evaluation

def evaluate(model, dataset, device, name="", batch_size=64,
             save_preds=None):
    """Compute AUPRC, AUROC and optionally save per-pair predictions."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_ppi, pin_memory=True)
    all_probs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for a, am, b, bm, labels in loader:
            a, am = a.to(device), am.to(device)
            b, bm = b.to(device), bm.to(device)
            probs = torch.sigmoid(model(a, am, b, bm)).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)

    auprc = average_precision_score(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)

    prefix = f"  [{name}] " if name else "  "
    log(f"{prefix}AUPRC={auprc:.4f}  AUROC={auroc:.4f}  "
        f"(n={len(y_true)}, pos={int(y_true.sum())}/{len(y_true)})")

    # Save raw predictions for paired bootstrap analysis
    if save_preds:
        np.savez(save_preds, probs=y_prob, labels=y_true)
        log(f"  predictions -> {save_preds}")

    return {
        "name": name, "auprc": auprc, "auroc": auroc,
        "n": len(y_true), "n_pos": int(y_true.sum()),
    }


# ----------------------------------------------------------- subcommands

def run_task(task_name, embedder, model_name, args):
    """Train + evaluate on one task."""
    task = TASKS[task_name]
    log(f"\n{'=' * 60}")
    log(f"TASK: {task_name}  |  model: {model_name}")
    log(f"{'=' * 60}")

    train_path = task["train"]
    if not Path(train_path).exists():
        log(f"  {train_path} not found, skipping")
        return []

    train_pairs = load_pairs(train_path)
    log(f"  train: {len(train_pairs)} pairs from {train_path}")

    val_path = task.get("val")
    val_pairs = []
    if val_path and Path(val_path).exists():
        val_pairs = load_pairs(val_path)
        log(f"  val:   {len(val_pairs)} pairs from {val_path}")

    test_spec = task["test"]
    if isinstance(test_spec, str) and Path(test_spec).is_dir():
        test_paths = sorted(str(p) for p in Path(test_spec).glob("test*.csv"))
    elif isinstance(test_spec, str):
        test_paths = [test_spec] if Path(test_spec).exists() else []
    else:
        test_paths = [p for p in test_spec if Path(p).exists()]

    all_pairs = list(train_pairs) + list(val_pairs)
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

    # Embed (per-residue) — global cache per model, shared across tasks
    cache = f"probes/{model_name}_embeddings.pkl"
    seq2emb = embed_sequences_residue(embedder, all_seqs,
                                       max_res=args.max_residues,
                                       cache_path=cache)

    # Determine embedding dimension
    d_emb = None
    for v in seq2emb.values():
        if v is not None:
            d_emb = v.shape[1]
            break
    log(f"  D={d_emb}, max_residues={args.max_residues}")

    # Build datasets
    train_ds = PPIDataset(train_pairs, seq2emb)
    log(f"  train dataset: {len(train_ds)} pairs")

    if val_pairs:
        val_ds = PPIDataset(val_pairs, seq2emb)
    else:
        n_val = max(int(0.1 * len(train_ds)), 100)
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(train_ds))
        val_items = [train_ds.items[i] for i in idx[:n_val]]
        train_ds.items = [train_ds.items[i] for i in idx[n_val:]]
        val_ds = PPIDataset.__new__(PPIDataset)
        val_ds.items = val_items
        log(f"  split off {n_val} pairs for validation")

    log(f"  val dataset: {len(val_ds)} pairs")

    # Train (or load existing probe)
    device = args.device
    probe_path = f"probes/{task_name}_{model_name}_xattn.pt"

    if Path(probe_path).exists() and not getattr(args, "retrain", False):
        log(f"  loading existing probe from {probe_path}")
        ckpt = torch.load(probe_path, map_location="cpu", weights_only=True)
        model = PPICrossAttnProbe(
            d_emb, d_attn=ckpt.get("d_attn", args.d_attn),
            d_hidden=ckpt.get("d_hidden", args.hidden)).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        log(f"  skipped training (use --retrain to force)")
    else:
        model, val_auprc = train_crossattn(
            train_ds, val_ds, d_emb, device,
            d_attn=args.d_attn, d_hidden=args.hidden,
            dropout=args.dropout, lr=args.lr,
            epochs=args.epochs, batch_size=args.batch_size,
            patience=args.patience)

    # Evaluate
    results = []
    for tp in test_paths:
        test_ds = PPIDataset(test_sets[tp], seq2emb)
        if len(test_ds) == 0:
            log(f"  [{Path(tp).stem}] no valid pairs")
            continue
        pred_file = str(Path(args.out) /
                        f"{task_name}_{Path(tp).stem}_preds.npz")
        r = evaluate(model, test_ds, device, name=Path(tp).stem,
                     batch_size=args.batch_size, save_preds=pred_file)
        r["task"] = task_name
        r["model"] = model_name
        results.append(r)

    # Save probe
    Path(probe_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "d_emb": d_emb, "d_attn": args.d_attn, "d_hidden": args.hidden,
        "model": model_name,
    }, probe_path)
    log(f"  saved probe -> {probe_path}")

    return results


def cmd_eval(args):
    """Run the benchmark."""
    # caid_disorder.py lives in tests/ (one level up)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from caid_disorder import Embedder

    embedder = Embedder(args)
    if args.esm:
        model_name = args.esm
    else:
        # Distinguish voxel / novox by checkpoint filename
        stem = Path(args.ckpt).stem          # e.g. "BioT_voxel_s0"
        model_name = stem.rsplit("_s", 1)[0] if "_s" in stem else stem

    tasks_to_run = list(TASKS.keys()) if args.task == "all" else [args.task]
    all_results = []

    for task_name in tasks_to_run:
        if task_name not in TASKS:
            log(f"Unknown task: {task_name}")
            continue
        results = run_task(task_name, embedder, model_name, args)
        all_results.extend(results)

    if all_results:
        log(f"\n{'=' * 70}")
        log(f"SUMMARY — {model_name}")
        log(f"{'=' * 70}")
        log(f"  {'task':<16} {'split':<16} {'AUPRC':>7} {'AUROC':>7} {'n':>8}")
        log(f"  {'-' * 16} {'-' * 16} {'-' * 7} {'-' * 7} {'-' * 8}")
        for r in all_results:
            log(f"  {r['task']:<16} {r['name']:<16} {r['auprc']:>7.4f} "
                f"{r['auroc']:>7.4f} {r['n']:>8}")
        log(f"{'=' * 70}")

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
    s.add_argument("--max-residues", type=int, default=MAX_RESIDUES,
                   help="truncate proteins to this many residues")
    s.add_argument("--d-attn", type=int, default=64,
                   help="cross-attention dimension")
    s.add_argument("--hidden", type=int, default=128,
                   help="classification head hidden dim")
    s.add_argument("--dropout", type=float, default=0.3)
    s.add_argument("--lr", type=float, default=5e-4)
    s.add_argument("--epochs", type=int, default=30)
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--patience", type=int, default=6)
    s.add_argument("--retrain", action="store_true",
                   help="force re-training even if probe exists")
    s.add_argument("--out", default="results/")
    s.set_defaults(func=cmd_eval)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
