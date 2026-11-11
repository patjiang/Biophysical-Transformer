#!/usr/bin/env python
"""Plot CAID benchmark results and embedding PCA scatterplots.

Subcommands::

    # Bar chart comparing methods on CAID metrics
    python plot_caid.py metrics \\
        --csv results/disorder_pdb.analysis.all.dataset.default.metrics.csv \\
        --out figures/caid_metrics.png

    # PCA of mean-pooled protein embeddings, one panel per predictor
    python plot_caid.py pca \\
        --reference CAID/demo-data/references/disorder_pdb.fasta \\
        --predictions predictions/ \\
        --ckpt ../../checkpoints/BioT_voxel_s0.safetensors \\
        --out figures/pca_proteins.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ metrics

HIGHLIGHT = ["aucroc", "f1s", "mcc", "ppv", "tpr"]
NICE = {"aucroc": "AUC-ROC", "f1s": "F1", "mcc": "MCC",
        "ppv": "Precision", "tpr": "Recall", "aucpr": "AUC-PR",
        "bac": "Balanced Acc", "inf": "Informedness"}


def cmd_metrics(args):
    df = pd.read_csv(args.csv, index_col=0)
    cols = args.metrics.split(",") if args.metrics else HIGHLIGHT
    cols = [c for c in cols if c in df.columns]

    fig, ax = plt.subplots(figsize=(1.8 * len(cols), 0.9 * len(df) + 1.2))
    x = np.arange(len(cols))
    h = 0.8 / len(df)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(df), 3)))

    for i, (method, row) in enumerate(df.iterrows()):
        vals = [row[c] for c in cols]
        offset = (i - len(df) / 2 + 0.5) * h
        bars = ax.barh(x + offset, vals, height=h, label=method,
                       color=colors[i], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=7)

    ax.set_yticks(x)
    ax.set_yticklabels([NICE.get(c, c) for c in cols])
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Score")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_title("CAID Disorder-PDB", fontweight="bold", fontsize=11)
    ax.invert_yaxis()
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out}")


# -------------------------------------------------------- .caid parsing

def parse_caid_scores(path):
    """Parse a .caid prediction file.

    Returns ``{accession: mean_score}`` — the mean per-residue disorder
    score for each protein.
    """
    scores = {}
    acc, vals = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if acc is not None and vals:
                    scores[acc] = np.mean(vals)
                acc = line[1:].split()[0]
                vals = []
            elif acc is not None and line:
                parts = line.split("\t")
                if len(parts) >= 3:
                    try:
                        vals.append(float(parts[2]))
                    except ValueError:
                        pass
    if acc is not None and vals:
        scores[acc] = np.mean(vals)
    return scores


# ------------------------------------------------------------------ PCA

def _embed_proteins_with(embedder, entries):
    """Mean-pool per-residue embeddings to one vector per protein."""
    vecs, pcts, accs = [], [], []
    for idx, (acc, seq, labels) in enumerate(entries):
        emb, clean = embedder.per_residue(seq)
        if emb is None:
            continue
        L = min(len(clean), len(labels), emb.shape[0])
        lab = labels[:L]
        mask = ~np.isnan(lab)
        if mask.sum() == 0:
            continue
        vecs.append(emb[:L].mean(axis=0))
        pcts.append(lab[mask].sum() / mask.sum())
        accs.append(acc)
        if (idx + 1) % 50 == 0 or idx + 1 == len(entries):
            print(f"  embedded {idx + 1}/{len(entries)}", flush=True)
    return np.stack(vecs), np.array(pcts), accs


def _reduce(X, n_pca=None):
    """PCA denoise then UMAP, using pre-PCA distances for the neighbour graph.

    Returns ``(xy, info_str)`` where *xy* is ``(N, 2)``.
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics import pairwise_distances
    from umap import UMAP

    # Pre-PCA pairwise distances (original space)
    D_orig = pairwise_distances(X, metric="euclidean")

    # PCA to a shared intermediate dimensionality
    if n_pca is None:
        n_pca = min(50, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=n_pca).fit(X)
    X_pca = pca.transform(X)
    var_kept = pca.explained_variance_ratio_.sum()

    # UMAP init from first two PCs (stable, reproducible)
    init = X_pca[:, :2] / (X_pca[:, :2].std(axis=0, keepdims=True) + 1e-8) * 1e-4

    # UMAP with precomputed distances from the original space
    reducer = UMAP(n_components=2, metric="precomputed",
                   n_neighbors=min(15, X.shape[0] - 1),
                   min_dist=0.3, init=init, random_state=42)
    xy = reducer.fit_transform(D_orig)

    info = f"PCA({n_pca}, {var_kept:.0%} var) → UMAP"
    return xy, info


def _scatter_panel(ax, xy, pct, title, info):
    """Draw one UMAP scatter panel coloured by % disorder."""
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=pct, cmap="RdYlBu_r",
                    vmin=0, vmax=1, s=18, alpha=0.8,
                    linewidths=0, rasterized=True)
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.text(0.02, 0.02, info, transform=ax.transAxes,
            fontsize=7, color="0.4", va="bottom")
    return sc


def cmd_pca(args):
    """PCA-denoised UMAP of each model's own embedding space.

    For each model: PCA to a shared number of dimensions (denoising),
    then UMAP with distances computed from the original (pre-PCA)
    embeddings.  Coloured by ground-truth % disorder.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from caid_disorder import Embedder, parse_reference_fasta
    import types

    entries = parse_reference_fasta(args.reference)
    n_prot = len(entries)
    n_pca = args.n_pca

    embedding_models = []

    # BioT
    print("Embedding with BioT...")
    biot_args = types.SimpleNamespace(
        ckpt=args.ckpt, esm=None, device=args.device,
        max_tokens=args.max_tokens, stride=args.stride, layer=args.layer)
    biot_emb = Embedder(biot_args)
    X_b, pct_b, acc_b = _embed_proteins_with(biot_emb, entries)
    print(f"  BioT: {len(X_b)} proteins, D={X_b.shape[1]}")
    xy_b, info_b = _reduce(X_b, n_pca)
    embedding_models.append(("BioT", xy_b, pct_b, info_b))

    # ESM2
    esm_name = args.esm or "esm2_t33_650M_UR50D"
    print(f"Embedding with {esm_name}...")
    esm_args = types.SimpleNamespace(
        ckpt=None, esm=esm_name, device=args.device,
        max_tokens=1022, stride=511, layer="int")
    try:
        esm_emb = Embedder(esm_args)
        X_e, pct_e, acc_e = _embed_proteins_with(esm_emb, entries)
        print(f"  ESM2: {len(X_e)} proteins, D={X_e.shape[1]}")
        xy_e, info_e = _reduce(X_e, n_pca)
        embedding_models.append((esm_name, xy_e, pct_e, info_e))
    except SystemExit as e:
        print(f"  ESM2 skipped: {e}")

    # ── Layout: side by side ──
    n = len(embedding_models)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), squeeze=False)

    for col, (name, xy, pct, info) in enumerate(embedding_models):
        sc = _scatter_panel(axes[0, col], xy, pct, name, info)

    cb = fig.colorbar(sc, ax=axes.ravel().tolist(),
                      pad=0.02, shrink=0.85, location="right")
    cb.set_label("% disordered", fontsize=9)

    fig.suptitle(f"Embedding UMAP — {n_prot} CAID proteins", fontsize=12,
                 fontweight="bold")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out}")


# ------------------------------------------------------------------- CLI

def main():
    import torch

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # metrics
    s1 = sub.add_parser("metrics", help="bar chart of CAID metrics")
    s1.add_argument("--csv", required=True,
                    help="dataset.default.metrics.csv from CAID assessment")
    s1.add_argument("--metrics", default=None,
                    help="comma-separated columns (default: aucroc,f1s,mcc,ppv,tpr)")
    s1.add_argument("--out", default="figures/caid_metrics.png")
    s1.set_defaults(func=cmd_metrics)

    # pca
    s2 = sub.add_parser("pca",
                         help="PCA of each model's own latent space")
    s2.add_argument("--reference", required=True)
    s2.add_argument("--ckpt", default="checkpoints/BioT_voxel_s0.safetensors")
    s2.add_argument("--esm", default=None, metavar="MODEL",
                    help="ESM2 model name (default: esm2_t33_650M_UR50D)")
    s2.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    s2.add_argument("--max-tokens", type=int, default=1024)
    s2.add_argument("--stride", type=int, default=512)
    s2.add_argument("--layer", default="int", choices=["int", "enc"])
    s2.add_argument("--n-pca", type=int, default=None,
                    help="PCA dims before UMAP (default: min(50, D))")
    s2.add_argument("--out", default="figures/umap_proteins.png")
    s2.set_defaults(func=cmd_pca)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
