#!/usr/bin/env python
"""Plot CAID benchmark results and embedding PCA scatterplots.

Subcommands::

    # Bar chart comparing methods on CAID metrics
    python plot_caid.py metrics \\
        --csv results/disorder_pdb.analysis.all.dataset.default.metrics.csv \\
        --out figures/caid_metrics.png

    # PCA of mean-pooled protein embeddings, coloured by % disorder
    python plot_caid.py pca \\
        --reference CAID/demo-data/references/disorder_pdb.fasta \\
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


# ------------------------------------------------------------------ PCA

def _load_embeddings(args):
    """Shared loader for PCA subcommands."""
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from caid_disorder import per_residue_embeddings, parse_reference_fasta
    from src.eval_utils import load_model

    model = load_model(args.ckpt, args.device)
    entries = parse_reference_fasta(args.reference)
    if args.max_proteins:
        entries = entries[:args.max_proteins]
    return model, entries


def _embed_proteins(args):
    """Mean-pool per-residue embeddings to one vector per protein.

    Returns ``(X, pct_disorder, accessions)`` where *X* is ``(N, D)``,
    *pct_disorder* is the fraction of labelled residues that are disordered,
    and *accessions* is the list of DisProt ids.
    """
    from sklearn.decomposition import PCA
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import torch
    from caid_disorder import per_residue_embeddings, parse_reference_fasta
    from src.eval_utils import load_model

    model = load_model(args.ckpt, args.device)
    entries = parse_reference_fasta(args.reference)

    vecs, pcts, accs = [], [], []
    for idx, (acc, seq, labels) in enumerate(entries):
        emb, clean = per_residue_embeddings(
            model, seq, args.device,
            max_tokens=args.max_tokens, stride=args.stride, layer=args.layer)
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


def cmd_pca(args):
    """PCA of mean-pooled protein embeddings, coloured by % disorder."""
    from sklearn.decomposition import PCA

    X, pct, accs = _embed_proteins(args)
    print(f"  {len(X)} proteins, D={X.shape[1]}")

    pca = PCA(n_components=2).fit(X)
    xy = pca.transform(X)

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=pct, cmap="RdYlBu_r",
                    vmin=0, vmax=1, s=18, alpha=0.8, edgecolors="0.3",
                    linewidths=0.3, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("% disordered residues", fontsize=9)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    ax.set_title(f"BioT mean-pooled embeddings ({len(X)} proteins)",
                 fontsize=10)
    fig.tight_layout()

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
                         help="PCA of mean-pooled embeddings, coloured by %% disorder")
    s2.add_argument("--reference", required=True)
    s2.add_argument("--ckpt", default="checkpoints/BioT_voxel_s0.safetensors")
    s2.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    s2.add_argument("--max-tokens", type=int, default=1024)
    s2.add_argument("--stride", type=int, default=512)
    s2.add_argument("--layer", default="int", choices=["int", "enc"])
    s2.add_argument("--out", default="figures/pca_proteins.png")
    s2.set_defaults(func=cmd_pca)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
