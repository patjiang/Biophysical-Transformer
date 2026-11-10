#!/usr/bin/env python
"""Plot CAID benchmark results and embedding PCA scatterplots.

Subcommands::

    # Bar chart comparing methods on CAID metrics
    python plot_caid.py metrics \\
        --csv results/disorder_pdb.analysis.all.dataset.default.metrics.csv \\
        --out figures/caid_metrics.png

    # PCA scatterplots of per-residue embeddings (one PNG per protein)
    python plot_caid.py pca \\
        --reference CAID/demo-data/references/disorder_pdb.fasta \\
        --ckpt ../../checkpoints/BioT_voxel_s0.safetensors \\
        --out figures/pca/ --max-proteins 20

    # Single PCA over all proteins (residues pooled)
    python plot_caid.py pca-global \\
        --reference CAID/demo-data/references/disorder_pdb.fasta \\
        --ckpt ../../checkpoints/BioT_voxel_s0.safetensors \\
        --out figures/pca_global.png --max-proteins 100
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


def cmd_pca(args):
    """One PCA scatterplot per protein, coloured by disorder label."""
    from sklearn.decomposition import PCA
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import torch
    from caid_disorder import per_residue_embeddings, parse_reference_fasta
    from src.eval_utils import load_model

    model = load_model(args.ckpt, args.device)
    entries = parse_reference_fasta(args.reference)
    if args.max_proteins:
        entries = entries[:args.max_proteins]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, (acc, seq, labels) in enumerate(entries):
        emb, clean = per_residue_embeddings(
            model, seq, args.device,
            max_tokens=args.max_tokens, stride=args.stride, layer=args.layer)
        if emb is None:
            continue

        L = min(len(clean), len(labels), emb.shape[0])
        emb, lab = emb[:L], labels[:L]
        mask = ~np.isnan(lab)
        if mask.sum() < 10:
            continue

        emb_m, lab_m = emb[mask], lab[mask]
        pca = PCA(n_components=2).fit(emb_m)
        xy = pca.transform(emb_m)

        fig, ax = plt.subplots(figsize=(5, 4))
        order = lab_m == 0
        disor = lab_m == 1
        ax.scatter(xy[order, 0], xy[order, 1], s=6, alpha=0.5,
                   c="#4878CF", label="ordered", rasterized=True)
        ax.scatter(xy[disor, 0], xy[disor, 1], s=6, alpha=0.5,
                   c="#D65F5F", label="disordered", rasterized=True)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
        ax.set_title(f"{acc}  (L={len(clean)})", fontsize=10)
        ax.legend(fontsize=8, markerscale=2)
        fig.tight_layout()

        path = out_dir / f"{acc}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        if (idx + 1) % 25 == 0 or idx + 1 == len(entries):
            print(f"  plotted {idx + 1}/{len(entries)}", flush=True)

    print(f"  saved -> {out_dir}/")


def cmd_pca_global(args):
    """Single PCA scatterplot pooling residues across all proteins."""
    from sklearn.decomposition import PCA
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import torch
    from caid_disorder import per_residue_embeddings, parse_reference_fasta
    from src.eval_utils import load_model

    model = load_model(args.ckpt, args.device)
    entries = parse_reference_fasta(args.reference)
    if args.max_proteins:
        entries = entries[:args.max_proteins]

    all_emb, all_lab = [], []
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
        all_emb.append(emb[:L][mask])
        all_lab.append(lab[mask])
        if (idx + 1) % 50 == 0:
            print(f"  embedded {idx + 1}/{len(entries)}", flush=True)

    X = np.concatenate(all_emb)
    y = np.concatenate(all_lab)
    print(f"  {len(X):,} residues ({int(y.sum()):,} disordered)")

    # Subsample if huge
    if len(X) > args.max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), args.max_points, replace=False)
        X, y = X[idx], y[idx]
        print(f"  subsampled to {args.max_points:,}")

    pca = PCA(n_components=2).fit(X)
    xy = pca.transform(X)

    fig, ax = plt.subplots(figsize=(6, 5))
    order = y == 0
    disor = y == 1
    ax.scatter(xy[order, 0], xy[order, 1], s=2, alpha=0.15,
               c="#4878CF", label="ordered", rasterized=True)
    ax.scatter(xy[disor, 0], xy[disor, 1], s=2, alpha=0.15,
               c="#D65F5F", label="disordered", rasterized=True)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    ax.set_title(f"BioT residue embeddings ({len(entries)} proteins)",
                 fontsize=10)
    ax.legend(fontsize=8, markerscale=4)
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

    def pca_common(sp):
        sp.add_argument("--reference", required=True)
        sp.add_argument("--ckpt", default="checkpoints/BioT_voxel_s0.safetensors")
        sp.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
        sp.add_argument("--max-tokens", type=int, default=1024)
        sp.add_argument("--stride", type=int, default=512)
        sp.add_argument("--layer", default="int", choices=["int", "enc"])
        sp.add_argument("--max-proteins", type=int, default=None)

    # pca (per-protein)
    s2 = sub.add_parser("pca", help="per-protein PCA scatterplots")
    pca_common(s2)
    s2.add_argument("--out", default="figures/pca/")
    s2.set_defaults(func=cmd_pca)

    # pca-global
    s3 = sub.add_parser("pca-global", help="pooled PCA scatterplot")
    pca_common(s3)
    s3.add_argument("--out", default="figures/pca_global.png")
    s3.add_argument("--max-points", type=int, default=50000)
    s3.set_defaults(func=cmd_pca_global)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
