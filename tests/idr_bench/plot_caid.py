#!/usr/bin/env python
"""Plot CAID benchmark results and embedding PCA scatterplots.

Subcommands::

    # Bar chart: AUC-ROC + MCC with protein-level bootstrap 95% CIs
    python plot_caid.py metrics \\
        --csv results/disorder_pdb.analysis.all.dataset.default.metrics.csv \\
        --predictions predictions/ \\
        --reference CAID/demo-data/references/disorder_pdb.fasta \\
        --out figures/caid_metrics.png

    # PCA of mean-pooled protein embeddings, one panel per predictor
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

# Only genuinely independent evidence: AUC-ROC (threshold-free) and MCC
# (best single confusion-matrix summary).  F1, precision, recall are
# functions of the same confusion matrix at one threshold — reporting all
# five is reporting one observation five times.
HIGHLIGHT = ["aucroc", "mcc"]
NICE = {"aucroc": "AUC-ROC", "f1s": "F1", "mcc": "MCC",
        "ppv": "Precision", "tpr": "Recall", "aucpr": "AUC-PR",
        "bac": "Balanced Acc", "inf": "Informedness"}

# Training-set overlap at 80% sequence identity (MMseqs2 clustering).
# AIUPred omitted: its DL component trains on DisProt, not PDB, so a
# PDB-corpus overlap number would be misleading.
OVERLAP = {
    "ESM2-650M":     0.909,
    "AlphaFold-rsa": 0.514,
    "BioT":          0.053,  # shared corpus for both voxel and novox
}


def _method_color(method):
    """Consistent colours across plots."""
    m = method.lower().replace("-", "").replace("_", "")
    if "voxel" in m and "no" not in m:
        return "#F4B183"
    if "novox" in m or "novoxel" in m:
        return "#A9D18E"
    if "esm" in m:
        return "#5B8DBE"
    if "alphafold" in m:
        return "#E07B54"
    if "aiupred" in m:
        return "#6DC8BF"
    if "lipnet" in m:
        return "#AAAAAA"
    if "biot" in m:
        return "#F4B183"
    return "#D4A0D4"


def _match_overlap(method):
    """Look up overlap for a method name, collapsing BioT variants."""
    m = method.lower().replace("-", "").replace("_", "").replace(" ", "")
    for key, val in OVERLAP.items():
        k = key.lower().replace("-", "").replace("_", "").replace(" ", "")
        if k in m or m in k:
            return val
    if "biot" in m:
        return OVERLAP.get("BioT", np.nan)
    return np.nan


# --------------------------------------------------- bootstrap over proteins

def _parse_reference(fasta_path):
    """Parse CAID reference FASTA -> {accession: np.array of 0/1/nan}."""
    entries = {}
    acc, seq_lines, label_lines = None, [], []
    state = 0  # 0=waiting, 1=sequence, 2=labels
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if acc is not None and label_lines:
                    labels = []
                    for ch in "".join(label_lines):
                        if ch == "1":
                            labels.append(1.0)
                        elif ch == "0":
                            labels.append(0.0)
                        else:
                            labels.append(np.nan)
                    entries[acc] = np.array(labels)
                acc = line[1:].split()[0]
                seq_lines, label_lines = [], []
                state = 1
            elif state == 1:
                if line and set(line) <= {"0", "1", "-"}:
                    state = 2
                    label_lines.append(line)
                else:
                    seq_lines.append(line)
            elif state == 2:
                if line:
                    label_lines.append(line)
    if acc is not None and label_lines:
        labels = []
        for ch in "".join(label_lines):
            if ch == "1":
                labels.append(1.0)
            elif ch == "0":
                labels.append(0.0)
            else:
                labels.append(np.nan)
        entries[acc] = np.array(labels)
    return entries


def _parse_caid_predictions(caid_path):
    """Parse .caid file -> {accession: (scores, states)} per residue."""
    preds = {}
    acc, scores, states = None, [], []
    with open(caid_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if acc is not None and scores:
                    preds[acc] = (np.array(scores), np.array(states))
                acc = line[1:].split()[0]
                scores, states = [], []
            elif acc is not None and line:
                parts = line.split("\t")
                if len(parts) >= 4:
                    try:
                        scores.append(float(parts[2]))
                        states.append(int(parts[3]))
                    except (ValueError, IndexError):
                        pass
    if acc is not None and scores:
        preds[acc] = (np.array(scores), np.array(states))
    return preds


def _paired_bootstrap(ref, all_preds, n_boot=2000, seed=42):
    """Paired protein-level bootstrap across all methods.

    Every bootstrap replicate draws the SAME protein resample for every
    method, so protein-difficulty variance cancels in pairwise
    differences.  Marginal CIs from independent resamples overlap even
    when the paired difference clearly excludes zero.

    Parameters
    ----------
    ref : dict  {accession: labels}
    all_preds : dict  {method_name: {accession: (scores, states)}}

    Returns
    -------
    marginals : {method: {metric: (median, lo95, hi95)}}
    differences : {(method_a, method_b): {metric: (mean_diff, lo95, hi95)}}
    """
    from sklearn.metrics import roc_auc_score, matthews_corrcoef

    # Drop methods whose .caid files couldn't be parsed (e.g. 3-column
    # format without a binary-state column)
    method_names = sorted(m for m in all_preds if len(all_preds[m]) > 0)
    skipped = sorted(set(all_preds) - set(method_names))
    if skipped:
        print(f"    Skipped (no parseable predictions): {skipped}")

    # Proteins common to every remaining method and the reference
    common = set(ref)
    print(f"    Reference proteins: {len(common)}")
    for m in method_names:
        overlap = len(common & set(all_preds[m]))
        print(f"    {m}: {len(all_preds[m])} predictions, "
              f"{overlap} in common")
        common &= set(all_preds[m])
    common = sorted(common)
    print(f"    Intersection: {len(common)} proteins")

    # Build aligned per-protein arrays (same indexing for every method)
    method_protein = {m: [] for m in method_names}
    kept = 0
    for acc in common:
        lab = ref[acc]
        per_method = {}
        valid = True
        for m in method_names:
            sc, st = all_preds[m][acc]
            L = min(len(lab), len(sc))
            mask = ~np.isnan(lab[:L])
            if mask.sum() == 0:
                valid = False
                break
            per_method[m] = (lab[:L][mask].astype(int),
                             sc[:L][mask], st[:L][mask])
        if valid:
            for m in method_names:
                method_protein[m].append(per_method[m])
            kept += 1

    print(f"    After filtering: {kept} proteins with valid data")
    if kept < 10:
        print("    (too few — skipping bootstrap)")
        return {}, {}

    rng = np.random.RandomState(seed)
    n = kept
    boot_vals = {m: {"aucroc": [], "mcc": []} for m in method_names}

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)   # ONE shared resample

        for m in method_names:
            y = np.concatenate([method_protein[m][i][0] for i in idx])
            s = np.concatenate([method_protein[m][i][1] for i in idx])
            b = np.concatenate([method_protein[m][i][2] for i in idx])
            if len(np.unique(y)) < 2:
                boot_vals[m]["aucroc"].append(np.nan)
                boot_vals[m]["mcc"].append(np.nan)
            else:
                boot_vals[m]["aucroc"].append(roc_auc_score(y, s))
                boot_vals[m]["mcc"].append(matthews_corrcoef(y, b))

    # ── Marginal CIs ──
    marginals = {}
    for m in method_names:
        marginals[m] = {}
        for metric in ["aucroc", "mcc"]:
            v = np.array(boot_vals[m][metric])
            v = v[~np.isnan(v)]
            if len(v) > 0:
                marginals[m][metric] = (np.median(v),
                                        np.percentile(v, 2.5),
                                        np.percentile(v, 97.5))

    # ── Paired differences (shared resample cancels protein difficulty) ──
    differences = {}
    for i, m_a in enumerate(method_names):
        for m_b in method_names[i + 1:]:
            differences[(m_a, m_b)] = {}
            for metric in ["aucroc", "mcc"]:
                a = np.array(boot_vals[m_a][metric])
                b = np.array(boot_vals[m_b][metric])
                ok = ~np.isnan(a) & ~np.isnan(b)
                d = a[ok] - b[ok]
                if len(d) > 0:
                    differences[(m_a, m_b)][metric] = (
                        np.mean(d),
                        np.percentile(d, 2.5),
                        np.percentile(d, 97.5))

    return marginals, differences


# ------------------------------------------------------------------ metrics

def cmd_metrics(args):
    df = pd.read_csv(args.csv, index_col=0)
    cols = args.metrics.split(",") if args.metrics else HIGHLIGHT
    cols = [c for c in cols if c in df.columns]

    methods = list(df.index)
    colors = [_method_color(m) for m in methods]

    # ── Paired bootstrap (shared resample across all methods) ──
    boot_ci = {}
    if args.predictions and args.reference:
        print("Paired protein-level bootstrap (n_boot="
              f"{args.n_boot})...")
        ref = _parse_reference(args.reference)
        pred_dir = Path(args.predictions)
        all_preds = {}
        for caid_file in sorted(pred_dir.glob("*.caid")):
            all_preds[caid_file.stem] = _parse_caid_predictions(
                str(caid_file))
        if all_preds:
            boot_ci, boot_diff = _paired_bootstrap(
                ref, all_preds, n_boot=args.n_boot)

            print("\n  Marginal 95% CIs:")
            for m in sorted(boot_ci):
                parts = ", ".join(
                    f"{k}={v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"
                    for k, v in boot_ci[m].items())
                print(f"    {m}: {parts}")

            print("\n  Paired differences (shared resample):")
            for (a, b), metrics in sorted(boot_diff.items()):
                for metric, (d, lo, hi) in metrics.items():
                    sig = " *" if lo > 0 or hi < 0 else ""
                    print(f"    {a} - {b} [{metric}]: "
                          f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]{sig}")
            print()

    # ── Overlap: one bar per unique pretraining corpus ──
    seen_corpus = {}
    overlap_names, overlap_vals, overlap_colors = [], [], []
    for m in methods:
        ov = _match_overlap(m)
        if np.isnan(ov):
            continue
        corpus_key = "biot" if "biot" in m.lower() else m
        if corpus_key in seen_corpus:
            continue
        seen_corpus[corpus_key] = True
        label = "BioT" if "biot" in m.lower() else m
        overlap_names.append(label)
        overlap_vals.append(ov)
        overlap_colors.append(_method_color(label))

    # ── Layout: stacked single-column figure ──
    fig, (ax_ov, ax_perf) = plt.subplots(
        2, 1, figsize=(3.5, 4.5),
        gridspec_kw={"height_ratios": [1, 1.8]})

    # ── Panel a: training-set overlap ──
    y_ov = np.arange(len(overlap_names))
    ax_ov.barh(y_ov, overlap_vals, color=overlap_colors,
               edgecolor="white", linewidth=0.4, height=0.6)
    for yi, v in zip(y_ov, overlap_vals):
        ax_ov.text(v + 0.02, yi, f"{v:.0%}", va="center", fontsize=6.5)
    ax_ov.set_yticks(y_ov)
    ax_ov.set_yticklabels(overlap_names, fontsize=7)
    ax_ov.set_xlim(0, 1.12)
    ax_ov.set_xlabel("Pretraining-set overlap (80% seq. id.)", fontsize=7)
    ax_ov.invert_yaxis()
    ax_ov.tick_params(labelsize=6.5)
    ax_ov.set_title("a", fontsize=8, fontweight="bold", loc="left", x=-0.02)

    # ── Panel b: performance with bootstrap CIs ──
    y_met = np.arange(len(cols))
    h = 0.82 / len(methods)

    for i, (method, row) in enumerate(df.iterrows()):
        vals = [row[c] for c in cols]
        offset = (i - len(methods) / 2 + 0.5) * h

        bars = ax_perf.barh(
            y_met + offset, vals, height=h, label=method,
            color=colors[i], edgecolor="white", linewidth=0.4)

        for bar, v in zip(bars, vals):
            ax_perf.text(bar.get_width() + 0.012,
                         bar.get_y() + bar.get_height() / 2,
                         f"{v:.2f}", va="center", fontsize=5.5)

    ax_perf.set_yticks(y_met)
    ax_perf.set_yticklabels([NICE.get(c, c) for c in cols], fontsize=7)
    ax_perf.set_xlim(0, 1.15)
    ax_perf.set_xlabel("Score", fontsize=7)
    ax_perf.invert_yaxis()
    ax_perf.tick_params(labelsize=6.5)
    ax_perf.legend(bbox_to_anchor=(0.5, -0.18), loc="upper center",
                   fontsize=5.5, framealpha=0.9, ncol=3,
                   handlelength=1.0, handletextpad=0.3,
                   columnspacing=0.6, borderaxespad=0)
    ax_perf.set_title("b", fontsize=8, fontweight="bold", loc="left", x=-0.02)

    fig.tight_layout(h_pad=1.0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out}")


# -------------------------------------------------------- .caid parsing

def parse_caid_scores(path):
    """Parse a .caid prediction file.

    Returns ``{accession: mean_score}`` -- the mean per-residue disorder
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

    info = f"PCA({n_pca}, {var_kept:.0%} var) -> UMAP"
    return xy, info


def _scatter_panel(ax, xy, pct, title, info):
    """Draw one UMAP scatter panel coloured by % disorder."""
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=pct, cmap="RdYlBu_r",
                    vmin=0, vmax=1, s=18, alpha=0.8,
                    linewidths=0, rasterized=True)
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    # PCA annotation goes in the caption, not the plot
    return sc


def cmd_pca(args):
    """PCA-denoised UMAP of each model's own embedding space.

    For each model: PCA to a shared number of dimensions (denoising),
    then UMAP with distances computed from the original (pre-PCA)
    embeddings.  Coloured by ground-truth % disorder.
    """
    # caid_disorder.py lives in tests/ (one level up)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from caid_disorder import Embedder, parse_reference_fasta
    import types

    entries = parse_reference_fasta(args.reference)
    n_prot = len(entries)
    n_pca = args.n_pca

    embedding_models = []

    # BioT voxel
    print("Embedding with BioT (voxel)...")
    biot_args = types.SimpleNamespace(
        ckpt=args.ckpt, esm=None, device=args.device,
        max_tokens=args.max_tokens, stride=args.stride, layer=args.layer)
    biot_emb = Embedder(biot_args)
    X_b, pct_b, acc_b = _embed_proteins_with(biot_emb, entries)
    print(f"  BioT voxel: {len(X_b)} proteins, D={X_b.shape[1]}")
    xy_b, info_b = _reduce(X_b, n_pca)
    embedding_models.append(("BioT +field", xy_b, pct_b, info_b))

    # BioT novox
    if args.ckpt_novox:
        print("Embedding with BioT (no-voxel)...")
        novox_args = types.SimpleNamespace(
            ckpt=args.ckpt_novox, esm=None, device=args.device,
            max_tokens=args.max_tokens, stride=args.stride, layer=args.layer)
        try:
            novox_emb = Embedder(novox_args)
            X_n, pct_n, acc_n = _embed_proteins_with(novox_emb, entries)
            print(f"  BioT novox: {len(X_n)} proteins, D={X_n.shape[1]}")
            xy_n, info_n = _reduce(X_n, n_pca)
            embedding_models.append(("BioT −field", xy_n, pct_n, info_n))
        except Exception as e:
            print(f"  BioT novox skipped: {e}")

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
        esm_display = "ESM-2 (650M)"
        embedding_models.append((esm_display, xy_e, pct_e, info_e))
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

    fig.suptitle(f"Per-protein embedding UMAP ({n_prot} CAID proteins)",
                 fontsize=11, fontweight="bold")

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
                    help="comma-separated columns (default: aucroc,mcc)")
    s1.add_argument("--predictions", default=None,
                    help="directory of .caid files for bootstrap CIs")
    s1.add_argument("--reference", default=None,
                    help="CAID reference FASTA (needed with --predictions)")
    s1.add_argument("--n-boot", type=int, default=2000,
                    help="bootstrap replicates (default: 2000)")
    s1.add_argument("--out", default="figures/caid_metrics.png")
    s1.set_defaults(func=cmd_metrics)

    # pca
    s2 = sub.add_parser("pca",
                         help="PCA of each model's own latent space")
    s2.add_argument("--reference", required=True)
    s2.add_argument("--ckpt", default="checkpoints/BioT_voxel_s0.safetensors")
    s2.add_argument("--ckpt-novox", default=None,
                    help="BioT no-voxel checkpoint for ablation comparison")
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
