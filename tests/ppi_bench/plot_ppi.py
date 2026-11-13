#!/usr/bin/env python
"""Plot PLM-Interact PPI benchmark results.

Compares BioT and ESM2 probe results against published baselines from
Liu et al. (2025) Nature Communications, with pretraining-set overlap
context and optional paired bootstrap for significance testing.

Usage::

    python plot_ppi.py \\
        --biot results_voxel/BioT_voxel_results.csv \\
        --biot-novox results_novox/BioT_novox_results.csv \\
        --esm results_esm2/esm2_t33_650M_UR50D_results.csv \\
        --preds-biot results_voxel/ \\
        --preds-novox results_novox/ \\
        --preds-esm results_esm2/ \\
        --out figures/ppi_bench.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Published baselines (Liu et al. 2025, Nature Comms) ──

PUBLISHED = {
    "cross_species": {
        "PLM-Interact (ESM2-650M)": {"auprc": 0.706, "auroc": np.nan},
        "TUnA":                     {"auprc": 0.641, "auroc": np.nan},
        "TT3D":                     {"auprc": 0.553, "auroc": np.nan},
        "Topsy-Turvy":              {"auprc": 0.490, "auroc": np.nan},
        "D-SCRIPT":                 {"auprc": 0.430, "auroc": np.nan},
    },
    "bernett": {
        "PLM-Interact (ESM2-650M)": {"auprc": 0.690, "auroc": 0.700},
        "TUnA":                     {"auprc": 0.690, "auroc": np.nan},
        "STEP (ProtTrans)":         {"auprc": 0.620, "auroc": np.nan},
        "Topsy-Turvy":              {"auprc": 0.580, "auroc": np.nan},
        "D-SCRIPT":                 {"auprc": 0.550, "auroc": np.nan},
    },
    "virus_human": {
        "PLM-Interact (ESM2-650M)": {"auprc": 0.950, "auroc": np.nan},
        "STEP (ProtTrans)":         {"auprc": 0.893, "auroc": np.nan},
    },
}

# ── Pretraining-set overlap at 80% seq identity (MMseqs2) ──

OVERLAP = {
    "cross_species": {"ESM2": 0.824, "BioT": 0.030},
    "bernett":       {"ESM2": 0.986, "BioT": 0.042},
    "virus_human":   {"ESM2": 0.961, "BioT": 0.044},
}

OVERLAP_COLORS = {"ESM2": "#5B8DBE", "BioT": "#F4B183"}

TASKS = ["cross_species", "bernett", "virus_human"]


# ── Helpers ──

def load_results(csv_path, label):
    """Load our results CSV -> {task: {metric: value}}."""
    df = pd.read_csv(csv_path)
    out = {}
    for _, row in df.iterrows():
        task = row["task"]
        if task not in out:
            out[task] = {}
        for m in ["auprc", "auroc"]:
            if m not in df.columns:
                continue
            if task in out and m in out[task]:
                out[task][m] = (out[task][m] + row[m]) / 2
            else:
                out[task][m] = row[m]
    return out


def _nice_task(name):
    return name.replace("_", " ").title()


# ── Paired bootstrap ──

def _load_preds(pred_dir, task):
    """Load {task}_test_preds.npz from a results directory."""
    pred_dir = Path(pred_dir)
    # Try common naming patterns
    for pattern in [f"{task}_test_preds.npz", f"{task}_preds.npz"]:
        p = pred_dir / pattern
        if p.exists():
            d = np.load(p)
            return d["probs"], d["labels"]
    return None, None


def paired_bootstrap_ppi(pred_dirs, n_boot=2000, seed=42):
    """Paired bootstrap of AUPRC/AUROC differences across models.

    All models are evaluated on the same test pairs in the same order,
    so we draw ONE shared resample of pair indices per replicate.

    Parameters
    ----------
    pred_dirs : dict  {model_name: pred_directory_path}

    Returns
    -------
    marginals : {model: {task: {metric: (med, lo, hi)}}}
    differences : {(model_a, model_b): {task: {metric: (mean, lo, hi)}}}
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    model_names = sorted(pred_dirs.keys())
    marginals = {m: {} for m in model_names}
    differences = {}

    for task in TASKS:
        # Load predictions for all models on this task
        preds = {}
        for m in model_names:
            probs, labels = _load_preds(pred_dirs[m], task)
            if probs is not None:
                preds[m] = (probs, labels)

        if len(preds) < 2:
            continue

        # Verify all have same length (same test pairs in same order)
        lengths = {m: len(p[0]) for m, p in preds.items()}
        if len(set(lengths.values())) > 1:
            print(f"  WARNING: {task} prediction lengths differ: {lengths}")
            continue

        n = list(lengths.values())[0]
        if n < 50:
            continue

        rng = np.random.RandomState(seed)
        boot = {m: {"auprc": [], "auroc": []} for m in preds}

        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)  # shared resample

            for m, (probs, labels) in preds.items():
                y = labels[idx]
                s = probs[idx]
                if len(np.unique(y)) < 2:
                    boot[m]["auprc"].append(np.nan)
                    boot[m]["auroc"].append(np.nan)
                else:
                    boot[m]["auprc"].append(average_precision_score(y, s))
                    boot[m]["auroc"].append(roc_auc_score(y, s))

        # Marginals
        for m in preds:
            marginals[m][task] = {}
            for metric in ["auprc", "auroc"]:
                v = np.array(boot[m][metric])
                v = v[~np.isnan(v)]
                if len(v) > 0:
                    marginals[m][task][metric] = (
                        np.median(v),
                        np.percentile(v, 2.5),
                        np.percentile(v, 97.5))

        # Paired differences
        ms = sorted(preds.keys())
        for i, m_a in enumerate(ms):
            for m_b in ms[i + 1:]:
                key = (m_a, m_b)
                if key not in differences:
                    differences[key] = {}
                differences[key][task] = {}
                for metric in ["auprc", "auroc"]:
                    a = np.array(boot[m_a][metric])
                    b = np.array(boot[m_b][metric])
                    ok = ~np.isnan(a) & ~np.isnan(b)
                    d = a[ok] - b[ok]
                    if len(d) > 0:
                        differences[key][task][metric] = (
                            np.mean(d),
                            np.percentile(d, 2.5),
                            np.percentile(d, 97.5))

    return marginals, differences


# ── Plotting ──

def plot_overlap(ax, task_name):
    """Horizontal bars showing pretraining overlap for one task."""
    ov = OVERLAP.get(task_name, {})
    if not ov:
        ax.set_visible(False)
        return

    names = list(ov.keys())
    vals = [ov[n] for n in names]
    colors = [OVERLAP_COLORS.get(n, "#999") for n in names]

    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.4,
            height=0.55)
    for yi, v in zip(y, vals):
        ax.text(v + 0.02, yi, f"{v:.0%}", va="center", fontsize=6.5)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlim(0, 1.12)
    ax.invert_yaxis()
    ax.tick_params(labelsize=6.5)
    ax.set_title(_nice_task(task_name), fontsize=8, fontweight="bold")


def plot_task(ax, task_name, our_results, metric="auprc"):
    """Horizontal bar chart for one task's performance."""
    published = PUBLISHED.get(task_name, {})

    methods, values, colors = [], [], []

    for label, results, color in our_results:
        if task_name in results:
            val = results[task_name].get(metric, np.nan)
            if not np.isnan(val):
                methods.append(label)
                values.append(val)
                colors.append(color)

    for method, metrics in published.items():
        val = metrics.get(metric, np.nan)
        if not np.isnan(val):
            methods.append(method)
            values.append(val)
            colors.append("#AAAAAA")

    if not methods:
        ax.set_visible(False)
        return

    y = np.arange(len(methods))
    bars = ax.barh(y, values, color=colors, edgecolor="white",
                   linewidth=0.4, height=0.65)

    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.008,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=6)

    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=6.5)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel(metric.upper(), fontsize=7)
    ax.invert_yaxis()
    ax.tick_params(labelsize=6.5)


# ── Main ──

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--biot", default=None, help="BioT voxel results CSV")
    p.add_argument("--biot-novox", default=None,
                   help="BioT no-voxel results CSV")
    p.add_argument("--esm", default=None, help="ESM2 results CSV")
    p.add_argument("--metric", default="auprc",
                   choices=["auprc", "auroc"])
    # Prediction directories for paired bootstrap
    p.add_argument("--preds-biot", default=None,
                   help="dir with {task}_test_preds.npz for BioT voxel")
    p.add_argument("--preds-novox", default=None,
                   help="dir with {task}_test_preds.npz for BioT novox")
    p.add_argument("--preds-esm", default=None,
                   help="dir with {task}_test_preds.npz for ESM2")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--out", default="figures/ppi_bench.png")
    args = p.parse_args()

    our_results = []
    if args.biot and Path(args.biot).exists():
        our_results.append(("BioT voxel",
                            load_results(args.biot, "BioT"), "#F4B183"))
    if args.biot_novox and Path(args.biot_novox).exists():
        our_results.append(("BioT no-voxel",
                            load_results(args.biot_novox, "BioT-novox"),
                            "#A9D18E"))
    if args.esm and Path(args.esm).exists():
        our_results.append(("ESM2-650M",
                            load_results(args.esm, "ESM2"), "#5B8DBE"))

    # ── Paired bootstrap ──
    pred_dirs = {}
    if args.preds_biot and Path(args.preds_biot).exists():
        pred_dirs["BioT_voxel"] = args.preds_biot
    if args.preds_novox and Path(args.preds_novox).exists():
        pred_dirs["BioT_novox"] = args.preds_novox
    if args.preds_esm and Path(args.preds_esm).exists():
        pred_dirs["ESM2"] = args.preds_esm

    if len(pred_dirs) >= 2:
        print(f"Paired bootstrap (n_boot={args.n_boot})...")
        marginals, differences = paired_bootstrap_ppi(
            pred_dirs, n_boot=args.n_boot)

        print("\n  Marginal 95% CIs:")
        for m in sorted(marginals):
            for task in TASKS:
                if task in marginals[m]:
                    parts = ", ".join(
                        f"{k}={v[0]:.4f} [{v[1]:.4f}, {v[2]:.4f}]"
                        for k, v in marginals[m][task].items())
                    print(f"    {m} / {task}: {parts}")

        print("\n  Paired differences (shared resample):")
        for (a, b), tasks_d in sorted(differences.items()):
            for task in TASKS:
                if task not in tasks_d:
                    continue
                for metric, (d, lo, hi) in tasks_d[task].items():
                    sig = " *" if lo > 0 or hi < 0 else ""
                    print(f"    {a} - {b} / {task} [{metric}]: "
                          f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]{sig}")
        print()

    # ── Plot ──
    n = len(TASKS)
    fig, axes = plt.subplots(
        2, n, figsize=(3.5 * n, 5.0),
        gridspec_kw={"height_ratios": [1, 2.5]})

    for col, task in enumerate(TASKS):
        plot_overlap(axes[0, col], task)
        plot_task(axes[1, col], task, our_results, metric=args.metric)

    axes[0, 0].set_ylabel("Overlap", fontsize=7, fontweight="bold")
    axes[1, 0].set_ylabel("Performance", fontsize=7, fontweight="bold")

    axes[0, 0].text(-0.02, 1.15, "a", transform=axes[0, 0].transAxes,
                    fontsize=9, fontweight="bold", va="top")
    axes[1, 0].text(-0.02, 1.08, "b", transform=axes[1, 0].transAxes,
                    fontsize=9, fontweight="bold", va="top")

    fig.tight_layout(h_pad=1.5, w_pad=1.0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
