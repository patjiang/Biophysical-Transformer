#!/usr/bin/env python
"""Combined transfer-learning benchmark figure (CAID + PPI).

Single figure* with 5 panels:
  (a) Pretraining overlap across all benchmarks
  (b) CAID disorder metrics (AUC-ROC, MCC)
  (c–e) PPI AUPRC per task

Usage::

    python plot_benchmarks.py \\
        --caid-csv results/disorder_pdb.analysis.all.dataset.default.metrics.csv \\
        --ppi-biot results_voxel/BioT_voxel_results.csv \\
        --ppi-novox results_novox/BioT_novox_results.csv \\
        --ppi-esm results_esm2/esm2_t33_650M_UR50D_results.csv \\
        --out figures/fig_benchmarks.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════ palette (house style)
C_FIELD   = "#388E3C"   # BioT +field (dark green)
C_NOFIELD = "#A5D6A7"   # BioT −field (light green)
C_ESM     = "#1976D2"   # ESM-2 (blue)
C_AF      = "#78909C"   # AlphaFold (blue-grey)
C_OTHER   = "#B0B0B0"   # published baselines

LABEL_FIELD   = "BioT +field"
LABEL_NOFIELD = "BioT −field"
LABEL_ESM     = "ESM-2 (650M)"

# Font sizes for ICML print
FS_ANNOT  = 6
FS_LABEL  = 6.5
FS_TITLE  = 7.5
FS_PANEL  = 9
FS_BAR_IN = 5


# ═══════════════════════════════════════════════ overlap data
# Same grouped-bar style as panel (b): methods as rows within each benchmark.

OVERLAP_ORDER = [
    # (display_name, colour)
    (LABEL_ESM,     C_ESM),
    ("AlphaFold-rsa",   C_AF),
    ("BioT",        C_FIELD),
]

OVERLAP_DATA = {
    # benchmark: {method: value}  (None = not applicable)
    "CAID":                {LABEL_ESM: 0.909, "AlphaFold-rsa": 0.514, "BioT": 0.053},
    "PPI: Cross-sp.":      {LABEL_ESM: 0.824, "BioT": 0.030},
    "PPI: Bernett":        {LABEL_ESM: 0.986, "BioT": 0.042},
    "PPI: Virus–hum.": {LABEL_ESM: 0.961, "BioT": 0.044},
}

OVERLAP_BENCHMARKS = list(OVERLAP_DATA.keys())


# ═══════════════════════════════════════════════ CAID data

CAID_METRICS = ["aucroc", "mcc"]
CAID_NICE = {"aucroc": "AUC-ROC", "mcc": "MCC"}

CAID_ORDER = [
    (LABEL_FIELD,     C_FIELD,   "ablation"),
    (LABEL_NOFIELD,   C_NOFIELD, "ablation"),
    (LABEL_ESM,       C_ESM,     "external"),
    ("AlphaFold-rsa", C_AF,      "external"),
    ("AIUPred",       C_OTHER,   "external"),
]

CAID_NAME_MAP = {
    "BioT_voxel":    LABEL_FIELD,
    "BioT_novox":    LABEL_NOFIELD,
    "ESM2-650M":     LABEL_ESM,
    "AlphaFold-rsa": "AlphaFold-rsa",
    "AIUPred":       "AIUPred",
}


# ═══════════════════════════════════════════════ PPI data

PPI_TASKS = ["cross_species", "bernett", "virus_human"]
PPI_NICE = {
    "cross_species": "Cross-species",
    "bernett":       "Bernett",
    "virus_human":   "Virus–human",
}

PPI_SOTA = {
    "cross_species": ("PLM-Interact", 0.706),
    "bernett":       ("PLM-Interact", 0.690),
    "virus_human":   ("PLM-Interact", 0.950),
}


# ═══════════════════════════════════════════════ helpers

def load_ppi_results(csv_path):
    df = pd.read_csv(csv_path)
    out = {}
    for _, row in df.iterrows():
        task = row["task"]
        if task not in out:
            out[task] = {}
        for m in ["auprc", "auroc"]:
            if m in df.columns:
                out[task][m] = row[m]
    return out


def _draw_labeled_bars(ax, y_positions, values, colors, names,
                       bar_h, fmt=".0%", xlim=1.08):
    """Draw horizontal bars with method name + value inside."""
    for yi, v, c, name in zip(y_positions, values, colors, names):
        ax.barh(yi, v, height=bar_h, color=c,
                edgecolor="white", linewidth=0.4)
        if v > 0.20:
            ax.text(v - 0.008, yi, f"{v:{fmt}}",
                    va="center", ha="right",
                    fontsize=FS_BAR_IN, color="white", fontweight="medium")
            ax.text(0.012, yi, name,
                    va="center", ha="left",
                    fontsize=FS_BAR_IN - 0.5, color="white")
        else:
            ax.text(v + 0.008, yi,
                    f"{name}  {v:{fmt}}",
                    va="center", ha="left", fontsize=FS_BAR_IN)


# ═══════════════════════════════════════════════ panels

def panel_overlap(ax):
    """(a) Overlap — same grouped-bar style as panel (b)."""
    benchmarks = OVERLAP_BENCHMARKS
    n_bench = len(benchmarks)
    all_methods = [m for m, _ in OVERLAP_ORDER]

    bar_h = 0.30
    gap = 0.08
    y_bench = np.arange(n_bench) * 1.1  # extra row spacing

    for bi, bench in enumerate(benchmarks):
        bench_data = OVERLAP_DATA[bench]
        present = [(m, c) for m, c in OVERLAP_ORDER if m in bench_data]
        n = len(present)

        positions, values, colors, names = [], [], [], []
        for j, (m, c) in enumerate(present):
            offset = (j - n / 2 + 0.5) * bar_h
            positions.append(y_bench[bi] + offset)
            values.append(bench_data[m])
            colors.append(c)
            names.append(m)

        _draw_labeled_bars(ax, positions, values, colors, names,
                           bar_h, fmt=".0%")

    ax.set_yticks(y_bench)
    ax.set_yticklabels(benchmarks, fontsize=FS_LABEL)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Pretraining overlap (80% seq. id.)", fontsize=FS_LABEL)
    ax.invert_yaxis()
    ax.tick_params(labelsize=FS_ANNOT)


def panel_caid(ax, csv_path):
    """(b) CAID AUC-ROC + MCC — ablation pair first, gap, then external."""
    df = pd.read_csv(csv_path, index_col=0)
    cols = [c for c in CAID_METRICS if c in df.columns]

    methods, colors, data, groups = [], [], [], []
    for display_name, color, group in CAID_ORDER:
        csv_name = None
        for k, v in CAID_NAME_MAP.items():
            if v == display_name and k in df.index:
                csv_name = k
                break
        if csv_name is None:
            continue
        methods.append(display_name)
        colors.append(color)
        data.append([df.loc[csv_name, c] for c in cols])
        groups.append(group)

    n_methods = len(methods)
    n_metrics = len(cols)
    y_met = np.arange(n_metrics)

    bar_h = 0.15
    gap = 0.08
    abl_count = sum(1 for g in groups if g == "ablation")

    for mi in range(n_metrics):
        positions, values, clrs, names = [], [], [], []
        for i, (method, color, vals, group) in enumerate(
                zip(methods, colors, data, groups)):
            offset = (i - n_methods / 2 + 0.5) * bar_h
            if i >= abl_count:
                offset += gap
            positions.append(mi + offset)
            values.append(vals[mi])
            clrs.append(color)
            names.append(method)

        _draw_labeled_bars(ax, positions, values, clrs, names,
                           bar_h, fmt=".2f")

    ax.set_yticks(y_met)
    ax.set_yticklabels([CAID_NICE.get(c, c) for c in cols], fontsize=FS_LABEL)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Score", fontsize=FS_LABEL)
    ax.invert_yaxis()
    ax.tick_params(labelsize=FS_ANNOT)


def panel_ppi(ax, task, our_results):
    """(c–e) PPI AUPRC bars + SOTA. No legend — labels inside bars."""
    methods, vals, colors = [], [], []
    for label, results, color in our_results:
        if task in results:
            v = results[task].get("auprc", np.nan)
            if not np.isnan(v):
                methods.append(label)
                vals.append(v)
                colors.append(color)

    sota = PPI_SOTA.get(task)
    if sota:
        methods.append(f"SOTA\n({sota[0]})")
        vals.append(sota[1])
        colors.append(C_OTHER)

    if not methods:
        ax.set_visible(False)
        return

    y = np.arange(len(methods))
    bar_h = 0.55

    for yi, v, c in zip(y, vals, colors):
        ax.barh(yi, v, height=bar_h, color=c,
                edgecolor="white", linewidth=0.4)
        if v > 0.35:
            ax.text(v - 0.01, yi, f"{v:.3f}",
                    va="center", ha="right",
                    fontsize=FS_ANNOT, color="white", fontweight="medium")
        else:
            ax.text(v + 0.01, yi, f"{v:.3f}",
                    va="center", ha="left", fontsize=FS_ANNOT)

    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=FS_LABEL)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("AUPRC", fontsize=FS_LABEL)
    ax.set_title(PPI_NICE.get(task, task), fontsize=FS_TITLE,
                 fontweight="bold", pad=4)
    ax.invert_yaxis()
    ax.tick_params(labelsize=FS_ANNOT)


# ═══════════════════════════════════════════════ main

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--caid-csv", required=True, help="CAID metrics CSV")
    p.add_argument("--ppi-biot", default=None)
    p.add_argument("--ppi-novox", default=None)
    p.add_argument("--ppi-esm", default=None)
    p.add_argument("--out", default="figures/fig_benchmarks.png")
    args = p.parse_args()

    ppi_results = []
    if args.ppi_biot and Path(args.ppi_biot).exists():
        ppi_results.append((LABEL_FIELD,
                            load_ppi_results(args.ppi_biot), C_FIELD))
    if args.ppi_novox and Path(args.ppi_novox).exists():
        ppi_results.append((LABEL_NOFIELD,
                            load_ppi_results(args.ppi_novox), C_NOFIELD))
    if args.ppi_esm and Path(args.ppi_esm).exists():
        ppi_results.append((LABEL_ESM,
                            load_ppi_results(args.ppi_esm), C_ESM))

    # ── Layout: 2 rows ──
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = gridspec.GridSpec(2, 1, figure=fig,
                           height_ratios=[1.2, 1],
                           hspace=0.5)

    gs_top = gs[0].subgridspec(1, 2, width_ratios=[1, 1.3], wspace=0.45)
    ax_ov = fig.add_subplot(gs_top[0])
    ax_caid = fig.add_subplot(gs_top[1])

    gs_bot = gs[1].subgridspec(1, 3, wspace=0.55)
    ax_ppi = [fig.add_subplot(gs_bot[i]) for i in range(3)]

    # ── Draw ──
    panel_overlap(ax_ov)
    panel_caid(ax_caid, args.caid_csv)
    for i, task in enumerate(PPI_TASKS):
        panel_ppi(ax_ppi[i], task, ppi_results)

    # ── Panel labels ──
    for letter, ax in [("a", ax_ov), ("b", ax_caid),
                       ("c", ax_ppi[0]), ("d", ax_ppi[1]), ("e", ax_ppi[2])]:
        ax.text(-0.02, 1.12, letter, transform=ax.transAxes,
                fontsize=FS_PANEL, fontweight="bold", va="top")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
