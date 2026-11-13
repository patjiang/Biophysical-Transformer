#!/usr/bin/env python
"""Paired-difference bootstrap distributions for the CAID ablation.

Reads the .caid prediction files and reference FASTA, runs the paired
protein-level bootstrap (shared resamples), and plots the distribution
of Δ AUC-ROC and Δ MCC (+field minus −field) as histograms with the
95% CI shaded.

Usage::

    python plot_bootstrap_delta.py \\
        --reference CAID/demo-data/references/disorder_pdb.fasta \\
        --pred-field predictions/BioT_voxel.caid \\
        --pred-nofield predictions/BioT_novox.caid \\
        --n-boot 2000 \\
        --out figures/bootstrap_delta.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ═══════════════════════════════════════════════ palette
C_FIELD = "#388E3C"
C_ZERO  = "#B0B0B0"

FS_ANNOT = 7
FS_LABEL = 8
FS_TITLE = 9


# ═══════════════════════════════════════════════ parsing (same as plot_caid)

def _parse_reference(fasta_path):
    entries = {}
    acc, seq_lines, label_lines = None, [], []
    state = 0
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


# ═══════════════════════════════════════════════ bootstrap

def paired_bootstrap_deltas(ref, preds_field, preds_nofield,
                            n_boot=2000, seed=42):
    """Return arrays of Δ AUC-ROC and Δ MCC (+field minus −field)."""
    from sklearn.metrics import roc_auc_score, matthews_corrcoef

    common = sorted(set(ref) & set(preds_field) & set(preds_nofield))
    print(f"  {len(common)} proteins common to both arms and reference")

    protein_data = []  # (labels, scores_field, states_field, scores_nf, states_nf)
    for acc in common:
        lab = ref[acc]
        sf, stf = preds_field[acc]
        sn, stn = preds_nofield[acc]
        L = min(len(lab), len(sf), len(sn))
        lab, sf, stf, sn, stn = lab[:L], sf[:L], stf[:L], sn[:L], stn[:L]
        mask = ~np.isnan(lab)
        if mask.sum() == 0:
            continue
        protein_data.append((
            lab[mask].astype(int),
            sf[mask], stf[mask],
            sn[mask], stn[mask],
        ))

    print(f"  {len(protein_data)} proteins with valid labels")
    if len(protein_data) < 10:
        sys.exit("Too few proteins for bootstrap")

    rng = np.random.RandomState(seed)
    n = len(protein_data)
    d_auroc, d_mcc = [], []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)

        y  = np.concatenate([protein_data[i][0] for i in idx])
        sf = np.concatenate([protein_data[i][1] for i in idx])
        bf = np.concatenate([protein_data[i][2] for i in idx])
        sn = np.concatenate([protein_data[i][3] for i in idx])
        bn = np.concatenate([protein_data[i][4] for i in idx])

        if len(np.unique(y)) < 2:
            continue

        auroc_f = roc_auc_score(y, sf)
        auroc_n = roc_auc_score(y, sn)
        mcc_f   = matthews_corrcoef(y, bf)
        mcc_n   = matthews_corrcoef(y, bn)

        d_auroc.append(auroc_f - auroc_n)
        d_mcc.append(mcc_f - mcc_n)

    return np.array(d_auroc), np.array(d_mcc)


# ═══════════════════════════════════════════════ plotting

def plot_delta_hist(ax, deltas, metric_name, color):
    """Clean histogram of bootstrap deltas + zero reference line."""
    lo = np.percentile(deltas, 2.5)
    hi = np.percentile(deltas, 97.5)
    med = np.median(deltas)

    ax.hist(deltas, bins=50, color=color, alpha=0.75, edgecolor="white",
            linewidth=0.3)

    # Zero reference
    ax.axvline(0, color=C_ZERO, ls="-", lw=1.0, zorder=3)

    ax.set_xlabel(f"Δ {metric_name}  (+field − −field)",
                  fontsize=FS_LABEL)
    ax.set_ylabel("Count", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_ANNOT)

    # Print to stdout for the caption
    excludes_zero = "yes" if lo > 0 or hi < 0 else "no"
    print(f"  {metric_name}: median={med:+.4f}  "
          f"95% CI=[{lo:+.4f}, {hi:+.4f}]  "
          f"excludes zero: {excludes_zero}")


# ═══════════════════════════════════════════════ main

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", required=True,
                   help="CAID reference FASTA")
    p.add_argument("--pred-field", required=True,
                   help=".caid predictions from +field model")
    p.add_argument("--pred-nofield", required=True,
                   help=".caid predictions from −field model")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--out", default="figures/bootstrap_delta.png")
    args = p.parse_args()

    print("Parsing reference and predictions...")
    ref = _parse_reference(args.reference)
    preds_field = _parse_caid_predictions(args.pred_field)
    preds_nofield = _parse_caid_predictions(args.pred_nofield)

    print(f"  Reference: {len(ref)} proteins")
    print(f"  +field:    {len(preds_field)} proteins")
    print(f"  −field:    {len(preds_nofield)} proteins")

    print(f"\nRunning paired bootstrap ({args.n_boot} replicates)...")
    d_auroc, d_mcc = paired_bootstrap_deltas(
        ref, preds_field, preds_nofield, n_boot=args.n_boot)

    print(f"\nResults ({len(d_auroc)} valid replicates):")

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))

    plot_delta_hist(ax1, d_auroc, "AUC-ROC", C_FIELD)
    plot_delta_hist(ax2, d_mcc, "MCC", C_FIELD)

    ax1.text(-0.02, 1.10, "a", transform=ax1.transAxes,
             fontsize=10, fontweight="bold", va="top")
    ax2.text(-0.02, 1.10, "b", transform=ax2.transAxes,
             fontsize=10, fontweight="bold", va="top")

    fig.tight_layout(w_pad=2.0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
