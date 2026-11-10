#!/usr/bin/env python
"""Regenerate every figure from cached results.

    python make_figures.py --audit overlap_audit --rerun rerun

Reads ``overlap_audit/results.json`` and ``rerun/*.json``, writes PDF + PNG
plus a ``captions.md`` with LaTeX-ready figure captions.
"""

import argparse
import json
import sys
from pathlib import Path

SURFACE = "#ffffff"
INK, INK2, INK3, GRID = "#111111", "#55534e", "#8a8984", "#e3e2de"
PAL = {"human": "#2a78d6", "human_cell": "#eb6834", "mixed_split": "#1baf7a"}
SPARE = ["#eda100", "#e87ba4", "#6250d6"]
C_FULL, C_REST = "#2a78d6", "#eb6834"
LADDER = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 1.00]

# The containment level the exposure bars report, matching the rest of the
# paper and the restricted re-run threshold.
EXPOSURE_LEVEL = "0.80"
EXPOSURE_SPAN = ("0.50", "0.95")

CAPTIONS = {}


def log(m=""):
    print(m, flush=True)


def color_for(split, i):
    return PAL.get(split, SPARE[i % len(SPARE)])


def style():
    import matplotlib as mpl
    mpl.use("Agg")
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "savefig.edgecolor": SURFACE,
        "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
        "axes.linewidth": 0.8, "xtick.major.size": 3, "ytick.major.size": 3,
        "grid.color": GRID, "grid.linewidth": 0.7, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=300,
                    facecolor=SURFACE)
    log(f"    {out / name}.pdf / .png")


def bare(ax, spines=("top", "right")):
    for s in spines:
        ax.spines[s].set_visible(False)


# ===================================================== audit figures

def fig_ladder(res, out):
    """Containment ladder by split."""
    import matplotlib.pyplot as plt
    style()
    x = list(range(len(LADDER)))
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    ymax = 0.0
    for i, (split, r) in enumerate(res["splits"].items()):
        y = [100 * r["ladder"][f"{t:.2f}"] for t in LADDER]
        ymax = max(ymax, max(y))
        c = color_for(split, i)
        ax.plot(x, y, "-o", color=c, lw=1.8, ms=5, mec=SURFACE, mew=1.2,
                label=split, zorder=3)
        ax.annotate(f"{split} ", (x[0], y[0]), textcoords="offset points",
                    xytext=(-6, 0), ha="right", va="center", color=c,
                    fontsize=8, annotation_clip=False)
    call_i = min(range(len(LADDER)), key=lambda k: abs(LADDER[k] - res["call"]))
    ax.axvline(call_i, color=INK3, lw=0.9, ls=(0, (3, 3)), zorder=1)
    ax.set_ylim(0, max(ymax * 1.12, 1.0))
    ax.set_xlim(-1.9, len(LADDER) - 0.75)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.2f}" for t in LADDER])
    ax.set_xlabel("fraction of the test protein present in a single "
                  "pretraining protein")
    ax.set_ylabel("FLIP test proteins (%)")
    ax.set_title("Test proteins already in the pretraining corpus",
                 loc="left", fontsize=10, pad=8)
    bare(ax)
    save(fig, out, "fig1_ladder")
    plt.close(fig)

    def _at(split, t):
        return res["splits"][split]["ladder"].get(f"{t:.2f}")
    order = ""
    if {"human", "mixed_split"} <= set(res["splits"]):
        lo_t, hi_t = 0.50, 0.95
        a_lo, b_lo = _at("human", lo_t), _at("mixed_split", lo_t)
        a_hi, b_hi = _at("human", hi_t), _at("mixed_split", hi_t)
        if None not in (a_lo, b_lo, a_hi, b_hi):
            order = (
                f" At {lo_t:.2f}, human is {100 * a_lo:.1f}% against "
                f"mixed_split's {100 * b_lo:.1f}%; at {hi_t:.2f} they are "
                f"{100 * a_hi:.1f}% and {100 * b_hi:.1f}%."
                + (" The ordering reverses, so the split-dependence sits in"
                   " partial rather than whole-protein overlap."
                   if (a_lo > b_lo) and (a_hi <= b_hi) else ""))
    CAPTIONS["fig1_ladder"] = (
        "\\textbf{Overlap between the FLIP test splits and the pretraining "
        "corpus.} For each test protein, the fraction of its overlapping "
        "3-residue words that appear within a single pretraining protein; an "
        "identical protein scores 1.0. Curves give the percentage of test "
        "proteins at or above each level, with the dashed rule at the level "
        f"reported in the text ({res['call']:.2f})." + order +
        " Counts are given in the accompanying table.")


def fig_null(res, out):
    """Observed vs shuffled containment, one panel per split."""
    import matplotlib.pyplot as plt
    import numpy as np
    style()
    sp = res["splits"]
    fig, axes = plt.subplots(1, len(sp), figsize=(2.5 * len(sp), 2.7),
                             sharey=True)
    if len(sp) == 1:
        axes = [axes]
    for i, ((split, r), ax) in enumerate(zip(sp.items(), axes)):
        c = color_for(split, i)
        for key, ls, lab in (("obs_ecdf", "-", "observed"),
                             ("null_ecdf", (0, (4, 2)), "shuffled")):
            e = r.get(key)
            if e:
                ax.plot(e["x"], 100 * np.array(e["y"]), ls=ls, color=c,
                        lw=1.8, label=lab, zorder=3)
        ax.axvline(res["call"], color=INK3, lw=0.9, ls=(0, (3, 3)), zorder=1)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)
        ax.set_title(split, loc="left", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_xlabel("overlap")
        bare(ax)
        if i == 0:
            ax.set_ylabel("test proteins at or below (%)")
            ax.legend(fontsize=7.5, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    save(fig, out, "fig2_null")
    plt.close(fig)
    CAPTIONS["fig2_null"] = (
        "\\textbf{Overlap against a composition-matched null.} Solid, the "
        "observed distribution of overlap with the best single pretraining "
        "protein; dashed, the same statistic after each test sequence is "
        "shuffled, preserving its length and amino-acid composition while "
        "destroying its 3-residue words. The statistic is a maximum over "
        "26{,}773 proteins, so its chance level varies with corpus size and "
        "query length and is established by the null rather than assumed. "
        "The distributions coincide through the bulk and separate in the "
        "upper tail. Dashed rule as in Fig. 1.")


def _ladder_at(res, split, level):
    """Ladder value at a level, accepting a string or float key."""
    lad = res["splits"][split]["ladder"]
    if level in lad:
        return lad[level]
    f = float(level)
    for k, v in lad.items():
        if abs(float(k) - f) < 1e-9:
            return v
    raise KeyError(f"no ladder level {level} for {split}; have {sorted(lad)}")


def fig_exposure(res, out):
    """SE3PLM exposure beside ESM's, grouped bars with whisker span."""
    import matplotlib.pyplot as plt
    import numpy as np
    ex = res.get("exposure")
    if not ex:
        return
    style()
    splits = [s for s in res["splits"] if s in ex]
    lo_k, hi_k = EXPOSURE_SPAN
    models = [("SE3PLM",
               lambda s: 100 * _ladder_at(res, s, EXPOSURE_LEVEL)),
              ("ESM-1b", lambda s: 100 * ex[s]["esm1b"]["of_test"]),
              ("ESM-1v", lambda s: 100 * ex[s]["esm1v"]["of_test"])]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    w, x = 0.26, np.arange(len(splits))
    for j, (name, f) in enumerate(models):
        v = [f(s) for s in splits]
        b = ax.bar(x + (j - 1) * (w + 0.02), v, w, label=name,
                   color=["#2a78d6", "#eb6834", "#1baf7a"][j], zorder=3, lw=0)
        if name == "SE3PLM":
            lo = [100 * _ladder_at(res, sp, hi_k) for sp in splits]
            hi = [100 * _ladder_at(res, sp, lo_k) for sp in splits]
            ax.errorbar(x + (j - 1) * (w + 0.02), v,
                        yerr=[np.array(v) - np.array(lo),
                              np.array(hi) - np.array(v)],
                        fmt="none", ecolor=INK3, elinewidth=1.0,
                        capsize=3, capthick=1.0, zorder=4)
        for k_, (rect, val) in enumerate(zip(b, v)):
            top = 100 * _ladder_at(res, splits[k_], lo_k) \
                if name == "SE3PLM" else val
            ax.annotate(f"{val:.1f}",
                        (rect.get_x() + rect.get_width() / 2, top),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=7.5, color=INK2)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylim(0, 108)
    ax.set_ylabel("test proteins seen in pretraining (%)")
    ax.set_title("Pretraining exposure to the FLIP test proteins",
                 loc="left", fontsize=10, pad=8)
    ax.legend(fontsize=8, labelcolor=INK2, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), columnspacing=2.2,
              handlelength=1.4)
    bare(ax)
    save(fig, out, "fig3_exposure")
    plt.close(fig)
    snap = res.get("esm1b_snapshot", "2018-03")
    CAPTIONS["fig3_exposure"] = (
        "\\textbf{Pretraining exposure to the FLIP test proteins, each model "
        "judged by its own criterion.} SE3PLM: test proteins with at least "
        f"{EXPOSURE_LEVEL} of their 3-mers inside a single protein of its "
        "26{{,}}773-structure corpus, the level used throughout the paper and "
        "the level at which proteins are removed for the restricted re-run. "
        f"The whiskers span {lo_k} to {hi_k} on the same ladder, since our "
        "criterion has a threshold and ESM's has none. ESM: test proteins "
        "present in UniProt before the release snapshot of the corpus each "
        "model was trained on; UniRef is a clustering of UniProt, so such an "
        "entry lay within that release and a sequence at or above the "
        "cluster identity was in the corpus (ESM-1v, UniRef90 release "
        f"2020-03, 98M sequences; ESM-1b, UniRef50 at 50\\% identity, release "
        f"{snap}). The two criteria are not commensurable and ours is the "
        "stricter. Test proteins without a UniProt match are counted as "
        "unmatched rather than unexposed, so the ESM values are lower "
        "bounds.")


def fig_fwd_rev(res, out):
    """Forward vs reverse containment scatter."""
    import matplotlib.pyplot as plt
    cands = res.get("candidates") or []
    if not cands:
        return
    style()
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.axhline(0.80, color=INK3, lw=0.9, ls=(0, (3, 3)), zorder=1)
    for i, split in enumerate(res["splits"]):
        pts = [c for c in cands if c["id"].startswith(split + ":")]
        if not pts:
            continue
        ax.scatter([p["forward"] for p in pts], [p["reverse"] for p in pts],
                   s=18, color=color_for(split, i), label=split,
                   edgecolor=SURFACE, linewidth=0.7, alpha=0.85, zorder=3)
    ax.set_xlabel("fraction of the test protein found in the match")
    ax.set_ylabel("fraction of the match found in the test protein")
    ax.set_title("Whole-protein matches, or fragments?",
                 loc="left", fontsize=10, pad=8)
    ax.set_xlim(res["call"] - 0.02, 1.02)
    ax.set_ylim(-0.02, 1.06)
    ax.legend(fontsize=8, labelcolor=INK2, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.20), columnspacing=1.6,
              handlelength=1.2)
    bare(ax)
    save(fig, out, "fig4_fwd_rev")
    plt.close(fig)
    n_same = sum(1 for c in cands if c.get("kind") == "same-protein")
    CAPTIONS["fig4_fwd_rev"] = (
        "\\textbf{Whole-protein matches and fragments among the flagged "
        "overlaps.} The overlap measure is directional and reaches 1.0 both "
        "when the corpus holds the same protein and when the test sequence "
        "is a sub-construct of a longer corpus entry. Measured in the reverse "
        "direction the two separate: the same protein lies near 1.0 on both "
        "axes, a contained sequence near its length ratio on the vertical "
        f"axis. {n_same} of {len(cands)} flagged pairs are whole-protein "
        "matches; the dashed rule marks 0.80.")


# ===================================================== rerun figure

def fig_rerun(runs, out, key="full", arms=None):
    """Full vs restricted delta, one row per split x arm."""
    import matplotlib.pyplot as plt
    g = {}
    for r in runs:
        arm = r.get("arm", "?")
        if arms and arm not in arms:
            continue
        g.setdefault((r.get("split", "?"), arm), {})[
            r.get("label", "full")] = r
    items = [(k, v) for k, v in g.items()
             if any(x["deltas"].get(key) for x in v.values())]
    if not items:
        return
    if len(items) > 8:
        log(f"    fig5_rerun_{key}: {len(items)} rows; pass --arms to filter")
    style()
    fig, ax = plt.subplots(figsize=(5.4, 0.82 * len(items) + 1.4))
    ax.set_axisbelow(True)
    ax.xaxis.grid(True)
    ax.axvline(0, color=INK3, lw=1.0, zorder=2)
    ticks, labels = [], []
    for i, ((split, arm), rr) in enumerate(items):
        base = -i
        ticks.append(base)
        labels.append(f"{split}\n{arm}")
        for label, c, dy in (("full", C_FULL, 0.17),
                             ("restricted", C_REST, -0.17)):
            r = rr.get(label)
            if not r or not r["deltas"].get(key):
                continue
            d = r["deltas"][key]
            yy = base + dy
            ax.plot([d["lo"], d["hi"]], [yy, yy], color=c, lw=2.0,
                    solid_capstyle="round", zorder=3)
            ax.plot([d["delta"]], [yy], "o", color=c, ms=6.5, mec=SURFACE,
                    mew=1.2, zorder=4, label=label if i == 0 else None)run_rerun
            ax.annotate(f"{d['delta']:+.3f}", (d["hi"], yy), xytext=(6, 0),
                        textcoords="offset points", va="center",
                        fontsize=7.5, color=INK2, annotation_clip=False)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-len(items) + 0.45, 0.55)
    ax.set_xlabel("Spearman, field minus control (95% CI)")
    ax.set_title("Effect of removing overlapping test proteins",
                 loc="left", fontsize=10, pad=8)
    ax.legend(fontsize=8, labelcolor=INK2, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.17 - 0.018 * len(items)),
              handlelength=1.2)
    bare(ax, ("top", "right", "left"))
    save(fig, out, f"fig5_rerun_{key}")
    plt.close(fig)
    ns = {r.get("split"): (r["n_test"], r.get("n_excluded", 0))
          for r in runs if r.get("label") == "restricted"}
    detail = "; ".join(f"{k}: {v[1]} of {v[0] + v[1]} removed"
                       for k, v in ns.items())
    CAPTIONS[f"fig5_rerun_{key}"] = (
        "\\textbf{Field-arm advantage before and after removing overlapping "
        "test proteins.} Points are the field-minus-control difference in "
        + ("composition-controlled Spearman, after cross-fitted "
           "residualisation of both predictions and target on the full "
           "400-dimensional dipeptide matrix"
           if key == "full" else "plain Spearman, without composition "
           "control") +
        ", averaged over seeds; bars are 95\\% intervals from a paired "
        "bootstrap over test sequences with the seed average taken within "
        "each resample. The restricted condition removes every test protein "
        f"flagged in Fig. 1 at or above the reported level ({detail}). The "
        "estimator is identical in both conditions; only the rows differ.")


# ===================================================== tables and driver

def table_rerun(runs, out, key="full"):
    """Write the complete rerun numbers as Markdown."""
    rows = []
    for r in runs:
        d = r["deltas"].get(key)
        if d:
            rows.append((r.get("split", "?"), r.get("arm", "?"),
                         r.get("label", "full"), r["n_test"],
                         r.get("n_excluded", 0), d["delta"], d["lo"],
                         d["hi"]))
    if not rows:
        return
    rows.sort(key=lambda t: (t[0], t[1], t[2] != "full"))
    out_lines = ["| split | head | rows | removed | delta | 95% CI |",
                 "|---|---|---:|---:|---:|---|"]
    for sp, arm, lab, n, ex, d, lo, hi in rows:
        star = " *" if (lo > 0 or hi < 0) else ""
        out_lines.append(
            f"| {sp} | {arm} | {lab}, n={n} | {ex} | {d:+.4f}{star} "
            f"| [{lo:+.4f}, {hi:+.4f}] |")
    out_lines += ["", "`*` interval excludes zero."]
    f = out / f"rerun_{key}.md"
    f.write_text("\n".join(out_lines) + "\n")
    log(f"    {f}")


def all_figures(res, out):
    fig_ladder(res, out)
    fig_null(res, out)
    fig_exposure(res, out)
    fig_fwd_rev(res, out)


def write_captions(out):
    if not CAPTIONS:
        return
    lines = ["# Figure captions", "",
             "LaTeX-ready. Generated by `make_figures.py`; regenerating "
             "refreshes the numbers in the text.", ""]
    for name, text in CAPTIONS.items():
        lines += [f"## {name}", "", text, ""]
    (out / "captions.md").write_text("\n".join(lines))
    log(f"    {out / 'captions.md'}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audit", default="overlap_audit",
                   help="directory holding results.json from overlap_audit.py")
    p.add_argument("--rerun", default=None,
                   help="directory of stack_check.py --json outputs")
    p.add_argument("--out", default=None,
                   help="where figures go (default: --audit)")
    p.add_argument("--arms", nargs="+", default=None,
                   help="heads to draw in the rerun figure (table keeps all)")
    a = p.parse_args()

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        raise SystemExit("matplotlib is not installed: pip install matplotlib")

    out = Path(a.out or a.audit)
    out.mkdir(parents=True, exist_ok=True)
    log("=" * 70)
    log("  FIGURES")
    log("=" * 70)

    res_path = Path(a.audit) / "results.json"
    if res_path.exists():
        all_figures(json.loads(res_path.read_text()), out)
    else:
        log(f"  no {res_path} -- skipping the audit figures")

    if a.rerun:
        runs = []
        for f in sorted(Path(a.rerun).glob("*.json")):
            d = json.loads(f.read_text())
            if "deltas" in d:
                runs.append(d)
        if runs:
            for key in ("full", "plain"):
                table_rerun(runs, out, key)
                fig_rerun(runs, out, key, a.arms)
        else:
            log(f"  no stack_check JSON in {a.rerun}")

    write_captions(out)
    log("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
