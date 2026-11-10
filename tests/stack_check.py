#!/usr/bin/env python
"""Composition control for the voxel ablation.

Tests whether the voxel arm's Meltome advantage survives residualisation
against amino-acid composition.  Three levels of control:

* **plain** -- raw per-arm Spearman, no control.
* **partial | scalar** -- residualise on the control arm's prediction
  (one direction only; overstates what survives).
* **partial | full** -- residualise on the full 400-d dipeptide matrix,
  cross-fitted out of fold.  This is the number to quote.

Usage:

    python stack_check.py
    python stack_check.py --run voxel res_vox mt_vox \\
                          --run novox res_nov mt_nov
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from flip_meltome import feat_aa_comp, feat_dipep, split_idx


def exclusion_mask(seqs, exclude_csv, split, call, label="exclude"):
    """Keep-mask over test rows, dropping sequences flagged as corpus overlaps.

    Joined on SHA-1 of the sequence (not row index) to avoid silent
    misalignment between the FLIP CSV row numbering and the test subset
    ordering.
    """
    df = pd.read_csv(exclude_csv)
    for col in ("id", "seq_sha1"):
        if col not in df.columns:
            raise SystemExit(
                f"{exclude_csv} has no `{col}` column (has "
                f"{sorted(df.columns)}). Regenerate with overlap_audit.py.")
    cc = next((c for c in ("forward", "containment") if c in df.columns), None)
    if cc is None:
        raise SystemExit(
            f"{exclude_csv} has neither `forward` nor `containment` (has "
            f"{sorted(df.columns)}).")
    if split:
        df = df[df["id"].astype(str).str.startswith(split + ":")]
    df = df[df[cc] >= call]
    bad = set(df["seq_sha1"])
    keep = np.array([hashlib.sha1(x.encode()).hexdigest() not in bad
                     for x in seqs], dtype=bool)
    n_drop = int((~keep).sum())
    print(f"  EXCLUDING {n_drop}/{len(keep)} test sequences at `{cc}` "
          f">= {call:.2f}"
          + (f" (split filter `{split}`)" if split else "")
          + f"  -> {int(keep.sum())} remain")
    if n_drop == 0 and len(bad):
        print(f"  !! {len(bad)} flagged hashes but none matched these test rows.")
    if int(keep.sum()) < 50:
        raise SystemExit(
            f"only {int(keep.sum())} rows survive -- too few to bootstrap.")
    return keep


def restrict_mask(seqs, table_csv, group, label="restrict"):
    """Keep-mask keeping only sequences whose group in *table_csv* matches *group*.

    Same SHA-1 join as ``exclusion_mask``.
    """
    df = pd.read_csv(table_csv)
    for col in ("seq_sha1", "group"):
        if col not in df.columns:
            raise SystemExit(
                f"{table_csv} has no `{col}` column (has "
                f"{sorted(df.columns)[:8]}).")
    want = set(df.loc[df["group"].astype(str) == str(group), "seq_sha1"])
    if not want:
        have = sorted(set(df["group"].astype(str)))
        raise SystemExit(
            f"no rows with group `{group}` in {table_csv}.\n"
            f"  available: {', '.join(have)}")
    keep = np.array([hashlib.sha1(x.encode()).hexdigest() in want
                     for x in seqs], dtype=bool)
    print(f"  RESTRICTING to group `{group}`: {int(keep.sum())}/{len(keep)} "
          f"test sequences")
    if int(keep.sum()) == 0:
        raise SystemExit(
            f"group `{group}` matched no test rows.")
    return keep


ALPHAS = np.logspace(-2, 5, 15)
CONTROLS = {"dipep": feat_dipep, "aa_comp": feat_aa_comp}


def load_rows(emb_dir):
    """Load per-row metadata from a cache without the heavy tensors."""
    emb_dir = Path(emb_dir)
    meta_path = emb_dir / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"no meta.json in {emb_dir}")
    meta = json.loads(meta_path.read_text())
    out = {"sequence": [], "set": [], "validation": [], "seq_len": []}
    for c in meta["chunks"]:
        p = torch.load(emb_dir / c, weights_only=False)
        if "sequence" not in p:
            raise SystemExit(
                f"{emb_dir / c} has no `sequence` field. Re-embed with the "
                f"current flip_meltome.py.")
        out["sequence"] += list(p["sequence"])
        out["set"] += list(p["set"])
        out["validation"] += list(p["validation"])
        out["seq_len"].append(p["seq_len"])
        del p
    out["seq_len"] = torch.cat(out["seq_len"])
    return out, meta


def test_sequences(emb_dir, pred, label):
    """Test-split sequences aligned to ``test_predictions.csv`` row order.

    Alignment is asserted via ``seq_len``, not assumed.
    """
    d, meta = load_rows(emb_dir)
    _, _, te = split_idx(d, verbose=False)
    seqs = [d["sequence"][i] for i in te]
    got = np.array([len(s) for s in seqs])
    want = pred["seq_len"].to_numpy()
    if len(got) != len(want):
        raise SystemExit(
            f"[{label}] {len(got)} test rows in {emb_dir} but {len(want)} in "
            f"predictions. Different embed runs?")
    bad = int((got != want).sum())
    if bad:
        raise SystemExit(
            f"[{label}] seq_len disagrees on {bad}/{len(got)} test rows.")
    return seqs, meta


def arm_draws(res_dir, pred, arm, mode="auto"):
    """Load per-seed predictions when available, else the best-val seed.

    Returns ``([prediction_arrays], description)``.
    """
    if mode != "best":
        p = Path(res_dir) / "test_predictions_seeds.csv"
        if p.exists():
            df = pd.read_csv(p)
            cols = [c for c in df.columns
                    if re.fullmatch(rf"{re.escape(arm)}_s\d+", c)]
            cols.sort(key=lambda c: int(c.rsplit("_s", 1)[1]))
            if cols:
                return ([df[c].to_numpy() for c in cols],
                        f"mean over {len(cols)} seeds")
    if mode == "best":
        return [pred[arm].to_numpy()], "best-validation seed only (--seeds best)"
    if (Path(res_dir) / "test_predictions_seeds.csv").exists():
        return ([pred[arm].to_numpy()],
                "single deterministic vector (no per-seed columns for this arm)")
    return ([pred[arm].to_numpy()],
            "best-validation seed only -- no test_predictions_seeds.csv")


def cross_fit_resid(X, y, n_splits=5, seed=0):
    """Out-of-fold residuals: ``y`` minus a cross-fitted RidgeCV prediction from ``X``."""
    out = np.empty(len(y), dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for a, b in kf.split(X):
        m = RidgeCV(alphas=ALPHAS).fit(X[a], y[a])
        out[b] = y[b] - m.predict(X[b])
    return out


def resid_pair(arm_vals, y_vals, X):
    """Rank-transform both sides, then residualise on *X*.

    Returns ``(r_arm, r_y)`` for a partial Spearman.
    """
    Xs = StandardScaler().fit_transform(X)
    return (cross_fit_resid(Xs, rankdata(arm_vals)),
            cross_fit_resid(Xs, rankdata(y_vals)))


def boot_ci(f, n, n_boot, seed=0):
    """Percentile 95% CI for ``f(idx)`` under bootstrap resampling."""
    rng = np.random.default_rng(seed)
    vals = np.array([f(rng.integers(0, n, n)) for _ in range(n_boot)])
    vals = vals[np.isfinite(vals)]
    if len(vals) < n_boot // 2:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _verdict(lo, hi):
    if not np.isfinite(lo):
        return "CI failed"
    return "separable" if lo > 0 or hi < 0 else "not separable"


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", nargs=3,
                   metavar=("LABEL", "RESULTS_DIR", "EMB_DIR"),
                   help="repeat exactly twice; the first is the baseline")
    p.add_argument("--arm", default="ridge_mean",
                   help="embedding arm to test")
    p.add_argument("--control", default="dipep", choices=sorted(CONTROLS),
                   help="composition control to residualise against")
    p.add_argument("--seeds", choices=("auto", "best"), default="auto",
                   help="auto: average over seeds when available")
    p.add_argument("--exclude", metavar="CANDIDATES_CSV",
                   help="overlap_audit.py's candidates.csv; drop flagged seqs")
    p.add_argument("--exclude-split", metavar="NAME")
    p.add_argument("--exclude-call", type=float, default=0.80)
    p.add_argument("--split-name", metavar="NAME",
                   help="FLIP split name (recorded in JSON for plotting)")
    p.add_argument("--restrict-to", metavar="TABLE_CSV",
                   help="keep only test sequences in one group of this table")
    p.add_argument("--restrict-group", metavar="NAME")
    p.add_argument("--json", metavar="OUT",
                   help="write results to this JSON file")
    p.add_argument("--label", default=None)
    p.add_argument("--n-boot", type=int, default=4000)
    p.add_argument("--folds", type=int, default=5)
    a = p.parse_args()

    runs = a.run or [["novox", "res_nov_e19", "mt_nov_e19"],
                     ["voxel", "res_vox_e19", "mt_vox_e19"]]
    if len(runs) != 2:
        raise SystemExit("pass --run exactly twice (baseline first)")
    ARM, CTRL = a.arm, a.control

    print("=" * 78)
    print(f"COMPOSITION CONTROL   arm={ARM}   control={CTRL} (full matrix, "
          f"cross-fitted)")
    print("=" * 78)

    rows, resids, y_ref = {}, {}, None
    n_seed, plains = {}, {}
    keep, keep_seqs = None, None
    for label, res_dir, emb_dir in runs:
        pred_path = Path(res_dir) / "test_predictions.csv"
        if not pred_path.exists():
            raise SystemExit(f"no {pred_path}; run `flip_meltome.py run` first")
        pred = pd.read_csv(pred_path)
        for col in (ARM, CTRL, "y_true", "seq_len"):
            if col not in pred.columns:
                raise SystemExit(
                    f"[{label}] {pred_path} has no `{col}` column.")

        seqs, meta = test_sequences(emb_dir, pred, label)
        draws, src = arm_draws(res_dir, pred, ARM, a.seeds)

        if a.exclude or a.restrict_to:
            if keep is None:
                keep = np.ones(len(seqs), dtype=bool)
                if a.exclude:
                    keep &= exclusion_mask(seqs, a.exclude, a.exclude_split,
                                           a.exclude_call)
                if a.restrict_to:
                    keep &= restrict_mask(seqs, a.restrict_to,
                                          a.restrict_group)
                if int(keep.sum()) < 50:
                    raise SystemExit(
                        f"only {int(keep.sum())} rows survive the combined mask.")
                keep_seqs = list(seqs)
            elif [len(x) for x in seqs] != [len(x) for x in keep_seqs]:
                raise SystemExit(
                    "the two runs have different test sequences.")
            seqs = [x for x, k in zip(seqs, keep) if k]
            pred = pred[keep].reset_index(drop=True)
            draws = [d[keep] for d in draws]

        y = pred["y_true"].to_numpy()
        if y_ref is None:
            y_ref = y
        elif len(y) != len(y_ref) or not np.allclose(y, y_ref):
            raise SystemExit(
                "the two runs have different y_true -- different test sets.")
        et = meta.get("embed_tensor", "forward_int (pre-flag)")
        print(f"  {label:<7} {res_dir:<14} {emb_dir:<14} "
              f"{len(seqs)} test seqs  embed_tensor={et}")

        n_seed[label] = len(draws)
        print(f"          {ARM}: {src}")

        X = CONTROLS[CTRL]({"sequence": seqs})
        ctrl_pred = pred[CTRL].to_numpy()
        _, r_y = resid_pair(draws[0], y, X)
        _, s_y = resid_pair(draws[0], y, ctrl_pred[:, None])
        r_arms = [resid_pair(p_s, y, X)[0] for p_s in draws]
        s_arms = [resid_pair(p_s, y, ctrl_pred[:, None])[0] for p_s in draws]

        rows[label] = {
            "plain": float(np.mean([spearmanr(p_s, y).correlation
                                    for p_s in draws])),
            "overlap": float(np.mean([spearmanr(p_s, ctrl_pred).correlation
                                      for p_s in draws])),
            "scalar": float(np.mean([pearsonr(sa, s_y)[0] for sa in s_arms])),
            "full": float(np.mean([pearsonr(ra, r_y)[0] for ra in r_arms])),
            "ctrl_alone": spearmanr(ctrl_pred, y).correlation,
        }
        resids[label] = {"full": (r_arms, r_y), "scalar": (s_arms, s_y)}
        plains[label] = draws

    (lo_lbl, *_), (hi_lbl, *_) = runs
    n = len(y_ref)

    print(f"\n  {CTRL} alone: SCC {rows[hi_lbl]['ctrl_alone']:.4f}")
    print(f"\n  {'run':<9}{'plain SCC':>12}{'corr w/ ' + CTRL:>18}"
          f"{'partial|scalar':>16}{'partial|full':>14}")
    for label, *_ in runs:
        r = rows[label]
        print(f"  {label:<9}{r['plain']:>12.4f}{r['overlap']:>18.4f}"
              f"{r['scalar']:>16.4f}{r['full']:>14.4f}")

    ns = sorted(set(n_seed.values()))
    seed_note = (f", {ns[0]} seeds averaged" if ns == [ns[0]] and ns[0] > 1
                 else f", seeds per run {n_seed}" if len(ns) > 1 else "")
    if len(ns) > 1:
        print(f"\n  !! the two runs have different seed counts {n_seed}")
    print(f"\n  PAIRED DELTAS  ({hi_lbl} - {lo_lbl}), 95% CI from {a.n_boot} "
          f"bootstrap resamples of {n} test sequences{seed_note}")
    plain_d = rows[hi_lbl]["plain"] - rows[lo_lbl]["plain"]

    def _mean_scc(draws, idx):
        return np.mean([spearmanr(p_s[idx], y_ref[idx]).correlation
                        for p_s in draws])

    def plain_f(idx):
        return (_mean_scc(plains[hi_lbl], idx)
                - _mean_scc(plains[lo_lbl], idx))

    lo, hi = boot_ci(plain_f, n, a.n_boot)
    print(f"    {'plain':<29}{plain_d:+.4f}  [{lo:+.4f}, {hi:+.4f}]"
          f"   {_verdict(lo, hi)}")
    out = {"label": a.label or ("restricted" if a.exclude else "full"),
           "split": a.split_name or a.exclude_split or "?",
           "arm": ARM, "control": CTRL, "n_test": int(n),
           "n_excluded": int((~keep).sum()) if keep is not None else 0,
           "exclude_call": a.exclude_call if a.exclude else None,
           "exclude_split": a.exclude_split,
           "restrict_group": a.restrict_group,
           "baseline": lo_lbl, "compared": hi_lbl,
           "n_boot": a.n_boot, "seeds": n_seed,
           "per_run": rows,
           "deltas": {"plain": {"delta": float(plain_d),
                                "lo": float(lo), "hi": float(hi)}}}

    n_ctrl = CONTROLS[CTRL]({"sequence": ["AAAA"]}).shape[1]
    deltas = {}
    for key, name in (("scalar", f"partial | {CTRL} scalar"),
                      ("full", f"partial | {CTRL} full {n_ctrl}d")):
        d = rows[hi_lbl][key] - rows[lo_lbl][key]
        deltas[key] = d

        def f(idx, k=key):
            ah, yh = resids[hi_lbl][k]
            al, yl = resids[lo_lbl][k]
            return (np.mean([pearsonr(a_s[idx], yh[idx])[0] for a_s in ah])
                    - np.mean([pearsonr(a_s[idx], yl[idx])[0] for a_s in al]))

        lo, hi = boot_ci(f, n, a.n_boot)
        print(f"    {name:<29}{d:+.4f}  [{lo:+.4f}, {hi:+.4f}]   "
              f"{_verdict(lo, hi)}")
        out["deltas"][key] = {"delta": float(d), "lo": float(lo),
                              "hi": float(hi), "name": name}

    print("\n" + "=" * 78)
    full_d = deltas.get("full")
    if plain_d > 0 and full_d is not None and full_d > plain_d:
        print(f"  SUPPRESSION: controlled delta is larger than raw "
              f"({plain_d:+.4f} -> {full_d:+.4f}).")
        print(f"  The shared compositional component dilutes the gap; removing "
              f"it reveals that {hi_lbl}'s advantage lives in the residual.")
    elif plain_d > 0 and full_d is not None and full_d < 0:
        print(f"  SIGN FLIP: raw delta favours {hi_lbl} ({plain_d:+.4f}) but "
              f"controlled delta favours {lo_lbl} ({full_d:+.4f}).")
        print(f"  {hi_lbl}'s raw advantage was composition.")
    elif plain_d > 0 and full_d is not None:
        frac = full_d / plain_d
        print(f"  {frac * 100:.0f}% of {hi_lbl}'s advantage survives the full "
              f"{CTRL} control ({plain_d:+.4f} -> {full_d:+.4f}).")
    elif full_d is not None:
        print(f"  {hi_lbl} has no plain advantage ({plain_d:+.4f}); "
              f"controlled delta is {full_d:+.4f}.")

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"\n  -> {a.json}")
    print("  Quote the `partial | full` row.")
    print("=" * 78)


if __name__ == "__main__":
    main()
