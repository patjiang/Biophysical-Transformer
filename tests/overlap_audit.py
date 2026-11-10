#!/usr/bin/env python
"""Pretraining-exposure audit: BioT's corpus, and ESM's, against FLIP test splits.

Usage::

    python overlap_audit.py --flip flip_data/meltome --species --uniref

Everything offline runs by default.  ``--uniref`` and ``--species`` need
network.  ``--plots-only`` re-plots from cached results.

Outputs: ``results.json``, ``ladder.csv``, ``candidates.csv``,
``exposure.csv``, and figures 1-4 (PDF + PNG).
"""

import argparse
import csv as _csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

NK = 8000
AA20 = "ACDEFGHIKLMNPQRSTVWY"
LADDER = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 1.00]

PAL = {"human": "#2a78d6", "human_cell": "#eb6834", "mixed_split": "#1baf7a"}
SPARE = ["#eda100", "#e87ba4", "#6250d6"]
INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e2e1dc"

MELTOME_TAXA = {
    "9606": "Homo sapiens", "10090": "Mus musculus",
    "7955": "Danio rerio", "7227": "Drosophila melanogaster",
    "6239": "Caenorhabditis elegans", "3702": "Arabidopsis thaliana",
    "4932": "Saccharomyces cerevisiae", "562": "Escherichia coli",
    "1423": "Bacillus subtilis", "274": "Thermus thermophilus",
    "1332": "Picrophilus torridus", "188908": "Oleispira antarctica",
    "1422": "Geobacillus stearothermophilus",
}

ESM1V_SNAPSHOT = "2020-03"
ESM1B_SNAPSHOT = "2018-03"


def log(m=""):
    print(m, flush=True)


def rule(c="="):
    log(c * 76)


def color_for(split, i):
    return PAL.get(split, SPARE[i % len(SPARE)])


# =========================================================== corpus loading

def load_project():
    """Import the data module from ``src``."""
    from src.data import PreprocessedBiophysicalDataset, seqs_to_tens
    return PreprocessedBiophysicalDataset, seqs_to_tens


def clean_tokens(seqs):
    """Extract 3-mer tokens the model reads, mirroring ``seqs_to_tens``."""
    return [(s.split("_")[0] if len(s.split("_")[0]) == 3 else None)
            for s in seqs]


def kmer_ids(toks, enc):
    good = [t for t in toks if t is not None]
    if not good:
        return set()
    return {int(x) for x in enc(good).tolist() if 0 <= int(x) < NK}


def seq_kmer_ids(s, enc):
    k = [s[i:i + 3] for i in range(len(s) - 2)]
    k = [x for x in k if all(c in AA20 for c in x)]
    if not k:
        return set()
    return {int(x) for x in enc(k).tolist() if 0 <= int(x) < NK}


def corpus_items(ds_cls, enc, d, limit=None):
    """Load corpus proteins as sets of 3-mer ids."""
    ds = ds_cls(str(d))
    n = min(len(ds), limit) if limit else len(ds)
    log(f"  {d}: {len(ds)} items" + (f", reading {n}" if limit else ""))
    items, ordinfo = [], []
    for i in range(n):
        it = ds[i]
        seqs = it.get("seqs") if isinstance(it, dict) else None
        if not seqs:
            continue
        toks = clean_tokens(seqs)
        ids = kmer_ids(toks, enc)
        if not ids:
            continue
        items.append({"pdbid": str(it.get("pdbid") or f"item{i}"),
                      "ids": ids,
                      "n_tok": sum(t is not None for t in toks),
                      "n_pad": sum(t is None for t in toks)})
        if len(ordinfo) < 200:
            ordinfo.append((toks, (it.get("metadata") or {})
                            .get("original_index_order")))
        if (i + 1) % 4000 == 0:
            log(f"    {i + 1}/{n}")
    log(f"  {len(items)} usable proteins")
    return items, ordinfo


# ======================================================== [0] token ordering

def order_report(ordinfo):
    """Check whether stored token order matches residue order."""
    rule()
    log("  [0]  is the stored token order residue order?")
    rule()
    if not ordinfo:
        log("  nothing to measure")
        return None
    tot = ok = 0
    for toks, _ in ordinfo:
        t = [x for x in toks if x is not None]
        for a, b in zip(t, t[1:]):
            tot += 1
            ok += (a[1:] == b[:-1])
    frac = ok / max(tot, 1)
    log(f"  adjacent token pairs sharing the required 2 residues: "
        f"{ok}/{tot} = {frac:.3f}  (over {len(ordinfo)} proteins)")
    o = next((x[1] for x in ordinfo if x[1] is not None), None)
    if o is not None:
        oo = list(o)
        log(f"  metadata['original_index_order']: {type(oo[0]).__name__}, "
            f"first {oo[:6]}, duplicates {len(oo) - len(set(oo))}")
    if frac > 0.95:
        log("  Token order IS residue order.")
    else:
        log("  Token order is NOT residue order.")
    return frac


# ==================================================== [4] containment + null

def build_corpus(items):
    """Build a sparse token-to-protein matrix for containment queries."""
    import numpy as np
    from scipy import sparse
    ri, ci = [], []
    for r, it in enumerate(items):
        ri.extend([r] * len(it["ids"]))
        ci.extend(it["ids"])
    C = sparse.csr_matrix((np.ones(len(ri), dtype=np.float32), (ri, ci)),
                          shape=(len(items), NK))
    log(f"  corpus 3-mer matrix: {C.shape[0]} proteins x {NK} tokens, "
        f"{C.nnz} nonzeros")
    return C.T.tocsr()


def max_containment(CT, idsets, chunk=2000):
    """Best containment of each query against any single corpus protein."""
    import numpy as np
    from scipy import sparse
    qr, qc, sz = [], [], []
    for r, ids in enumerate(idsets):
        qr.extend([r] * len(ids))
        qc.extend(ids)
        sz.append(len(ids))
    Q = sparse.csr_matrix((np.ones(len(qr), dtype=np.float32), (qr, qc)),
                          shape=(len(sz), NK))
    sz = np.array(sz, dtype=np.float32)
    best = np.zeros(len(sz), dtype=np.float32)
    arg = np.zeros(len(sz), dtype=np.int64)
    for s in range(0, Q.shape[0], chunk):
        P = (Q[s:s + chunk] @ CT).toarray()
        P /= sz[s:s + chunk, None]
        best[s:s + chunk] = P.max(1)
        arg[s:s + chunk] = P.argmax(1)
    return best, arg


def shuffled_null(CT, keep, enc, reps, min_kmers, seed=0):
    """Composition-matched null: shuffle each test sequence and recompute."""
    import numpy as np
    rng = np.random.default_rng(seed)
    sets, who = [], []
    for _, s in keep:
        n_obs = len(seq_kmer_ids(s, enc))
        a = np.frombuffer(s.encode(), dtype=np.uint8)
        for _ in range(reps):
            t = a.copy()
            rng.shuffle(t)
            sh = seq_kmer_ids(t.tobytes().decode(), enc)
            if len(sh) >= min_kmers:
                sets.append(sh)
                who.append(n_obs)
    if not sets:
        return None
    vals, _ = max_containment(CT, sets)
    return np.array(who), np.array(vals)


def calibrate(obs_n, obs_v, null_n, null_v, alpha, nbins=10):
    """Bin-wise thresholds from the shuffled null."""
    import numpy as np
    edges = np.quantile(obs_n, np.linspace(0, 1, nbins + 1))
    edges[0] -= 1
    edges[-1] += 1
    thr = np.zeros(len(obs_v))
    hard = np.zeros(len(obs_v))
    rows = []
    for b in range(nbins):
        lo, hi = edges[b], edges[b + 1]
        m = (obs_n > lo) & (obs_n <= hi)
        nm = (null_n > lo) & (null_n <= hi)
        if not m.any():
            continue
        pool = null_v[nm]
        t = float(np.quantile(pool, 1 - alpha)) if pool.size >= 20 else 1.01
        mx = float(pool.max()) if pool.size else 1.01
        thr[m], hard[m] = t, mx
        rows.append({"lo": int(lo) + 1, "hi": int(hi), "n": int(m.sum()),
                     "null_n": int(nm.sum()),
                     "null_med": float(np.median(pool)) if pool.size else None,
                     "alpha_thr": t, "null_max": mx,
                     "n_over_alpha": int((obs_v[m] > t).sum()),
                     "n_over_max": int((obs_v[m] > mx).sum())})
    return thr, hard, rows


# ========================================= [5] reverse containment

def reverse_pass(items, keep, enc, obs, arg, call):
    """Forward + reverse containment for called candidates."""
    out = []
    for k in range(len(obs)):
        if obs[k] < call:
            continue
        fid, s = keep[k]
        q = seq_kmer_ids(s, enc)
        it = items[int(arg[k])]
        inter = len(q & it["ids"])
        out.append({"id": fid, "seq_sha1": hashlib.sha1(s.encode()).hexdigest(),
                    "len": len(s), "n_kmers": len(q),
                    "forward": round(float(obs[k]), 4),
                    "reverse": round(inter / max(len(it["ids"]), 1), 4),
                    "match": it["pdbid"], "match_tokens": it["n_tok"],
                    "kind": ("same-protein"
                             if inter / max(len(it["ids"]), 1) >= 0.80
                             else "test-is-fragment")})
    return sorted(out, key=lambda r: -r["forward"])


# ============================================== [6] UniProt exposure

SCOPES = {
    "reference": "(taxonomy_id:{t}) AND (keyword:KW-1185)",
    "reviewed": "(taxonomy_id:{t}) AND (reviewed:true)",
    "all": "(taxonomy_id:{t})",
}


def uniprot_table(organisms, cache, scope="reference"):
    """Fetch ``accession -> (sequence, date_created)`` per taxon, cached."""
    import urllib.error
    import urllib.parse
    import urllib.request

    store = {"scope": scope, "taxa": {}}
    if cache.exists():
        try:
            old = json.loads(cache.read_text())
        except json.JSONDecodeError:
            old = {}
        if isinstance(old, dict) and "taxa" in old:
            if old.get("scope") == scope:
                store = old
            else:
                log(f"    cache was scope={old.get('scope')}, now {scope}"
                    f" -- discarding")
        elif isinstance(old, dict) and old:
            log(f"    cache has no taxon record -- discarding and refetching")

    base = "https://rest.uniprot.org/uniprotkb/stream"
    tmpl = SCOPES[scope]
    want = [str(o) for o in organisms]
    cached = [t for t in want if t in store["taxa"]]
    todo = [t for t in want if t not in store["taxa"]]
    log(f"    scope={scope}  {len(want)} taxa: {len(cached)} cached, "
        f"{len(todo)} to fetch")
    for org in todo:
        q = tmpl.format(t=org)
        url = (f"{base}?query={urllib.parse.quote(q)}"
               f"&fields=accession,sequence,date_created,organism_name"
               f"&format=tsv")
        try:
            with urllib.request.urlopen(url, timeout=1800) as r:
                text = r.read().decode()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            log(f"    taxon {org}: FAILED ({e})")
            continue
        lines_ = text.splitlines()
        if len(lines_) < 2:
            log(f"    taxon {org}: empty response")
            store["taxa"][org] = {}
            continue
        hdr = lines_[0].split("\t")
        try:
            ia, isq, idt = (hdr.index("Entry"), hdr.index("Sequence"),
                            hdr.index("Date of creation"))
            ion = hdr.index("Organism") if "Organism" in hdr else None
        except ValueError:
            log(f"    unexpected header: {hdr}")
            continue
        rec, seen = {}, Counter()
        for ln in lines_[1:]:
            pp = ln.split("\t")
            if len(pp) > max(ia, isq, idt):
                rec[pp[ia]] = [pp[isq], pp[idt]]
                if ion is not None and len(pp) > ion:
                    seen[pp[ion].split(" (")[0]] += 1
        store["taxa"][org] = rec
        nm = MELTOME_TAXA.get(org, "?")
        top = ", ".join(k for k, _ in seen.most_common(2))
        log(f"    taxon {org:<8} {nm:<32} {len(rec):>7} entries"
            + (f"   [{top}]" if top else ""))
        cache.write_text(json.dumps(store))
    cache.write_text(json.dumps(store))

    merged = {}
    for t in want:
        merged.update(store["taxa"].get(t, {}))
    per = {t: len(store["taxa"].get(t, {})) for t in want}
    log(f"    {len(merged)} distinct accessions over {len(want)} taxa")
    log("    per taxon: " + "  ".join(f"{t}:{n}" for t, n in per.items()))
    return merged


def exposure(flip, uni, snapshots):
    """Fraction of test proteins in UniProt before each snapshot."""
    byseq = {}
    for acc, (sq, dt) in uni.items():
        byseq.setdefault(sq, (acc, dt))
    out = {}
    for split, parts in flip.items():
        rows = parts["test"]
        hit, dates, miss = 0, [], 0
        for _, s in rows:
            r = byseq.get(s)
            if r is None:
                miss += 1
                continue
            hit += 1
            dates.append(r[1])
        rec = {"n_test": len(rows), "matched": hit, "unmatched": miss,
               "match_rate": hit / max(len(rows), 1)}
        for name, cut in snapshots.items():
            n = sum(1 for d in dates if _ym(d) and _ym(d) <= cut)
            rec[name] = {"n": n, "of_test": n / max(len(rows), 1),
                         "of_matched": n / max(hit, 1)}
        out[split] = rec
    return out


def _ym(d):
    """Parse UniProt dates to ``'YYYY-MM'``."""
    if not d:
        return None
    d = d.strip()
    if len(d) >= 7 and d[4] == "-":
        return d[:7]
    M = {m: f"{i + 1:02d}" for i, m in enumerate(
        ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
         "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
    p = d.upper().split("-")
    if len(p) == 3 and p[1] in M:
        return f"{p[2]}-{M[p[1]]}"
    return None


# ======================================================= [7] species mixture

def species_mix(weights, out_dir, do_lookup):
    """Resolve corpus PDB accessions to organisms via RCSB."""
    import urllib.error
    import urllib.request
    acc = sorted(weights)
    (out_dir / "pdb_accessions.txt").write_text("\n".join(acc) + "\n")
    log(f"  {len(acc)} distinct accessions -> {out_dir / 'pdb_accessions.txt'}")
    if not do_lookup:
        log("  organism lookup skipped (pass --species)")
        return None
    names = {}
    q = ("query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id "
         "polymer_entities{rcsb_entity_source_organism"
         "{ncbi_scientific_name}}}}")
    log(f"  querying RCSB for {len(acc)} entries")
    for s in range(0, len(acc), 50):
        body = json.dumps({"query": q,
                           "variables": {"ids": acc[s:s + 50]}}).encode()
        req = __import__("urllib.request", fromlist=["Request"]).Request(
            "https://data.rcsb.org/graphql", data=body,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read())
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            log(f"    batch at {s} failed ({e}); stopping")
            break
        for e in (d.get("data", {}).get("entries") or []):
            for pe in (e.get("polymer_entities") or []):
                for so in (pe.get("rcsb_entity_source_organism") or []):
                    if so.get("ncbi_scientific_name"):
                        names[e["rcsb_id"].upper()] = \
                            so["ncbi_scientific_name"]
                        break
        if (s // 50) % 40 == 0:
            log(f"    {min(s + 50, len(acc))}/{len(acc)}")
    if not names:
        log("  no organisms resolved")
        return None
    per = Counter()
    for a, w in weights.items():
        if a in names:
            per[names[a]] += w
    tot = sum(per.values())
    log(f"\n  resolved {len(names)}/{len(acc)} accessions, {tot} proteins\n")
    log(f"    {'organism':<42}{'proteins':>10}{'share':>9}")
    for nm, c in per.most_common(15):
        log(f"    {nm:<42}{c:>10}{c / tot:>8.1%}")
    human = per.get("Homo sapiens", 0) / max(tot, 1)
    log(f"\n    Homo sapiens share: {human:.1%}")
    return {"per_organism": dict(per), "human_share": human, "total": tot}


# ============================================================ FLIP loading

def flip_sequences(flip_dir):
    """Load FLIP CSVs into ``{split: {train: [...], test: [...]}}``."""
    import pandas as pd
    out = {}
    for csvf in sorted(Path(flip_dir).glob("*.csv")):
        df = pd.read_csv(csvf)
        if "sequence" not in df.columns:
            continue
        setcol = df["set"] if "set" in df.columns else None
        rec = {"train": [], "test": []}
        for i, row in df.iterrows():
            part = "test" if (setcol is not None and row["set"] == "test") \
                else "train"
            rec[part].append((f"{csvf.stem}:{i}", str(row["sequence"]).upper()))
        out[csvf.stem] = rec
        log(f"  {csvf.name}: {len(rec['train'])} train, "
            f"{len(rec['test'])} test")
    if not out:
        raise SystemExit(f"no CSVs with a `sequence` column in {flip_dir}")
    return out


# =================================================================== PLOTS

def _style():
    import matplotlib as mpl
    mpl.use("Agg")
    mpl.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb", "font.size": 9,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.titlecolor": INK, "axes.linewidth": 0.8,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "grid.color": GRID, "grid.linewidth": 0.7,
        "legend.frameon": False, "pdf.fonttype": 42,
    })


def _finish(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=220)
    log(f"    {out / name}.pdf / .png")


def fig_ladder(res, out):
    """Containment ladder by split."""
    import matplotlib.pyplot as plt
    _style()
    x = list(range(len(LADDER)))
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    ymax = 0.0
    for i, (split, r) in enumerate(res["splits"].items()):
        y = [100 * r["ladder"][f"{t:.2f}"] for t in LADDER]
        ymax = max(ymax, max(y))
        c = color_for(split, i)
        ax.plot(x, y, "-o", color=c, lw=1.8, ms=5.5, mec="#fcfcfb",
                mew=1.2, label=split, zorder=3)
        ax.annotate(f"{split}  {y[0]:.1f}% ", (x[0], y[0]),
                    textcoords="offset points", xytext=(-7, 0), ha="right",
                    color=c, fontsize=8, va="center", annotation_clip=False)
    call_i = min(range(len(LADDER)),
                 key=lambda k: abs(LADDER[k] - res["call"]))
    ax.axvline(call_i, color=INK3, lw=0.9, ls=(0, (3, 3)), zorder=1)
    ax.set_ylim(0, max(ymax * 1.18, 1.0))
    ax.annotate(f"reported\nthreshold", (call_i, 0), xytext=(4, 6),
                textcoords="offset points", color=INK3, fontsize=7.5,
                va="bottom")
    ax.set_xlim(-2.0, len(LADDER) - 0.75)
    ax.set_xlabel("3-mer containment against the best single corpus protein")
    ax.set_ylabel("FLIP test proteins at or above (%)")
    ax.set_title("How much of FLIP's test set is in BioT's corpus",
                 loc="left", fontsize=10, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.2f}" for t in LADDER])
    ax.legend(fontsize=8, labelcolor=INK2, ncol=len(res["splits"]),
              loc="upper center", bbox_to_anchor=(0.5, -0.22),
              columnspacing=1.6, handlelength=1.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _finish(fig, out, "fig1_ladder")
    plt.close(fig)


def fig_null(res, out):
    """Observed vs shuffled containment, one panel per split."""
    import matplotlib.pyplot as plt
    import numpy as np
    _style()
    sp = res["splits"]
    fig, axes = plt.subplots(1, len(sp), figsize=(2.6 * len(sp), 2.9),
                             sharey=True)
    if len(sp) == 1:
        axes = [axes]
    for i, ((split, r), ax) in enumerate(zip(sp.items(), axes)):
        c = color_for(split, i)
        for key, ls, lab in (("obs_ecdf", "-", "observed"),
                             ("null_ecdf", (0, (4, 2)), "shuffled")):
            e = r.get(key)
            if not e:
                continue
            ax.plot(e["x"], 100 * np.array(e["y"]), ls=ls, color=c, lw=1.8,
                    label=lab, zorder=3)
        ax.axvline(res["call"], color=INK3, lw=0.9, ls=(0, (3, 3)), zorder=1)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)
        ax.set_title(split, loc="left", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_xlabel("containment")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if i == 0:
            ax.set_ylabel("test proteins at or below (%)")
            ax.legend(loc="lower right", fontsize=7.5, labelcolor=INK2)
    fig.suptitle("Observed containment against a composition-matched null",
                 x=0.005, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _finish(fig, out, "fig2_null")
    plt.close(fig)


def fig_exposure(res, out):
    """BioT exposure beside ESM's, grouped bars."""
    import matplotlib.pyplot as plt
    import numpy as np
    ex = res.get("exposure")
    if not ex:
        return
    _style()
    splits = [s for s in res["splits"] if s in ex]
    snap1b = res.get("esm1b_snapshot", ESM1B_SNAPSHOT)
    models = [("BioT", lambda s: 100 * res["splits"][s]["ladder"]["0.95"]),
              ("ESM-1b", lambda s: 100 * ex[s]["esm1b"]["of_test"]),
              ("ESM-1v", lambda s: 100 * ex[s]["esm1v"]["of_test"])]
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    w = 0.26
    x = np.arange(len(splits))
    for j, (name, f) in enumerate(models):
        v = [f(s) for s in splits]
        c = ["#2a78d6", "#eb6834", "#1baf7a"][j]
        b = ax.bar(x + (j - 1) * (w + 0.02), v, w, label=name, color=c,
                   zorder=3, linewidth=0)
        for rect, val in zip(b, v):
            ax.annotate(f"{val:.1f}", (rect.get_x() + rect.get_width() / 2,
                                       val), xytext=(0, 3),
                        textcoords="offset points", ha="center",
                        fontsize=7.5, color=INK2)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("FLIP test proteins seen in pretraining (%)")
    ax.set_title("Pretraining exposure to the FLIP test splits, by model",
                 loc="left", fontsize=10, pad=10)
    ax.set_ylim(0, 108)
    ax.legend(fontsize=8, labelcolor=INK2, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), columnspacing=2.2,
              handlelength=1.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.annotate(
        f"BioT: >=0.95 3-mer containment against its own corpus.\n"
        f"ESM-1b: UniRef50 (50% identity), snapshot {snap1b}.  "
        f"ESM-1v: UniRef90 {ESM1V_SNAPSHOT}, 98M sequences.\n"
        f"ESM bars count test proteins present in UniProt before the "
        f"snapshot.",
        (0, -0.30), xycoords="axes fraction", fontsize=6.8,
        color=INK3, va="top")
    _finish(fig, out, "fig3_exposure")
    plt.close(fig)


def fig_fwd_rev(res, out):
    """Forward vs reverse containment scatter for called candidates."""
    import matplotlib.pyplot as plt
    cands = res.get("candidates") or []
    if not cands:
        return
    _style()
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.axhline(0.80, color=INK3, lw=0.9, ls=(0, (3, 3)), zorder=1)
    for i, split in enumerate(res["splits"]):
        pts = [c for c in cands if c["id"].startswith(split + ":")]
        if not pts:
            continue
        ax.scatter([p["forward"] for p in pts], [p["reverse"] for p in pts],
                   s=22, color=color_for(split, i), label=split,
                   edgecolor="#fcfcfb", linewidth=0.8, alpha=0.9, zorder=3)
    ax.set_xlabel("forward containment")
    ax.set_ylabel("reverse containment")
    ax.set_title("Same protein, or a fragment of a longer one?",
                 loc="left", fontsize=10, pad=10)
    ax.annotate("same protein", (0.985, 0.95), ha="right", fontsize=7.5,
                color=INK2)
    ax.annotate("test is a sub-construct", (0.985, 0.08), ha="right",
                fontsize=7.5, color=INK2)
    ax.set_xlim(res["call"] - 0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, labelcolor=INK2, loc="center left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _finish(fig, out, "fig4_fwd_rev")
    plt.close(fig)


def write_tables(res, out):
    with open(out / "ladder.csv", "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["split", "n_scored"] + [f">={t:.2f}" for t in LADDER]
                   + [f">={t:.2f}_pct" for t in LADDER])
        for split, r in res["splits"].items():
            n = r["n_scored"]
            w.writerow([split, n]
                       + [int(round(r["ladder"][f'{t:.2f}'] * n))
                          for t in LADDER]
                       + [f"{100 * r['ladder'][f'{t:.2f}']:.2f}"
                          for t in LADDER])
    log(f"    {out / 'ladder.csv'}")
    cands = res.get("candidates") or []
    if cands:
        with open(out / "candidates.csv", "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(cands[0].keys()))
            w.writeheader()
            w.writerows(cands)
        log(f"    {out / 'candidates.csv'}")
    if res.get("exposure"):
        with open(out / "exposure.csv", "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["split", "n_test", "matched_in_uniprot", "match_rate",
                        "BioT_ge095", "esm1b_n", "esm1b_pct",
                        "esm1v_n", "esm1v_pct"])
            for split, e in res["exposure"].items():
                lad = res["splits"][split]["ladder"]["0.95"]
                w.writerow([split, e["n_test"], e["matched"],
                            f"{e['match_rate']:.4f}",
                            f"{100 * lad:.2f}",
                            e["esm1b"]["n"], f"{100 * e['esm1b']['of_test']:.2f}",
                            e["esm1v"]["n"], f"{100 * e['esm1v']['of_test']:.2f}"])
        log(f"    {out / 'exposure.csv'}")


def make_plots(res, out):
    rule()
    log("  [8]  figures and tables")
    rule()
    write_tables(res, out)
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        log("    matplotlib not installed -- CSVs written, figures skipped.")
        return
    fig_ladder(res, out)
    fig_null(res, out)
    fig_exposure(res, out)
    fig_fwd_rev(res, out)


# ===================================================================== main

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-dir", default="../kstruct_pt_preprocessed/train")
    p.add_argument("--val-dir", default="../kstruct_pt_preprocessed/val")
    p.add_argument("--flip", default="flip_data/meltome")
    p.add_argument("--out", default="overlap_audit")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--call", type=float, default=0.80,
                   help="containment threshold for candidate overlaps")
    p.add_argument("--alpha", type=float, default=1e-3)
    p.add_argument("--null-reps", type=int, default=5)
    p.add_argument("--min-kmers", type=int, default=30)
    p.add_argument("--species", action="store_true",
                   help="resolve corpus species via RCSB (network)")
    p.add_argument("--uniref", action="store_true",
                   help="compute ESM exposure via UniProt dates (network)")
    p.add_argument("--organisms", nargs="+", default=list(MELTOME_TAXA))
    p.add_argument("--uniprot-scope", choices=sorted(SCOPES),
                   default="reference")
    p.add_argument("--esm1b-snapshot", default=ESM1B_SNAPSHOT,
                   help="UniRef50 release for ESM-1b as YYYY-MM (not verified)")
    p.add_argument("--plots-only", action="store_true",
                   help="re-plot from cached results.json")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "results.json"

    if a.plots_only:
        if not cache.exists():
            raise SystemExit(f"no cached results at {cache}")
        make_plots(json.loads(cache.read_text()), out)
        return 0

    import numpy as np
    ds_cls, enc = load_project()
    rule()
    log("  PRETRAINING EXPOSURE AUDIT")
    rule()

    log("\n[1] BioT pretraining corpus")
    tr, ordinfo = corpus_items(ds_cls, enc, a.train_dir, a.limit)
    va = []
    if Path(a.val_dir).exists():
        log("  held-out split:")
        va, _ = corpus_items(ds_cls, enc, a.val_dir, a.limit)
    allit = tr + va
    nt = sorted(x["n_tok"] for x in allit)
    npad = sum(x["n_pad"] for x in allit)
    log(f"  train {len(tr)}  held-out {len(va)}  tokens/protein median "
        f"{nt[len(nt) // 2]}, max {nt[-1]}  PAD {npad} "
        f"({npad / max(sum(x['n_tok'] + x['n_pad'] for x in allit), 1):.2%})")

    log()
    order_frac = order_report(ordinfo)

    log("\n[2] pretraining train vs held-out")
    H = {frozenset(x["ids"]) for x in tr}
    dup = [x["pdbid"] for x in va if frozenset(x["ids"]) in H]
    log(f"  identical 3-mer token sets: {len(dup)}/{len(va)} "
        f"({len(dup) / max(len(va), 1):.1%})"
        + (f"   e.g. {dup[:5]}" if dup else ""))

    log("\n[3] FLIP corpus")
    flip = flip_sequences(a.flip)

    log("\n[4] 3-mer containment against the best single corpus protein")
    CT = build_corpus(allit)
    res = {"call": a.call, "alpha": a.alpha, "null_reps": a.null_reps,
           "order_frac": order_frac, "n_corpus": len(allit),
           "n_train": len(tr), "n_heldout": len(va),
           "heldout_dups": len(dup), "splits": {}, "candidates": []}

    for split, parts in flip.items():
        rows = parts["test"]
        if not rows:
            continue
        keep = [(i, s) for i, s in rows
                if len(seq_kmer_ids(s, enc)) >= a.min_kmers]
        nlow = len(rows) - len(keep)
        sets = [seq_kmer_ids(s, enc) for _, s in keep]
        sizes = np.array([len(x) for x in sets])
        obs, arg = max_containment(CT, sets)
        nl = shuffled_null(CT, keep, enc, a.null_reps, a.min_kmers)
        log(f"\n  {split}   {len(keep)}/{len(rows)} scored"
            + (f", {nlow} low-complexity excluded" if nlow else ""))
        q = np.quantile(obs, [0.5, 0.9, 0.99, 1.0])
        log(f"      observed  median {q[0]:.3f}  p90 {q[1]:.3f}  "
            f"p99 {q[2]:.3f}  max {q[3]:.3f}")
        rec = {"n_test": len(rows), "n_scored": len(keep),
               "n_low_complexity": nlow,
               "observed_q": dict(zip(("p50", "p90", "p99", "max"),
                                      map(float, q)))}
        if nl is not None:
            nn, nv = nl
            qn = np.quantile(nv, [0.5, 0.9, 0.99, 1.0])
            log(f"      shuffled  median {qn[0]:.3f}  p90 {qn[1]:.3f}  "
                f"p99 {qn[2]:.3f}  max {qn[3]:.3f}   (n={nv.size})")
            thr, hard, brows = calibrate(sizes, obs, nn, nv, a.alpha)
            rec["null_q"] = dict(zip(("p50", "p90", "p99", "max"),
                                     map(float, qn)))
            rec["bins"] = brows
            rec["n_over_alpha"] = int((obs > thr).sum())
            rec["n_over_null_max"] = int((obs > hard).sum())
            xs = np.linspace(0, 1, 201)
            rec["null_ecdf"] = {"x": xs.tolist(),
                                "y": [float((nv <= t).mean()) for t in xs]}
            rec["obs_ecdf"] = {"x": xs.tolist(),
                               "y": [float((obs <= t).mean()) for t in xs]}
        rec["ladder"] = {f"{t:.2f}": float((obs >= t).mean())
                         for t in LADDER}
        log(f"\n      {'containment':>14}{'n':>8}{'of scored':>12}")
        for t in LADDER:
            c = int((obs >= t).sum())
            log(f"      {f'>= {t:.2f}':>14}{c:>8}{c / len(keep):>11.1%}"
                + ("   <-- reported" if abs(t - a.call) < 1e-9 else ""))
        rev = reverse_pass(allit, keep, enc, obs, arg, a.call)
        rec["n_called"] = len(rev)
        same = sum(1 for r in rev if r["kind"] == "same-protein")
        log(f"\n      {len(rev)} at >= {a.call:.0%}: {same} same-protein, "
            f"{len(rev) - same} test-is-fragment")
        for r in rev[:5]:
            log(f"        {r['id']:<18} fwd {r['forward']:.3f}  rev "
                f"{r['reverse']:.3f}  {r['match']}  {r['kind']}")
        res["candidates"].extend(rev)
        res["splits"][split] = rec

    log("\n[6] ESM pretraining exposure")
    if not a.uniref:
        log("  skipped (pass --uniref).")
    else:
        uni = uniprot_table(a.organisms, out / "uniprot.json",
                            scope=a.uniprot_scope)
        if not uni:
            log("  no UniProt entries fetched")
        else:
            snaps = {"esm1v": ESM1V_SNAPSHOT, "esm1b": a.esm1b_snapshot}
            res["esm1b_snapshot"] = a.esm1b_snapshot
            ex = exposure(flip, uni, snaps)
            res["exposure"] = ex
            log(f"  {len(uni)} UniProt entries\n")
            log(f"  {'split':<14}{'matched':>14}{'ESM-1b':>14}{'ESM-1v':>14}")
            for split, e in ex.items():
                log(f"  {split:<14}{e['match_rate']:>13.1%}"
                    f"{e['esm1b']['of_test']:>13.1%}"
                    f"{e['esm1v']['of_test']:>13.1%}")

    log("\n[7] BioT corpus species mixture")
    w = Counter(x["pdbid"].split("_")[0].upper() for x in allit)
    sp = species_mix(w, out, a.species)
    if sp:
        res["species"] = sp

    cache.write_text(json.dumps(res))
    log(f"\n  cached -> {cache}")
    log()
    make_plots(res, out)
    rule()
    log("  Re-plot:  python overlap_audit.py --plots-only --out " + str(out))
    rule()
    return 0


if __name__ == "__main__":
    sys.exit(main())
