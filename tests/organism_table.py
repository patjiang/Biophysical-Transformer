#!/usr/bin/env python
"""Build per-organism labels for mixed-split test proteins.

Produces a CSV with columns ``(seq_sha1, group, taxon)`` that
``stack_check.py --restrict-to`` uses to run the composition control
within a single organism.

Usage::

    python organism_table.py \\
        --res-dir res_voxel_mixed_split \\
        --emb-dir emb_mixed_split \\
        --uniprot overlap_audit/uniprot.json \\
        --out organisms.csv

Then per group::

    python stack_check.py \\
        --run control res_novox emb_mixed \\
        --run field   res_voxel emb_mixed \\
        --arm attn --restrict-to organisms.csv \\
        --restrict-group "Homo sapiens"

Labels come from inverting ``overlap_audit.py``'s UniProt cache
(sequence -> taxon for the 13 Meltome Atlas organisms).
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

TAXA = {
    "9606": "Homo sapiens", "10090": "Mus musculus",
    "7955": "Danio rerio", "7227": "Drosophila melanogaster",
    "6239": "Caenorhabditis elegans", "3702": "Arabidopsis thaliana",
    "4932": "Saccharomyces cerevisiae", "562": "Escherichia coli",
    "1423": "Bacillus subtilis", "274": "Thermus thermophilus",
    "1332": "Picrophilus torridus", "188908": "Oleispira antarctica",
    "1422": "Geobacillus stearothermophilus",
}


def log(s=""):
    print(s, flush=True)


def rule(c="-"):
    log(c * 74)


def sequence_to_taxon(uniprot_json):
    """Invert the UniProt cache: sequence SHA-1 -> taxon id.

    Returns ``(by_hash, ambiguous)`` where ambiguous hashes (appearing
    under multiple taxa) are excluded from the map.
    """
    store = json.loads(Path(uniprot_json).read_text())
    if not isinstance(store, dict) or "taxa" not in store:
        raise SystemExit(
            f"\n  {uniprot_json} has no `taxa` key. Re-run "
            f"overlap_audit.py section [6] to regenerate it.\n")
    log(f"  cache scope={store.get('scope', '?')}, "
        f"{len(store['taxa'])} taxa")
    by_hash, seen = {}, defaultdict(set)
    for taxon, rec in store["taxa"].items():
        if not rec:
            log(f"    taxon {taxon:<8}{TAXA.get(taxon, '?'):<32}EMPTY")
            continue
        for acc, pair in rec.items():
            seq = pair[0] if isinstance(pair, (list, tuple)) else pair
            if not isinstance(seq, str) or not seq:
                continue
            h = hashlib.sha1(seq.encode()).hexdigest()
            by_hash[h] = taxon
            seen[h].add(taxon)
        log(f"    taxon {taxon:<8}{TAXA.get(taxon, '?'):<32}"
            f"{len(rec):>7} entries")
    ambiguous = {h for h, t in seen.items() if len(t) > 1}
    for h in ambiguous:
        by_hash.pop(h, None)
    return by_hash, ambiguous


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res-dir", required=True,
                   help="results dir for the mixed split")
    p.add_argument("--emb-dir", required=True,
                   help="embedding cache that results dir was built from")
    p.add_argument("--uniprot", default="overlap_audit/uniprot.json")
    p.add_argument("--out", default="organisms.csv")
    p.add_argument("--min-group", type=int, default=200,
                   help="warn below this; stack_check refuses below 50")
    a = p.parse_args()

    from stack_check import test_sequences

    rule("=")
    log("organism_table.py")
    rule("=")

    pred_path = Path(a.res_dir) / "test_predictions.csv"
    if not pred_path.exists():
        raise SystemExit(f"\n  no {pred_path}\n")
    pred = pd.read_csv(pred_path)
    seqs, _ = test_sequences(a.emb_dir, pred, "mixed")
    log(f"  {len(seqs)} mixed-split test sequences")
    log()

    log(f"  reading {a.uniprot}")
    by_hash, ambiguous = sequence_to_taxon(a.uniprot)
    log(f"  {len(by_hash):,} distinct sequences mapped to a taxon"
        + (f"; {len(ambiguous)} dropped as ambiguous" if ambiguous else ""))
    log()

    rows, unmatched = [], 0
    for s in seqs:
        h = hashlib.sha1(s.encode()).hexdigest()
        t = by_hash.get(h)
        if t is None:
            unmatched += 1
            continue
        rows.append({"seq_sha1": h, "group": TAXA.get(t, f"taxon {t}"),
                     "taxon": t})
    df = pd.DataFrame(rows).drop_duplicates(subset="seq_sha1")

    rule()
    log("GROUPS")
    rule()
    if df.empty:
        raise SystemExit(
            "\n  no test sequence matched a cached UniProt entry.\n"
            "  Check that --res-dir/--emb-dir are the mixed split and "
            "uniprot.json is from the right scope.\n")
    counts = Counter(df["group"])
    log(f"  {'organism':<34}{'n':>7}{'share':>9}")
    rule(".")
    for g, n in counts.most_common():
        log(f"  {g:<34}{n:>7}{n / len(seqs):>8.1%}"
            + ("   (too small for an interval)" if n < a.min_group else ""))
    rule(".")
    matched = len(df)
    log(f"  {'matched':<34}{matched:>7}{matched / len(seqs):>8.1%}")
    log(f"  {'unmatched (no group)':<34}{unmatched:>7}"
        f"{unmatched / len(seqs):>8.1%}")

    df.to_csv(a.out, index=False)
    log()
    log(f"  wrote {a.out}  ({len(df)} rows)")

    usable = [g for g, n in counts.items() if n >= a.min_group]
    rule("=")
    log("NEXT")
    rule("=")
    if not usable:
        log(f"  No group reaches {a.min_group} proteins.")
        return
    log(f"  {len(usable)} group(s) at or above {a.min_group}. For each:")
    log()
    for g in sorted(usable, key=lambda x: -counts[x]):
        tag = g.replace(" ", "_")
        r_nov = a.res_dir.replace("voxel", "novox").replace("e75", "e79")
        m_nov = a.emb_dir.replace("voxel", "novox").replace("e75", "e79")
        log(f"  python stack_check.py \\")
        log(f"      --run control {r_nov} {m_nov} \\")
        log(f"      --run field   {a.res_dir} {a.emb_dir} \\")
        log(f"      --arm attn --restrict-to {a.out} \\")
        log(f"      --restrict-group \"{g}\" --split-name mixed_{tag} \\")
        log(f"      --json strat_{tag}.json")
        log()


if __name__ == "__main__":
    sys.exit(main() or 0)
