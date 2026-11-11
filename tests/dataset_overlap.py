#!/usr/bin/env python
"""Training-set overlap for CAID benchmark proteins via MMseqs2 clustering.

For each training corpus, concatenates the CAID query proteins with the
corpus, clusters at 80% sequence identity, and reports which CAID proteins
land in a cluster with a training sequence.

Usage::

    python dataset_overlap.py \\
        --caid CAID/demo-data/references/disorder_pdb.fasta \\
        --corpus esm2=training_sets/uniref50.fasta \\
        --corpus alphafold=training_sets/pdb_seqres.fasta \\
        --corpus biot=/path/to/biot_corpus.fasta \\
        --out overlap/

Requires ``mmseqs`` on PATH (https://github.com/soedinglab/MMseqs2).
"""

import argparse
import csv
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


def log(s=""):
    print(s, flush=True)


def check_mmseqs():
    try:
        r = subprocess.run(["mmseqs", "version"], capture_output=True, text=True)
        log(f"  mmseqs2 version: {r.stdout.strip()}")
    except FileNotFoundError:
        sys.exit("mmseqs2 not found. Install: conda install -c bioconda mmseqs2")


def strip_caid_labels(caid_path, out_path):
    """Write a plain FASTA from a CAID reference FASTA (skip label lines)."""
    label_chars = set("01-\n\r ")
    with open(caid_path) as fin, open(out_path, "w") as fout:
        expect_label = False
        for line in fin:
            if line.startswith(">"):
                fout.write(line)
                expect_label = True
            elif expect_label and set(line.strip()) <= set("01-"):
                expect_label = False  # skip label line
            else:
                fout.write(line)
                expect_label = False


def read_ids(fasta_path):
    """Return set of sequence ids from a FASTA."""
    ids = set()
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                ids.add(line[1:].split()[0])
    return ids


def run_overlap(caid_fasta, corpus_path, corpus_name, out_dir, seqid, threads):
    """Concatenate CAID + corpus, cluster, report overlapping CAID proteins."""
    work = out_dir / f"_tmp_{corpus_name}"
    work.mkdir(parents=True, exist_ok=True)

    # Concatenate
    combined = work / "combined.fasta"
    log(f"  concatenating CAID + {corpus_name}...")
    with open(combined, "w") as fout:
        for src in [caid_fasta, corpus_path]:
            with open(src) as fin:
                for line in fin:
                    fout.write(line)

    caid_ids = read_ids(caid_fasta)
    log(f"  {len(caid_ids)} CAID proteins")

    # MMseqs2 cluster
    db = str(work / "db")
    clu = str(work / "clu")
    tsv = str(work / "clu.tsv")

    log(f"  clustering at {seqid:.0%} identity...")
    subprocess.run([
        "mmseqs", "createdb", str(combined), db
    ], check=True, capture_output=True)

    subprocess.run([
        "mmseqs", "cluster", db, clu, str(work / "tmp"),
        "--min-seq-id", str(seqid),
        "-c", "0.8",
        "--cov-mode", "0",
        "--threads", str(threads),
    ], check=True, capture_output=True)

    subprocess.run([
        "mmseqs", "createtsv", db, db, clu, tsv
    ], check=True, capture_output=True)

    # Parse clusters: rep -> [members]
    clusters = defaultdict(set)
    with open(tsv) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                rep, member = parts[0], parts[1]
                clusters[rep].add(member)

    # Find CAID proteins that share a cluster with a training sequence
    overlapping = {}
    for rep, members in clusters.items():
        caid_in = members & caid_ids
        train_in = members - caid_ids
        if caid_in and train_in:
            for cid in caid_in:
                overlapping[cid] = {
                    "cluster_rep": rep,
                    "cluster_size": len(members),
                    "train_members": len(train_in),
                }

    # Write CSV
    csv_path = out_dir / f"{corpus_name}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["caid_id", "cluster_rep", "cluster_size",
                     "train_members", "overlaps"])
        for cid in sorted(caid_ids):
            if cid in overlapping:
                o = overlapping[cid]
                w.writerow([cid, o["cluster_rep"], o["cluster_size"],
                            o["train_members"], True])
            else:
                w.writerow([cid, "", 0, 0, False])
    log(f"  wrote {csv_path}")

    # Cleanup mmseqs temp files
    import shutil
    shutil.rmtree(work, ignore_errors=True)

    return {
        "name": corpus_name,
        "n_caid": len(caid_ids),
        "n_overlap": len(overlapping),
        "pct": len(overlapping) / max(len(caid_ids), 1),
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--caid", required=True,
                   help="CAID reference FASTA")
    p.add_argument("--corpus", action="append", required=True,
                   metavar="NAME=PATH",
                   help="training-set FASTA, e.g. esm2=training_sets/uniref50.fasta")
    p.add_argument("--seqid", type=float, default=0.80,
                   help="MMseqs2 clustering identity threshold (default 0.80)")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", default="overlap/")
    a = p.parse_args()

    check_mmseqs()

    # Parse corpus args
    corpora = []
    for spec in a.corpus:
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            name, path = Path(spec).stem, spec
        corpora.append((name.strip(), path.strip()))

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Strip CAID labels to plain FASTA
    caid_fasta = out_dir / "_caid_seqs.fasta"
    strip_caid_labels(a.caid, caid_fasta)

    summaries = []
    for name, path in corpora:
        log()
        log(f"{'=' * 60}")
        log(f"CORPUS: {name}")
        log(f"{'=' * 60}")
        s = run_overlap(caid_fasta, path, name, out_dir, a.seqid, a.threads)
        summaries.append(s)

    # Cleanup
    caid_fasta.unlink(missing_ok=True)

    # Summary
    log()
    log(f"{'=' * 60}")
    log(f"OVERLAP SUMMARY (clustering at {a.seqid:.0%} identity)")
    log(f"{'=' * 60}")
    log(f"  {'corpus':<20} {'overlap':>10} {'total':>8} {'rate':>8}")
    log(f"  {'-' * 20} {'-' * 10} {'-' * 8} {'-' * 8}")
    for s in summaries:
        log(f"  {s['name']:<20} {s['n_overlap']:>10} "
            f"{s['n_caid']:>8} {s['pct']:>7.1%}")
    log(f"{'=' * 60}")


if __name__ == "__main__":
    sys.exit(main() or 0)
