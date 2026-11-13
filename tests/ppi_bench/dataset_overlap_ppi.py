#!/usr/bin/env python
"""Training-set overlap for PLM-Interact benchmark proteins via MMseqs2.

Extracts unique protein sequences from PLM-Interact test CSVs,
clusters them with each pretraining corpus at 80% sequence identity,
and reports which test proteins co-cluster with a training sequence.

Usage::

    python dataset_overlap_ppi.py \\
        --data data/ \\
        --corpus esm2=../idr_bench/training_sets/uniref50.fasta \\
        --corpus biot=/path/to/biot_corpus.fasta \\
        --out overlap/

Requires ``mmseqs`` on PATH (https://github.com/soedinglab/MMseqs2).
"""

import argparse
import csv
import hashlib
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd


def log(s=""):
    print(s, flush=True)


def check_mmseqs():
    try:
        r = subprocess.run(["mmseqs", "version"], capture_output=True, text=True)
        log(f"  mmseqs2 version: {r.stdout.strip()}")
    except FileNotFoundError:
        sys.exit("mmseqs2 not found. Install: conda install -c bioconda mmseqs2")


# ─────────────────────────────────── extract sequences from PPI CSVs

def _seq_id(seq):
    """Stable short ID for a protein sequence."""
    return "PPI_" + hashlib.md5(seq.encode()).hexdigest()[:12]


def extract_ppi_sequences(data_dir):
    """Extract unique protein sequences from PLM-Interact CSVs.

    Returns
    -------
    seqs : dict  {seq_id: sequence}
    task_membership : dict  {seq_id: set of task names}
    task_counts : dict  {task_name: n_unique_seqs}
    """
    data_dir = Path(data_dir)
    seqs = {}          # seq_id -> sequence
    seq_lookup = {}    # sequence -> seq_id  (dedup)
    task_membership = defaultdict(set)  # seq_id -> {task, ...}
    task_counts = {}

    for task_dir in sorted(data_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        task_seqs = set()

        for csv_file in sorted(task_dir.glob("*.csv")):
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                log(f"  WARNING: could not read {csv_file}: {e}")
                continue

            for col in ["query", "text"]:
                if col not in df.columns:
                    continue
                for seq in df[col].dropna().unique():
                    seq = str(seq).strip().upper()
                    if len(seq) < 10:
                        continue
                    if seq not in seq_lookup:
                        sid = _seq_id(seq)
                        seq_lookup[seq] = sid
                        seqs[sid] = seq
                    sid = seq_lookup[seq]
                    task_membership[sid].add(task_name)
                    task_seqs.add(sid)

        task_counts[task_name] = len(task_seqs)
        log(f"  {task_name}: {len(task_seqs)} unique proteins")

    log(f"  Total unique proteins across all tasks: {len(seqs)}")
    return seqs, dict(task_membership), task_counts


def write_fasta(seqs, out_path):
    """Write {id: sequence} as a FASTA file."""
    with open(out_path, "w") as f:
        for sid, seq in sorted(seqs.items()):
            f.write(f">{sid}\n{seq}\n")
    return len(seqs)


# ─────────────────────────────────── MMseqs2 overlap

def run_overlap(query_fasta, query_ids, corpus_path, corpus_name,
                out_dir, seqid, threads):
    """Cluster query + corpus, report which query seqs co-cluster
    with a corpus sequence."""
    csv_path = out_dir / f"{corpus_name}_overlap.csv"

    # Reuse cached result if it exists
    if csv_path.exists():
        log(f"  {csv_path} exists — loading cached result")
        overlapping = set()
        with open(csv_path) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row["overlaps"] == "True":
                    overlapping.add(row["seq_id"])
        log(f"  {len(overlapping)}/{len(query_ids)} overlapping")
        return overlapping

    work = out_dir / f"_tmp_{corpus_name}"
    work.mkdir(parents=True, exist_ok=True)

    # Concatenate
    combined = work / "combined.fasta"
    log(f"  concatenating PPI seqs + {corpus_name}...")
    with open(combined, "w") as fout:
        for src in [query_fasta, corpus_path]:
            with open(src) as fin:
                for line in fin:
                    fout.write(line)

    # MMseqs2 cluster
    db = str(work / "db")
    clu = str(work / "clu")
    tsv = str(work / "clu.tsv")

    log(f"  creating mmseqs DB...")
    subprocess.run(["mmseqs", "createdb", str(combined), db],
                   check=True)

    log(f"  clustering at {seqid:.0%} identity...")
    subprocess.run([
        "mmseqs", "cluster", db, clu, str(work / "tmp"),
        "--min-seq-id", str(seqid),
        "-c", "0.8",
        "--cov-mode", "0",
        "--threads", str(threads),
    ], check=True)

    log(f"  extracting TSV...")
    subprocess.run(["mmseqs", "createtsv", db, db, clu, tsv],
                   check=True)

    # Parse clusters
    clusters = defaultdict(set)
    with open(tsv) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                clusters[parts[0]].add(parts[1])

    # Find PPI proteins that share a cluster with a training sequence
    overlapping = set()
    for rep, members in clusters.items():
        ppi_in = members & query_ids
        train_in = members - query_ids
        if ppi_in and train_in:
            overlapping |= ppi_in

    # Write per-protein CSV
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seq_id", "overlaps"])
        for sid in sorted(query_ids):
            w.writerow([sid, sid in overlapping])
    log(f"  wrote {csv_path}")

    # Cleanup
    import shutil
    shutil.rmtree(work, ignore_errors=True)

    return overlapping


# ─────────────────────────────────── main

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True,
                   help="PLM-Interact data/ directory with task subdirs")
    p.add_argument("--corpus", action="append", required=True,
                   metavar="NAME=PATH",
                   help="pretraining FASTA, e.g. esm2=path/to/uniref50.fasta")
    p.add_argument("--seqid", type=float, default=0.80,
                   help="clustering identity threshold (default 0.80)")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", default="overlap/")
    args = p.parse_args()

    check_mmseqs()

    # Parse corpus specs
    corpora = []
    for spec in args.corpus:
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            name, path = Path(spec).stem, spec
        corpora.append((name.strip(), path.strip()))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract sequences from PPI CSVs
    log("Extracting PPI benchmark sequences...")
    seqs, task_membership, task_counts = extract_ppi_sequences(args.data)

    if not seqs:
        sys.exit("No sequences found. Check --data directory.")

    # Write query FASTA
    query_fasta = out_dir / "_ppi_seqs.fasta"
    write_fasta(seqs, query_fasta)
    query_ids = set(seqs.keys())

    # Run overlap for each corpus
    results = []
    for name, path in corpora:
        log()
        log(f"{'=' * 60}")
        log(f"CORPUS: {name}")
        log(f"{'=' * 60}")
        overlapping = run_overlap(
            query_fasta, query_ids, path, name,
            out_dir, args.seqid, args.threads)

        # Per-task breakdown
        per_task = {}
        for task_name, n_total in sorted(task_counts.items()):
            task_ids = {sid for sid, tasks in task_membership.items()
                        if task_name in tasks}
            n_overlap = len(task_ids & overlapping)
            pct = n_overlap / max(n_total, 1)
            per_task[task_name] = (n_overlap, n_total, pct)

        results.append({
            "name": name,
            "n_total": len(query_ids),
            "n_overlap": len(overlapping),
            "pct": len(overlapping) / max(len(query_ids), 1),
            "per_task": per_task,
        })

    # Cleanup
    query_fasta.unlink(missing_ok=True)

    # Summary
    log()
    log(f"{'=' * 60}")
    log(f"PPI OVERLAP SUMMARY (clustering at {args.seqid:.0%} identity)")
    log(f"{'=' * 60}")

    for r in results:
        log(f"\n  {r['name']}:")
        log(f"    Overall: {r['n_overlap']}/{r['n_total']} "
            f"({r['pct']:.1%})")
        for task, (n_ov, n_tot, pct) in sorted(r["per_task"].items()):
            log(f"    {task}: {n_ov}/{n_tot} ({pct:.1%})")

    # Write summary CSV
    summary_path = out_dir / "ppi_overlap_summary.csv"
    with open(summary_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["corpus", "task", "n_overlap", "n_total", "pct"])
        for r in results:
            w.writerow([r["name"], "all", r["n_overlap"],
                        r["n_total"], f"{r['pct']:.4f}"])
            for task, (n_ov, n_tot, pct) in sorted(r["per_task"].items()):
                w.writerow([r["name"], task, n_ov, n_tot, f"{pct:.4f}"])
    log(f"\n  Summary -> {summary_path}")
    log(f"{'=' * 60}")


if __name__ == "__main__":
    sys.exit(main() or 0)
