#!/usr/bin/env python
"""Training-set overlap for CAID benchmark proteins.

Wraps the same 3-mer containment + shuffled null from fasta_overlap.py,
but runs all compared models in one invocation::

    python dataset_overlap.py \\
        --caid CAID/demo-data/references/disorder_pdb.fasta \\
        --corpus esm2=training_sets/uniref50.fasta \\
        --corpus alphafold=training_sets/pdb_seqres.fasta \\
        --corpus biot=/path/to/biot_corpus.fasta \\
        --out overlap/

For each corpus the script writes ``overlap/<name>.csv`` (one row per
CAID protein, columns compatible with ``stack_check.py --exclude``) and
prints a summary table comparing all models.

The containment metric, threshold calibration and output columns are
identical to ``fasta_overlap.py`` so results are directly comparable
to the paper's pretraining-exposure audit.
"""

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

AA20 = set("ACDEFGHIKLMNPQRSTVWY")


def log(s=""):
    print(s, flush=True)


def rule(c="-"):
    log(c * 74)


# ----------------------------------------------------------------- sequences

def read_fasta(path):
    """(id, sequence) pairs from a FASTA file."""
    out, name, buf = [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line[0] == ">":
                if name is not None:
                    out.append((name, "".join(buf)))
                name, buf = line[1:].split()[0] if len(line) > 1 else "?", []
            else:
                buf.append(line.upper())
    if name is not None:
        out.append((name, "".join(buf)))
    return out


def read_caid_fasta(path):
    """Parse CAID reference FASTA (>acc / seq / labels). Returns (id, seq)."""
    label_chars = set("01-")
    entries = []
    with open(path) as f:
        lines = [l.rstrip("\n\r") for l in f if l.strip() and not l.startswith("#")]
    i = 0
    while i < len(lines):
        if lines[i].startswith(">"):
            acc = lines[i][1:].split()[0]
            seq = lines[i + 1]
            # Skip label line if present
            if i + 2 < len(lines) and set(lines[i + 2]) <= label_chars:
                i += 3
            else:
                i += 2
            entries.append((acc, seq.upper()))
        else:
            i += 1
    return entries


def kmers(s):
    """Distinct 3-mers over the 20 standard residues."""
    return {s[i:i + 3] for i in range(len(s) - 2)
            if s[i] in AA20 and s[i + 1] in AA20 and s[i + 2] in AA20}


# ------------------------------------------------------------------- index

class Index:
    """Inverted 3-mer index for fast all-pairs containment."""

    def __init__(self, kmer_sets):
        self.sizes = np.array([len(s) for s in kmer_sets], dtype=np.int64)
        self.n = len(kmer_sets)
        vocab, rows, cols = {}, [], []
        for j, s in enumerate(kmer_sets):
            for w in s:
                k = vocab.get(w)
                if k is None:
                    k = vocab[w] = len(vocab)
                rows.append(k)
                cols.append(j)
        self.vocab = vocab
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        order = np.argsort(rows, kind="stable")
        self.post = cols[order]
        self.offs = np.searchsorted(rows[order], np.arange(len(vocab) + 1))

    def best(self, q):
        """(forward containment, corpus index, shared count)."""
        ks = [self.vocab[w] for w in q if w in self.vocab]
        if not ks:
            return 0.0, -1, 0
        c = np.bincount(np.concatenate(
            [self.post[self.offs[k]:self.offs[k + 1]] for k in ks]),
            minlength=self.n)
        j = int(np.argmax(c))
        return float(c[j]) / max(len(q), 1), j, int(c[j])


# ---------------------------------------------------------- overlap check

def run_overlap(queries, corpus_kmer_sets, corpus_names, call=0.80,
                min_kmers=30, null_reps=5, null_limit=2000, seed=0):
    """Run overlap analysis. Returns (rows, summary_dict)."""
    idx = Index(corpus_kmer_sets)
    log(f"  {idx.n:,} corpus proteins, {len(idx.vocab):,} distinct 3-mers")

    # Score queries
    rows, fwd = [], np.zeros(len(queries))
    for i, (name, seq, kset) in enumerate(queries):
        f, j, inter = idx.best(kset)
        fwd[i] = f
        rev = inter / max(int(idx.sizes[j]), 1) if j >= 0 else 0.0
        rows.append({
            "id": name,
            "seq_sha1": hashlib.sha1(seq.encode()).hexdigest(),
            "len": len(seq), "n_kmers": len(kset),
            "forward": round(f, 4), "reverse": round(rev, 4),
            "match": corpus_names[j] if j >= 0 else "",
            "match_tokens": int(idx.sizes[j]) if j >= 0 else 0,
            "kind": ("same-protein" if rev >= 0.80 else "query-is-fragment"),
        })
        if (i + 1) % 500 == 0:
            log(f"    {i + 1:,}/{len(queries):,}")

    # Shuffled null
    rng = np.random.default_rng(seed)
    sub = queries[:null_limit]
    null = []
    for _, seq, _ in sub:
        arr = np.frombuffer(seq.encode(), dtype="S1").copy()
        for _ in range(null_reps):
            rng.shuffle(arr)
            k = kmers(arr.tobytes().decode())
            if len(k) >= min_kmers:
                null.append(idx.best(k)[0])
    null = np.asarray(null) if null else np.zeros(1)

    n_call = int((fwd >= call).sum())
    summary = {
        "n_query": len(queries),
        "n_flagged": n_call,
        "pct_flagged": n_call / max(len(queries), 1),
        "null_max": float(null.max()),
        "null_valid": null.max() < call,
        "median_fwd": float(np.median(fwd)),
        "p99_fwd": float(np.quantile(fwd, 0.99)),
    }
    return rows, summary


# ------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--caid", required=True,
                   help="CAID reference FASTA (query sequences)")
    p.add_argument("--corpus", action="append", required=True,
                   metavar="NAME=PATH",
                   help="training-set FASTA, e.g. esm2=training_sets/uniref50.fasta")
    p.add_argument("--call", type=float, default=0.80,
                   help="containment threshold for flagging overlap")
    p.add_argument("--min-kmers", type=int, default=30)
    p.add_argument("--null-reps", type=int, default=5)
    p.add_argument("--null-limit", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="overlap/",
                   help="output directory for per-corpus CSVs")
    a = p.parse_args()

    # Parse corpus arguments
    corpora = []
    for spec in a.corpus:
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            name = Path(spec).stem
            path = spec
        corpora.append((name.strip(), path.strip()))

    # Load CAID query sequences
    rule("=")
    log("dataset_overlap.py")
    rule("=")
    log(f"  CAID reference: {a.caid}")
    caid_entries = read_caid_fasta(a.caid)
    queries = []
    short = 0
    for name, seq in caid_entries:
        k = kmers(seq)
        if len(k) < a.min_kmers:
            short += 1
            continue
        queries.append((name, seq, k))
    log(f"  {len(queries):,} query proteins"
        + (f" ({short:,} excluded under --min-kmers {a.min_kmers})"
           if short else ""))
    log()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run each corpus
    all_summaries = []
    cols = ["id", "seq_sha1", "len", "n_kmers", "forward", "reverse",
            "match", "match_tokens", "kind"]

    for corpus_name, corpus_path in corpora:
        rule("=")
        log(f"CORPUS: {corpus_name}  ({corpus_path})")
        rule("=")

        log("  loading corpus...")
        raw = read_fasta(corpus_path)
        corpus_ksets = []
        corpus_names = []
        for n, s in raw:
            k = kmers(s)
            if k:
                corpus_ksets.append(k)
                corpus_names.append(n)
        log(f"  {len(corpus_ksets):,} proteins with valid 3-mers")

        log("  building index and scoring...")
        rows, summary = run_overlap(
            queries, corpus_ksets, corpus_names,
            call=a.call, min_kmers=a.min_kmers,
            null_reps=a.null_reps, null_limit=a.null_limit, seed=a.seed)

        # Write CSV
        csv_path = out_dir / f"{corpus_name}.csv"
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in sorted(rows, key=lambda x: -x["forward"]):
                w.writerow(r)
        log(f"  wrote {csv_path}")

        # Print per-corpus summary
        fwd = np.array([r["forward"] for r in rows])
        log()
        log(f"  {'median':>9}{'99th':>9}{'max':>9}{'flagged':>12}")
        log(f"  {np.median(fwd):>9.3f}{np.quantile(fwd, .99):>9.3f}"
            f"{fwd.max():>9.3f}"
            f"  {summary['n_flagged']:>5}/{len(queries)}"
            f" ({summary['pct_flagged']:.1%})")
        if not summary["null_valid"]:
            log(f"  !! null max {summary['null_max']:.3f} >= --call {a.call} "
                f"-- threshold not calibrated for this corpus")
        log()

        summary["name"] = corpus_name
        all_summaries.append(summary)

    # Final comparison table
    rule("=")
    log("COMPARISON")
    rule("=")
    log(f"  {'corpus':<20} {'flagged':>10} {'rate':>8} "
        f"{'null_max':>10} {'calibrated':>12}")
    rule(".")
    for s in all_summaries:
        cal = "YES" if s["null_valid"] else "NO"
        log(f"  {s['name']:<20} {s['n_flagged']:>10} "
            f"{s['pct_flagged']:>7.1%} "
            f"{s['null_max']:>10.3f} {cal:>12}")
    rule("=")
    return 0


if __name__ == "__main__":
    sys.exit(main())
