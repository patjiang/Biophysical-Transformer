#!/usr/bin/env bash
# Download training-set FASTAs for overlap analysis against CAID.
# Run from tests/idr_bench/
set -e

mkdir -p training_sets

# ── ESM2: trained on UniRef50 (2022_01) ──
# ~47M sequences, ~7GB compressed.  This is the full training corpus.
ESM_URL="https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2022_01/uniref/uniref2022_01/uniref50.fasta.gz"
ESM_OUT="training_sets/uniref50_2022_01.fasta"
if [ -f "${ESM_OUT}" ]; then
    echo "ESM2 training set already downloaded"
else
    echo "Downloading ESM2 training set (UniRef50 2022_01, ~7GB)..."
    echo "  ${ESM_URL}"
    wget -q --show-progress -O "${ESM_OUT}.gz" "${ESM_URL}" \
        || curl -L -o "${ESM_OUT}.gz" "${ESM_URL}"
    echo "Decompressing..."
    gunzip "${ESM_OUT}.gz"
    echo "  -> ${ESM_OUT} ($(grep -c '^>' "${ESM_OUT}") sequences)"
fi

# ── AlphaFold2: trained on PDB + UniRef30 + BFD ──
# PDB sequences are the identifiable part; UniRef30/BFD are huge.
# We fetch PDB sequences from RCSB (~200k chains).
AF_OUT="training_sets/pdb_seqres.fasta"
if [ -f "${AF_OUT}" ]; then
    echo "AlphaFold PDB sequences already downloaded"
else
    echo "Downloading PDB sequences (RCSB pdb_seqres.txt, ~90MB)..."
    wget -q --show-progress -O "${AF_OUT}.gz" \
        "https://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz" \
        || curl -L -o "${AF_OUT}.gz" \
        "https://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz"
    gunzip "${AF_OUT}.gz"
    # Filter to protein chains only (remove DNA/RNA)
    python3 -c "
import sys
keep, buf = False, []
with open('${AF_OUT}') as f:
    for line in f:
        if line.startswith('>'):
            if keep and buf:
                sys.stdout.write(header)
                sys.stdout.write(''.join(buf))
            header = line
            keep = 'mol:protein' in line.lower() or 'mol:na' not in line.lower()
            buf = []
        else:
            buf.append(line)
    if keep and buf:
        sys.stdout.write(header)
        sys.stdout.write(''.join(buf))
" > "${AF_OUT}.tmp" && mv "${AF_OUT}.tmp" "${AF_OUT}"
    echo "  -> ${AF_OUT} ($(grep -c '^>' "${AF_OUT}") protein chains)"
fi

# ── IUPred / AIUPred: trained on PDB missing electron density ──
# Same PDB source as AlphaFold; the relevant overlap is PDB structures.
echo "AIUPred uses PDB structures -> same file as AlphaFold: ${AF_OUT}"

echo ""
echo "============================================================"
echo "Training sets in training_sets/:"
ls -lh training_sets/*.fasta
echo ""
echo "Run overlap checks with fasta_overlap.py:"
echo ""
echo "  # BioT overlap (use your corpus)"
echo "  python ../../tests/fasta_overlap.py \\"
echo "      --fasta CAID/demo-data/references/disorder_pdb.fasta \\"
echo "      --corpus /path/to/biot/corpus.fasta \\"
echo "      --out overlap_biot.csv"
echo ""
echo "  # ESM2 overlap"
echo "  python ../../tests/fasta_overlap.py \\"
echo "      --fasta CAID/demo-data/references/disorder_pdb.fasta \\"
echo "      --corpus training_sets/uniref50_2022_01.fasta \\"
echo "      --out overlap_esm2.csv"
echo ""
echo "  # AlphaFold / AIUPred overlap"
echo "  python ../../tests/fasta_overlap.py \\"
echo "      --fasta CAID/demo-data/references/disorder_pdb.fasta \\"
echo "      --corpus training_sets/pdb_seqres.fasta \\"
echo "      --out overlap_alphafold.csv"
