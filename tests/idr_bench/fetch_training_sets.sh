#!/usr/bin/env bash
# Download training-set FASTAs for overlap analysis against CAID.
# Run from tests/idr_bench/
set -e

mkdir -p training_sets
conda install -c bioconda mmseqs2 -y
# ── ESM2: trained on UniRef50 ──
# Current release (~47M sequences, ~7GB compressed).
# ESM2 used the 2022_01 snapshot but the current release is
# a superset and works for overlap checking.
ESM_OUT="training_sets/uniref50.fasta"
if [ -f "${ESM_OUT}" ]; then
    echo "ESM2 training set already downloaded"
else
    echo "Downloading ESM2 training set (UniRef50, ~7GB compressed)..."
    curl -L --progress-bar -o "${ESM_OUT}.gz" \
        "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"
    # Verify it's actually gzip
    if file "${ESM_OUT}.gz" | grep -q gzip; then
        echo "Decompressing..."
        gunzip "${ESM_OUT}.gz"
    else
        echo "ERROR: download is not gzip — check the URL or try manually:"
        echo "  https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"
        rm -f "${ESM_OUT}.gz"
        exit 1
    fi
    echo "  -> ${ESM_OUT} ($(grep -c '^>' "${ESM_OUT}") sequences)"
fi

# ── AlphaFold2 / AIUPred: trained on PDB structures ──
# PDB sequences from RCSB (~200k protein chains).
AF_OUT="training_sets/pdb_seqres.fasta"
if [ -f "${AF_OUT}" ]; then
    echo "AlphaFold PDB sequences already downloaded"
else
    echo "Downloading PDB sequences (~90MB)..."
    curl -L --progress-bar -o "${AF_OUT}.gz" \
        "https://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz"
    if file "${AF_OUT}.gz" | grep -q gzip; then
        gunzip "${AF_OUT}.gz"
    else
        echo "ERROR: download is not gzip"
        rm -f "${AF_OUT}.gz"
        exit 1
    fi
    # Keep protein chains only
    python3 -c "
import sys
keep, header, buf = False, '', []
with open('${AF_OUT}') as f:
    for line in f:
        if line.startswith('>'):
            if keep and buf:
                sys.stdout.write(header + ''.join(buf))
            header, buf = line, []
            keep = 'mol:protein' in line.lower()
        else:
            buf.append(line)
    if keep and buf:
        sys.stdout.write(header + ''.join(buf))
" > "${AF_OUT}.tmp" && mv "${AF_OUT}.tmp" "${AF_OUT}"
    echo "  -> ${AF_OUT} ($(grep -c '^>' "${AF_OUT}") protein chains)"
fi

echo ""
echo "============================================================"
echo "Training sets:"
ls -lh training_sets/*.fasta 2>/dev/null || echo "  (none)"
echo ""
echo "Run overlap with fasta_overlap.py:"
echo ""
echo "  # ESM2"
echo "  python ../../tests/fasta_overlap.py \\"
echo "      --fasta CAID/demo-data/references/disorder_pdb.fasta \\"
echo "      --corpus training_sets/uniref50.fasta \\"
echo "      --out overlap_esm2.csv"
echo ""
echo "  # AlphaFold / AIUPred"
echo "  python ../../tests/fasta_overlap.py \\"
echo "      --fasta CAID/demo-data/references/disorder_pdb.fasta \\"
echo "      --corpus training_sets/pdb_seqres.fasta \\"
echo "      --out overlap_alphafold.csv"
