#!/usr/bin/env bash
# Sets up tests/idr_bench/ with the CAID assessment tool and reference data.
# Run from the repo root:  bash tests/idr_bench/setup_caid.sh
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${DIR}"

echo "Setting up CAID benchmark in ${DIR}"

# Clone the assessment tool
if [ -d "CAID" ]; then
    echo "CAID/ already exists, pulling latest"
    cd CAID && git pull --ff-only && cd ..
else
    echo "Cloning CAID assessment tool"
    git clone https://github.com/BioComputingUP/CAID.git
fi

# CAID pins numpy>=3.11-only; install deps without version pins
pip install numpy pandas scipy tqdm

# Create working directories
mkdir -p probes predictions results

# Verify reference data exists
REF="CAID/demo-data/references/disorder_pdb.fasta"
if [ -f "${REF}" ]; then
    N=$(grep -c '^>' "${REF}")
    echo "Reference: ${REF} (${N} proteins)"
else
    echo "ERROR: ${REF} not found. The CAID repo may have changed."
    exit 1
fi

echo ""
echo "Done. From tests/idr_bench/, run:"
echo ""
echo "  # Train"
echo "  python ../caid_disorder.py train \\"
echo "      --reference CAID/demo-data/references/disorder_pdb.fasta \\"
echo "      --ckpt ../../checkpoints/BioT_voxel_s0.safetensors \\"
echo "      --out probes/disorder.pkl"
echo ""
echo "  # Predict"
echo "  python ../caid_disorder.py predict \\"
echo "      --fasta CAID/demo-data/references/disorder_pdb.fasta \\"
echo "      --probe probes/disorder.pkl \\"
echo "      --out predictions/BioT.caid"
echo ""
echo "  # Evaluate"
echo "  python CAID/caid.py \\"
echo "      CAID/demo-data/references/disorder_pdb.fasta \\"
echo "      predictions/ -o results/"
