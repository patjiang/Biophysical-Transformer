#!/usr/bin/env bash
# Run all model variants on the CAID IDR benchmark.
# Execute from tests/idr_bench/
set -e

REF="CAID/demo-data/references/disorder_pdb.fasta"
VOXEL="../../checkpoints/BioT_voxel_s0.safetensors"
NOVOX="../../checkpoints/BioT_novox_s0.safetensors"

# caid_disorder.py is the shared embedder in tests/
CAID_DISORDER="../caid_disorder.py"

echo "=== BioT (voxel) ==="
python "$CAID_DISORDER" train \
    --reference "$REF" --ckpt "$VOXEL" \
    --out probes/disorder_BioT_voxel.pkl
python "$CAID_DISORDER" predict \
    --fasta "$REF" --ckpt "$VOXEL" \
    --probe probes/disorder_BioT_voxel.pkl \
    --out predictions/BioT_voxel.caid

echo ""
echo "=== BioT (novox) ==="
python "$CAID_DISORDER" train \
    --reference "$REF" --ckpt "$NOVOX" \
    --out probes/disorder_BioT_novox.pkl
python "$CAID_DISORDER" predict \
    --fasta "$REF" --ckpt "$NOVOX" \
    --probe probes/disorder_BioT_novox.pkl \
    --out predictions/BioT_novox.caid

echo ""
echo "=== ESM2-650M ==="
python "$CAID_DISORDER" train \
    --esm esm2_t33_650M_UR50D \
    --reference "$REF" \
    --out probes/disorder_ESM2_650M.pkl
python "$CAID_DISORDER" predict \
    --esm esm2_t33_650M_UR50D \
    --fasta "$REF" \
    --probe probes/disorder_ESM2_650M.pkl \
    --out predictions/ESM2-650M.caid

echo ""
echo "=== Evaluate all ==="
cp CAID/demo-data/predictions/*.caid predictions/
python CAID/caid.py "$REF" predictions/ -o results/

echo ""
echo "=== Plot (with protein-level bootstrap CIs) ==="
python plot_caid.py metrics \
    --csv results/disorder_pdb.analysis.all.dataset.default.metrics.csv \
    --predictions predictions/ \
    --reference "$REF" \
    --n-boot 2000 \
    --out figures/caid_metrics.png

echo "Done."
