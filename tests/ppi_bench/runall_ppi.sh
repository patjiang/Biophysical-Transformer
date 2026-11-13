#!/usr/bin/env bash
# Run all model variants on the PLM-Interact PPI benchmark.
# Execute from tests/ppi_bench/
set -e

VOXEL="../../checkpoints/BioT_voxel_s0.safetensors"
NOVOX="../../checkpoints/BioT_novox_s0.safetensors"

echo "=== BioT (voxel) ==="
python ppi_bench.py eval --task all --ckpt "$VOXEL" \
    --out results_voxel/

echo ""
echo "=== BioT (novox) ==="
python ppi_bench.py eval --task all --ckpt "$NOVOX" \
    --out results_novox/

echo ""
echo "=== ESM2-650M ==="
python ppi_bench.py eval --task all --esm esm2_t33_650M_UR50D \
    --out results_esm2/

echo ""
echo "=== Dataset overlap ==="
# Reuse training-set FASTAs from idr_bench (run fetch_training_sets.sh there first)
IDR="../idr_bench"
if [ -f "${IDR}/training_sets/uniref50.fasta" ]; then
    python dataset_overlap_ppi.py \
        --data data/ \
        --corpus esm2="${IDR}/training_sets/uniref50.fasta" \
        --corpus biot="${IDR}/training_sets/biot_corpus.fasta" \
        --out overlap/
else
    echo "  Skipping overlap — run idr_bench/fetch_training_sets.sh first"
fi

echo ""
echo "=== Plot (with paired bootstrap) ==="
python plot_ppi.py \
    --biot results_voxel/BioT_voxel_results.csv \
    --biot-novox results_novox/BioT_novox_results.csv \
    --esm results_esm2/esm2_t33_650M_UR50D_results.csv \
    --preds-biot results_voxel/ \
    --preds-novox results_novox/ \
    --preds-esm results_esm2/ \
    --n-boot 2000 \
    --out figures/ppi_bench.png

echo "Done."
