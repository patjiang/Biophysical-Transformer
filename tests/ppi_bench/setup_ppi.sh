#!/usr/bin/env bash
# Sets up tests/ppi_bench/ with PLM-Interact benchmark datasets.
# Run from the repo root:  bash tests/ppi_bench/setup_ppi.sh
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${DIR}"

echo "Setting up PLM-Interact PPI benchmark in ${DIR}"

pip install datasets huggingface_hub

mkdir -p data

# Download each benchmark split from HuggingFace
python3 -I -c "
from datasets import load_dataset
import os

tasks = {
    'cross_species': 'danliu1226/cross_species_benchmarking',
    'bernett':       'danliu1226/Bernett_benchmarking',
    'virus_human':   'danliu1226/virus_human_benchmarking',
    'mutation':      'danliu1226/Mutation_effect_dataset',
}

for name, repo in tasks.items():
    out = f'data/{name}'
    if os.path.isdir(out) and any(f.endswith('.csv') for f in os.listdir(out)):
        print(f'{name}: already downloaded')
        continue
    print(f'downloading {name} from {repo}...')
    os.makedirs(out, exist_ok=True)
    try:
        ds = load_dataset(repo, trust_remote_code=True)
        for split_name, split_ds in ds.items():
            path = f'{out}/{split_name}.csv'
            split_ds.to_csv(path, index=False)
            print(f'  {split_name}: {len(split_ds)} rows -> {path}')
    except Exception as e:
        print(f'  ERROR: {e}')
        print(f'  Download manually from https://huggingface.co/datasets/{repo}')
"

mkdir -p results figures probes

echo ""
echo "Done. Contents:"
find data -name '*.csv' -exec wc -l {} \; | sort -t/ -k2
echo ""
echo "From tests/ppi_bench/, run:"
echo "  python ppi_bench.py eval --task cross_species --ckpt ../../checkpoints/BioT_voxel_s0.safetensors"
echo "  bash run_all_ppi.sh   # all models + overlap + plot"
