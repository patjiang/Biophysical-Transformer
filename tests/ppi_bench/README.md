# PPI Bench — PLM-Interact Protein-Protein Interaction Benchmark

Evaluates BioT and ESM2 embeddings on the [PLM-Interact](https://www.nature.com/articles/s41467-025-64512-w) PPI prediction tasks using mean-pooled per-residue embeddings with a logistic regression probe.

## Setup

From the **repo root**:

```bash
bash tests/ppi_bench/setup_ppi.sh
```

This downloads the benchmark datasets from [HuggingFace](https://huggingface.co/danliu1226) into `data/`.

## Run

All commands below assume you are in `tests/ppi_bench/`.

```bash
cd tests/ppi_bench
```

### Evaluate BioT on all tasks

```bash
python ../ppi_bench.py eval --task all \
    --ckpt ../../checkpoints/BioT_voxel_s0.safetensors
```

### Evaluate ESM2 on all tasks

```bash
python ../ppi_bench.py eval --task all \
    --esm esm2_t33_650M_UR50D
```

### Single task

```bash
python ../ppi_bench.py eval --task cross_species \
    --ckpt ../../checkpoints/BioT_voxel_s0.safetensors

python ../ppi_bench.py eval --task bernett \
    --ckpt ../../checkpoints/BioT_voxel_s0.safetensors

python ../ppi_bench.py eval --task virus_human \
    --ckpt ../../checkpoints/BioT_voxel_s0.safetensors
```

### View results

```bash
cat results/BioT_results.csv
cat results/esm2_t33_650M_UR50D_results.csv
```

## Tasks

| Task | Train | Test | Pos:Neg | Source |
|------|-------|------|---------|--------|
| cross_species | Human PPIs | Mouse, fly, worm, yeast, E. coli | 1:10 | STRING v11 |
| bernett | Human PPIs (leakage-free) | Human PPIs | 1:1 | HIPPIE v2.3 |
| virus_human | Virus-human PPIs | Virus-human PPIs | 1:10 | HPIDB 3.0 |

## Metrics

AUPRC (primary, matches PLM-Interact paper), AUROC, best-F1, MCC, accuracy.

## Directory layout

```
tests/ppi_bench/
  setup_ppi.sh       setup script (run once from repo root)
  README.md          this file
  data/              downloaded benchmark CSVs
    cross_species/   train.csv, test_mouse.csv, test_fly.csv, ...
    bernett/         train.csv, validation.csv, test.csv
    virus_human/     train.csv, validation.csv, test.csv
  probes/            trained classifiers and embedding caches
  results/           output CSVs with per-split metrics
```
