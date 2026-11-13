# PPI Bench — PLM-Interact Protein–Protein Interaction Benchmark

Evaluates BioT and ESM-2 embeddings on the [PLM-Interact](https://www.nature.com/articles/s41467-025-64512-w) PPI prediction tasks using a cross-attention probe over frozen per-residue embeddings.

The probe architecture is minimal by design: residues in protein A attend to residues in protein B (and vice versa), pooled representations are concatenated, and a small MLP classifies the pair. No backbone fine-tuning, no PPI-specific features. This tests what the embedding space already knows about interactions.

## Setup

From the **repo root**:

```bash
bash tests/ppi_bench/setup_ppi.sh
```

This downloads the benchmark datasets from [HuggingFace](https://huggingface.co/danliu1226) into `tests/ppi_bench/data/`.

For the dataset overlap analysis, also run (once):

```bash
bash tests/idr_bench/fetch_training_sets.sh
```

## Run

All commands below assume you are in `tests/ppi_bench/`.

```bash
cd tests/ppi_bench
```

### Full pipeline (all models + overlap + plots)

```bash
bash run_all_ppi.sh
```

This runs BioT +field, BioT −field, and ESM-2 on all tasks, computes pretraining-set overlap via MMseqs2, generates the performance plot, and prints paired bootstrap significance tests.

### Individual models

```bash
# BioT +field
python ppi_bench.py eval --task all \
    --ckpt ../../checkpoints/BioT_voxel_s0.safetensors \
    --out results_voxel/

# BioT −field (ablation control)
python ppi_bench.py eval --task all \
    --ckpt ../../checkpoints/BioT_novox_s0.safetensors \
    --out results_novox/

# ESM-2 (650M)
python ppi_bench.py eval --task all \
    --esm esm2_t33_650M_UR50D \
    --out results_esm2/
```

### Single task

```bash
python ppi_bench.py eval --task cross_species \
    --ckpt ../../checkpoints/BioT_voxel_s0.safetensors \
    --out results_voxel/
```

### Re-running without retraining

Probes and embedding caches are saved automatically. On subsequent runs, cached embeddings are loaded and existing probes are reused (eval only). To force retraining:

```bash
python ppi_bench.py eval --task all --retrain \
    --ckpt ../../checkpoints/BioT_voxel_s0.safetensors \
    --out results_voxel/
```

### Dataset overlap

Requires MMseqs2 (`conda install -c bioconda mmseqs2`) and the training-set FASTAs from `idr_bench/fetch_training_sets.sh`:

```bash
python dataset_overlap_ppi.py \
    --data data/ \
    --corpus esm2=../idr_bench/training_sets/uniref50.fasta \
    --corpus biot=../idr_bench/training_sets/biot_corpus.fasta \
    --out overlap/
```

Results are cached — re-runs skip completed corpora.

### Plotting

Performance bars with SOTA comparison:

```bash
python plot_ppi.py \
    --biot results_voxel/BioT_voxel_results.csv \
    --biot-novox results_novox/BioT_novox_results.csv \
    --esm results_esm2/esm2_t33_650M_UR50D_results.csv \
    --out figures/ppi_bench.png
```

Combined figure (CAID + PPI, for the paper):

```bash
python plot_benchmarks.py \
    --caid-csv ../idr_bench/results/disorder_pdb.analysis.all.dataset.default.metrics.csv \
    --ppi-biot results_voxel/BioT_voxel_results.csv \
    --ppi-novox results_novox/BioT_novox_results.csv \
    --ppi-esm results_esm2/esm2_t33_650M_UR50D_results.csv \
    --out figures/fig_benchmarks.png
```

## Tasks

| Task | Train | Test | Pos:Neg | Source |
|------|-------|------|---------|--------|
| cross_species | Human PPIs | Mouse, fly, worm, yeast, E. coli | 1:10 | STRING v11 |
| bernett | Human PPIs (leakage-free) | Human PPIs | 1:1 | HIPPIE v2.3 |
| virus_human | Virus–human PPIs | Virus–human PPIs | 1:10 | HPIDB 3.0 |

## Metrics

- **AUPRC** — primary metric, matches the PLM-Interact paper
- **AUROC** — threshold-free discrimination

MCC, F1, precision, and recall are omitted: they depend on a binarisation threshold and are redundant with the threshold-free metrics above.

## Pretraining-set overlap

At 80% sequence identity (MMseqs2 clustering), ESM-2's pretraining corpus (UniRef50) overlaps with 82–99% of the benchmark proteins depending on the task. BioT overlaps with 3–4%. This context is essential for interpreting the performance gap.

| Task | ESM-2 overlap | BioT overlap |
|------|--------------|-------------|
| cross_species | 82% | 3% |
| bernett | 99% | 4% |
| virus_human | 96% | 4% |

## Directory layout

```
tests/ppi_bench/
  ppi_bench.py              cross-attention probe (train + eval)
  plot_ppi.py               per-task performance bars
  plot_benchmarks.py         combined CAID + PPI figure
  dataset_overlap_ppi.py    MMseqs2 overlap analysis
  setup_ppi.sh              dataset download (run from repo root)
  run_all_ppi.sh            full pipeline
  README.md                 this file
  data/                     downloaded benchmark CSVs
    cross_species/           train.csv, validation.csv, test.csv
    bernett/                 train.csv, validation.csv, test.csv
    virus_human/             train.csv, test.csv
  probes/                   trained probes + embedding caches
  results_voxel/            BioT +field results + predictions
  results_novox/            BioT −field results + predictions
  results_esm2/             ESM-2 results + predictions
  overlap/                  MMseqs2 overlap CSVs
  figures/                  generated plots
```
