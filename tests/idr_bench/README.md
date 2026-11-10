# IDR Bench — CAID Intrinsic Disorder Benchmark

Evaluates BioT per-residue embeddings on the [CAID](https://caid.idpcentral.org/challenge) intrinsic disorder prediction task.

## Setup

From the **repo root**:

```bash
bash tests/idr_bench/setup_caid.sh
```

This clones the [CAID assessment tool](https://github.com/BioComputingUP/CAID), installs its dependencies, and creates the working directories.

## Run

All commands below assume you are in `tests/idr_bench/`.

```bash
cd tests/idr_bench
```

### 1. Train the disorder probe

```bash
python ../caid_disorder.py train \
    --reference CAID/demo-data/references/disorder_pdb.fasta \
    --ckpt ../../checkpoints/BioT_voxel_s0.safetensors \
    --out probes/disorder.pkl
```

Fits a per-residue logistic regression on DisProt Disorder-PDB labels.

### 2. Predict

```bash
python ../caid_disorder.py predict \
    --fasta CAID/demo-data/references/disorder_pdb.fasta \
    --ckpt ../../checkpoints/BioT_voxel_s0.safetensors \
    --probe probes/disorder.pkl \
    --out predictions/BioT.caid
```

### 3. Evaluate

```bash
cp CAID/demo-data/predictions/*.caid predictions/

python CAID/caid.py \
    CAID/demo-data/references/disorder_pdb.fasta \
    predictions/ -o results/
```

Copies the CAID3 demo predictors (AIUPred, AlphaFold-rsa, LIPNet) alongside BioT so the assessment produces a side-by-side comparison.

### 4. View results

```bash
cat results/disorder_pdb.analysis.all.dataset.default.metrics.csv
```

### 5. Plot

```bash
python ../plot_caid.py metrics \
    --csv results/disorder_pdb.analysis.all.dataset.default.metrics.csv \
    --out figures/caid_metrics.png

python ../plot_caid.py pca \
    --reference CAID/demo-data/references/disorder_pdb.fasta \
    --ckpt ../../checkpoints/BioT_voxel_s0.safetensors \
    --out figures/pca/ --max-proteins 20

python ../plot_caid.py pca-global \
    --reference CAID/demo-data/references/disorder_pdb.fasta \
    --ckpt ../../checkpoints/BioT_voxel_s0.safetensors \
    --out figures/pca_global.png
```

### Optional: cross-validate

```bash
python ../caid_disorder.py cv \
    --reference CAID/demo-data/references/disorder_pdb.fasta \
    --ckpt ../../checkpoints/BioT_voxel_s0.safetensors \
    --folds 5
```

Reports protein-level GroupKFold AUC, F1, and MCC.

## Directory layout

```
tests/idr_bench/
  setup_caid.sh        setup script (run once from repo root)
  README.md            this file
  CAID/                cloned assessment tool + demo data
  probes/              trained probe (.pkl)
  predictions/         .caid prediction files
  results/             CAID assessment output (CSVs)
  figures/             plots
    caid_metrics.png
    pca_global.png
    pca/               per-protein PCA scatterplots
```
