# IDR Bench — CAID Intrinsic Disorder Benchmark

Evaluates BioT and ESM2 per-residue embeddings on the [CAID](https://caid.idpcentral.org/challenge) intrinsic disorder prediction task, with training-set overlap analysis.

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

### 1. Train disorder probes

**BioT:**

```bash
python ../caid_disorder.py train \
    --reference CAID/demo-data/references/disorder_pdb.fasta \
    --ckpt ../../checkpoints/BioT_voxel_s0.safetensors \
    --out probes/disorder_biot.pkl
```

**ESM2** (requires `pip install fair-esm`):

```bash
python ../caid_disorder.py train \
    --esm esm2_t33_650M_UR50D \
    --reference CAID/demo-data/references/disorder_pdb.fasta \
    --out probes/disorder_esm2.pkl
```

Both fit a per-residue logistic regression on DisProt Disorder-PDB labels.

### 2. Predict

```bash
python ../caid_disorder.py predict \
    --fasta CAID/demo-data/references/disorder_pdb.fasta \
    --ckpt ../../checkpoints/BioT_voxel_s0.safetensors \
    --probe probes/disorder_biot.pkl \
    --out predictions/BioT.caid

python ../caid_disorder.py predict \
    --esm esm2_t33_650M_UR50D \
    --fasta CAID/demo-data/references/disorder_pdb.fasta \
    --probe probes/disorder_esm2.pkl \
    --out predictions/ESM2-650M.caid
```

### 3. Evaluate

```bash
cp CAID/demo-data/predictions/*.caid predictions/

python CAID/caid.py \
    CAID/demo-data/references/disorder_pdb.fasta \
    predictions/ -o results/
```

Copies the CAID3 demo predictors (AIUPred, AlphaFold-rsa, LIPNet) alongside BioT and ESM2 so the assessment produces a side-by-side comparison.

### 4.1 View results (if preferred)

```bash
cat results/disorder_pdb.analysis.all.dataset.default.metrics.csv
```

### 4.2 Plot (if preferred)

```bash
python ../plot_caid.py metrics \
    --csv results/disorder_pdb.analysis.all.dataset.default.metrics.csv \
    --out figures/caid_metrics.png

python ../plot_caid.py pca \
    --reference CAID/demo-data/references/disorder_pdb.fasta \
    --predictions predictions/ \
    --ckpt ../../checkpoints/BioT_voxel_s0.safetensors \
    --out figures/pca_proteins.png
```

### 5. Training-set overlap

Download training FASTAs for the compared models:

```bash
bash fetch_training_sets.sh
```

This fetches UniRef50 (ESM2's training corpus) and PDB sequences (AlphaFold/AIUPred). Then check overlap against the CAID proteins using `fasta_overlap.py`:

```bash
# BioT (use your own corpus)
python ../../tests/fasta_overlap.py \
    --fasta CAID/demo-data/references/disorder_pdb.fasta \
    --corpus /path/to/biot/corpus.fasta \
    --out overlap_biot.csv

# ESM2
python ../../tests/fasta_overlap.py \
    --fasta CAID/demo-data/references/disorder_pdb.fasta \
    --corpus training_sets/uniref50_2022_01.fasta \
    --out overlap_esm2.csv

# AlphaFold / AIUPred
python ../../tests/fasta_overlap.py \
    --fasta CAID/demo-data/references/disorder_pdb.fasta \
    --corpus training_sets/pdb_seqres.fasta \
    --out overlap_alphafold.csv
```

### Optional: cross-validate

```bash
python ../caid_disorder.py cv \
    --reference CAID/demo-data/references/disorder_pdb.fasta \
    --ckpt ../../checkpoints/BioT_voxel_s0.safetensors \
    --folds 5

python ../caid_disorder.py cv \
    --esm esm2_t33_650M_UR50D \
    --reference CAID/demo-data/references/disorder_pdb.fasta \
    --folds 5
```

Reports protein-level GroupKFold AUC, F1, and MCC.

## Directory layout

```
tests/idr_bench/
  setup_caid.sh            setup script (run once from repo root)
  fetch_training_sets.sh   download training FASTAs for overlap analysis
  README.md                this file
  CAID/                    cloned assessment tool + demo data
  probes/                  trained probes (.pkl)
  predictions/             .caid prediction files
  results/                 CAID assessment output (CSVs)
  training_sets/           downloaded training FASTAs
  figures/                 plots
    caid_metrics.png
    pca_proteins.png
```
