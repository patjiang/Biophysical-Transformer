# IDR Bench — CAID Intrinsic Disorder Benchmark

Evaluates BioT and ESM-2 per-residue embeddings on the [CAID](https://caid.idpcentral.org/challenge) intrinsic disorder prediction task, with training-set overlap analysis and paired bootstrap significance testing.

A logistic regression probe is trained on per-residue embeddings (5-fold GroupKFold by protein), with binary predictions thresholded at 0.5 on the probe's output probability (not optimised on the test set).

## Setup

From the **repo root**:

```bash
bash tests/idr_bench/setup_caid.sh
```

This clones the [CAID assessment tool](https://github.com/BioComputingUP/CAID), installs its dependencies, and creates the working directories.

For the training-set overlap analysis, also download the pretraining corpora:

```bash
cd tests/idr_bench
bash fetch_training_sets.sh
```

This fetches UniRef50 (ESM-2's pretraining corpus) and PDB sequences (AlphaFold). Large downloads (~7 GB compressed for UniRef50).

## Run

All commands below assume you are in `tests/idr_bench/`.

```bash
cd tests/idr_bench
```

### Full pipeline (all models + evaluation + plots)

```bash
bash run_all_caid.sh
```

This trains and evaluates BioT +field, BioT −field, and ESM-2 probes, runs the CAID assessment against published methods (AIUPred, AlphaFold-rsa, LIPNet), generates the metrics plot with paired bootstrap CIs, and prints significance tests to stdout.

### Individual models

```bash
# BioT +field
python ../caid_disorder.py train \
    --reference CAID/demo-data/references/disorder_pdb.fasta \
    --ckpt ../../checkpoints/BioT_voxel_s0.safetensors \
    --out probes/disorder_BioT_voxel.pkl

python ../caid_disorder.py predict \
    --fasta CAID/demo-data/references/disorder_pdb.fasta \
    --ckpt ../../checkpoints/BioT_voxel_s0.safetensors \
    --probe probes/disorder_BioT_voxel.pkl \
    --out predictions/BioT_voxel.caid

# BioT −field (ablation control)
python ../caid_disorder.py train \
    --reference CAID/demo-data/references/disorder_pdb.fasta \
    --ckpt ../../checkpoints/BioT_novox_s0.safetensors \
    --out probes/disorder_BioT_novox.pkl

python ../caid_disorder.py predict \
    --fasta CAID/demo-data/references/disorder_pdb.fasta \
    --ckpt ../../checkpoints/BioT_novox_s0.safetensors \
    --probe probes/disorder_BioT_novox.pkl \
    --out predictions/BioT_novox.caid

# ESM-2 (650M) — requires: pip install fair-esm
python ../caid_disorder.py train \
    --esm esm2_t33_650M_UR50D \
    --reference CAID/demo-data/references/disorder_pdb.fasta \
    --out probes/disorder_ESM2_650M.pkl

python ../caid_disorder.py predict \
    --esm esm2_t33_650M_UR50D \
    --fasta CAID/demo-data/references/disorder_pdb.fasta \
    --probe probes/disorder_ESM2_650M.pkl \
    --out predictions/ESM2-650M.caid
```

### Evaluate with CAID assessment

```bash
# Copy published method predictions alongside ours
cp CAID/demo-data/predictions/*.caid predictions/

# Run CAID assessment
python CAID/caid.py \
    CAID/demo-data/references/disorder_pdb.fasta \
    predictions/ -o results/

cat results/disorder_pdb.analysis.all.dataset.default.metrics.csv
```

### Plotting

Metrics figure with paired bootstrap CIs:

```bash
python plot_caid.py metrics \
    --csv results/disorder_pdb.analysis.all.dataset.default.metrics.csv \
    --predictions predictions/ \
    --reference CAID/demo-data/references/disorder_pdb.fasta \
    --n-boot 2000 \
    --out figures/caid_metrics.png
```

Paired-difference bootstrap histograms (appendix):

```bash
python plot_bootstrap_delta.py \
    --reference CAID/demo-data/references/disorder_pdb.fasta \
    --pred-field predictions/BioT_voxel.caid \
    --pred-nofield predictions/BioT_novox.caid \
    --n-boot 2000 \
    --out figures/bootstrap_delta.png
```

UMAP of per-protein embeddings (appendix):

```bash
python plot_caid.py pca \
    --reference CAID/demo-data/references/disorder_pdb.fasta \
    --ckpt ../../checkpoints/BioT_voxel_s0.safetensors \
    --ckpt-novox ../../checkpoints/BioT_novox_s0.safetensors \
    --out figures/umap_proteins.png
```

Overlap vs. specificity/sensitivity scatter:

```bash
python plot_overlap_vs_error.py \
    --csv results/disorder_pdb.analysis.all.dataset.default.metrics.csv \
    --out figures/overlap_vs_error.png
```

### Dataset overlap

Requires MMseqs2 (`conda install -c bioconda mmseqs2`) and the training-set FASTAs from `fetch_training_sets.sh`:

```bash
python dataset_overlap.py \
    --caid CAID/demo-data/references/disorder_pdb.fasta \
    --corpus esm2=training_sets/uniref50.fasta \
    --corpus alphafold=training_sets/pdb_seqres.fasta \
    --corpus biot=/path/to/biot_corpus.fasta \
    --out overlap/
```

### Cross-validate (optional)

```bash
python ../caid_disorder.py cv \
    --reference CAID/demo-data/references/disorder_pdb.fasta \
    --ckpt ../../checkpoints/BioT_voxel_s0.safetensors \
    --folds 5
```

Reports protein-level GroupKFold AUC-ROC and MCC.

## Metrics

- **AUC-ROC** — threshold-free ranking quality
- **MCC** — best single confusion-matrix summary (threshold fixed at 0.5)

F1, precision, and recall are omitted: they are functions of the same confusion matrix at one threshold and do not constitute independent evidence.

## Pretraining-set overlap

At 80% sequence identity (MMseqs2 clustering):

| Corpus | Overlap with CAID |
|--------|-------------------|
| ESM-2 (UniRef50) | 91% |
| AlphaFold (PDB) | 51% |
| BioT | 5% |

AIUPred is excluded from the overlap analysis: its deep-learning component trains on DisProt (the source of the CAID ground-truth labels), not PDB.

## Ablation significance

A paired protein-level bootstrap (2000 replicates, shared protein resamples) confirms that the +field vs −field differences in AUC-ROC and MCC exclude zero at the 95% confidence level. The bootstrap uses shared resamples so that protein-difficulty variance cancels in the paired difference.

## Directory layout

```
tests/idr_bench/
  run_all_caid.sh            full pipeline
  setup_caid.sh              setup script (run from repo root)
  fetch_training_sets.sh     download pretraining corpora
  plot_caid.py               metrics bars + UMAP
  plot_bootstrap_delta.py    paired Δ histograms (appendix)
  plot_overlap_vs_error.py   overlap vs error scatter
  dataset_overlap.py         MMseqs2 overlap analysis
  README.md                  this file
  CAID/                      cloned assessment tool + demo data
  training_sets/             downloaded pretraining corpora
  probes/                    trained probes (.pkl)
  predictions/               .caid prediction files
  results/                   CAID assessment output
  overlap/                   MMseqs2 overlap CSVs
  figures/                   generated plots
```
