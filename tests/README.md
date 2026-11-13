# Tests — Benchmark and Evaluation Suite

This directory contains all evaluation scripts for BioT. Each benchmark lives in its own subfolder with a self-contained pipeline (setup → run → plot). The shared embedding infrastructure (`caid_disorder.py`) sits at this level and is imported by both benchmarks.

## Structure

```
tests/
  caid_disorder.py           shared Embedder class (BioT + ESM-2)
  idr_bench/                 CAID intrinsic disorder benchmark
  ppi_bench/                 PLM-Interact protein–protein interaction benchmark
  flip_meltome.py            Meltome thermal stability evaluation
  make_figures.py            paper figure generation (Meltome)
  organism_table.py          organism-level analysis
  overlap_audit.py           overlap sanity checks
  stack_check.py             model architecture verification
  run_rerun.sh               Meltome re-run script
```

## Benchmarks

### Intrinsic Disorder (`idr_bench/`)

Evaluates per-residue disorder prediction on the [CAID](https://caid.idpcentral.org/challenge) benchmark (319 proteins). Compares BioT +field, BioT −field, and ESM-2 against published methods (AIUPred, AlphaFold-rsa, LIPNet) using the official CAID assessment tool. Includes MMseqs2 pretraining-set overlap analysis and paired bootstrap significance testing.

```bash
bash tests/idr_bench/setup_caid.sh    # from repo root, once
cd tests/idr_bench && bash run_all_caid.sh
```

See [`idr_bench/README.md`](idr_bench/README.md) for details.

### Protein–Protein Interaction (`ppi_bench/`)

Evaluates pairwise interaction prediction on the [PLM-Interact](https://www.nature.com/articles/s41467-025-64512-w) benchmark (cross-species, Bernett, virus–human tasks). Uses a cross-attention probe over frozen per-residue embeddings. Includes pretraining-set overlap analysis and paired bootstrap significance testing.

```bash
bash tests/ppi_bench/setup_ppi.sh     # from repo root, once
cd tests/ppi_bench && bash run_all_ppi.sh
```

See [`ppi_bench/README.md`](ppi_bench/README.md) for details.

## Shared: `caid_disorder.py`

The `Embedder` class wraps both BioT and ESM-2 backends behind a uniform `per_residue(sequence)` interface. Both benchmarks import it:

- **BioT backend**: k-mer tokenisation → sliding-window embedding → k-mer-to-residue averaging
- **ESM-2 backend**: `fair-esm` library, per-residue from the last hidden layer

This file also contains the CAID-specific probe training (`train`), prediction (`predict`), and cross-validation (`cv`) subcommands used by `idr_bench/`.

## Quick start

```bash
# 1. Set up both benchmarks (from repo root)
bash tests/idr_bench/setup_caid.sh
bash tests/ppi_bench/setup_ppi.sh
bash tests/idr_bench/fetch_training_sets.sh   # for overlap analysis

# 2. Run CAID
cd tests/idr_bench && bash run_all_caid.sh

# 3. Run PPI
cd ../ppi_bench && bash run_all_ppi.sh

# 4. Combined figure (from ppi_bench/)
python plot_benchmarks.py \
    --caid-csv ../idr_bench/results/disorder_pdb.analysis.all.dataset.default.metrics.csv \
    --ppi-biot results_voxel/BioT_voxel_results.csv \
    --ppi-novox results_novox/BioT_novox_results.csv \
    --ppi-esm results_esm2/esm2_t33_650M_UR50D_results.csv \
    --out figures/fig_benchmarks.png
```

## Dependencies

Beyond the base BioT environment (`setup.sh`):

- `fair-esm` — for ESM-2 embeddings (`pip install fair-esm`)
- `mmseqs2` — for overlap analysis (`conda install -c bioconda mmseqs2`)
- `umap-learn` — for UMAP plots (`pip install umap-learn`)
- `scikit-learn` — probes and bootstrap (`pip install scikit-learn`)
