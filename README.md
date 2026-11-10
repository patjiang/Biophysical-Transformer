# SE3PLM

Code for *<PAPER TITLE>* (<VENUE/YEAR>).

SE3PLM learns per-residue representations by reconstructing local electrostatic
environments through a rotationally-invariant spherical-harmonic descriptor,
jointly with masked language modelling, DINO and iBOT. The encoder is a small
RoPE transformer (5 layers, d=64) over stride-1 3-mer tokens. The paper's
central claim is a paired comparison between this model and a matched control
that drops the voxel objective (`--no-voxel`), transferred to FLIP Meltome
thermostability.

---

## Setup

### Automatic (recommended)

```bash
bash setup.sh
conda activate EBPT
pip install -e .
python test.py
```

`setup.sh` creates a conda environment called **EBPT** (Python 3.10), installs
PyTorch with CUDA 12.6, the [se3cnn fork](https://github.com/patjiang/se3cnn-patch)
for equivariant 3D convolutions, and all remaining dependencies. It skips
creation if the environment already exists, and runs a quick SE3 convolution
smoke test at the end.

`pip install -e .` makes `src/` importable so the evaluation scripts in
`tests/` can find the model code.

`test.py` is the full integration check: it builds the encoder, the
interaction block, the voxel decoder, and the Lightning module, then runs a
forward pass and verifies the output shapes. If it passes, you're ready.

### Manual (step by step)

If you'd rather see what each piece does, or need to adapt something for your
system:

**1. Create the environment**

```bash
conda create -n EBPT python=3.10
conda activate EBPT
```

**2. Install PyTorch (CUDA 12.6)**

```bash
pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu126
```

**3. Install the SE3-CNN fork**

> [!NOTE]
> This fork includes a fix for `scipy.linalg.eigh` calls in `SO3.py`.

```bash
git clone https://github.com/patjiang/se3cnn-patch.git
cd se3cnn-patch
python setup.py install
cd ..
```

**4. Install helper libraries**

```bash
pip install appdirs lie_learn scipy torchinfo tensorboard scikit-learn pandas
```

**5. Install training dependencies**

```bash
pip install \
    pytorch-lightning==2.5.2 \
    pot==0.9.5 \
    einops==0.8.1 \
    matplotlib==3.10.3
```

**6. Verify SE3-CNN**

```python
import torch
from se3cnn.image.convolution import SE3Convolution

size = 32
scalar_field = torch.randn(1, 1, size, size, size)
conv = SE3Convolution(Rs_in=[(1, 0)], Rs_out=[(1, 1)], size=5)
vector_field = conv(scalar_field)
print("SE3-CNN OK")
```

**7. Install the package and run the test suite**

```bash
pip install -e .
python test.py
```

---

## What you can reproduce today

**Everything downstream of FLIP Meltome.** FLIP is a public benchmark, and
the released checkpoints are all you need. That covers the main transfer
table, the composition-controlled analysis, the pretraining-exposure audit,
and the restricted re-run -- the paper's headline numbers and the appendix
that stress-tests them.

**The pretraining corpus is not yet public.** The electrostatic volume dataset
(~21k proteins, 33³ potential boxes) will be released separately. Until then,
the field-accuracy ladder, descriptor breakdown, frame-assumption tests and
reconstruction figures are not independently reproducible. We'd rather say so
than ship scripts that can't run.

---

## Repository layout

```
setup.sh               environment setup (conda + dependencies)
test.py                integration smoke test
pyproject.toml         package config for pip install -e .

src/                   model and training code
  model.py               LightningModule (networks, staging, optimisation)
  loss.py                reconstruction and distillation losses
  invariant_loss.py      spherical-harmonic descriptor and invariant terms
  blocks.py              encoder, decoder, collapse guard
  data.py                dataset and collation
  span_masking.py        residue-anchored span masking
  train.py               training entry point and ablation harness
  eval_utils.py          checkpoint loading, tokenisation, embedding

tests/                 evaluation and analysis scripts
  flip_meltome.py        embed → fit probes → postmortem
  stack_check.py         arm deltas, bootstrap CIs, composition control
  overlap_audit.py       pretraining-exposure audit
  organism_table.py      per-organism labels for stratified tests
  make_figures.py        all figures from cached results
  run_rerun.sh           drives the full and overlap-removed re-run
```

---

## Reproducing the Meltome results

You'll need both checkpoints -- the claim is a *difference* between arms, so
the control is not optional.

| | checkpoint | epoch |
|---|---|---|
| field arm | `<ARCHIVE>/se3plm_voxel_s0.ckpt` | 75 |
| control arm | `<ARCHIVE>/se3plm_novox_s0.ckpt` | 79 |

Each arm is evaluated at its own best validation epoch.

### Step 1 — Embed and fit probes

```bash
python tests/flip_meltome.py embed --help
python tests/flip_meltome.py run   --help
```

`embed` turns a checkpoint into a per-split embedding cache. `run` fits the
probe heads and writes `test_predictions.csv`. Run both for each arm and split.

Name the output directories like this (the downstream scripts discover them
by name):

```
res_{novox,voxel}_{human,human_cell,mixed_split}_{e79,e75}
mt_{novox,voxel}_{human,human_cell,mixed_split}_{e79,e75}
```

### Step 2 — Arm deltas and composition control

```bash
python tests/stack_check.py --arm attn \
    --run novox res_novox_human_e79 mt_novox_human_e79 \
    --run voxel res_voxel_human_e75 mt_voxel_human_e75 \
    --n-boot 4000 --split-name human --json human_attn.json
```

This backs the main transfer table and the composition-controlled table.

### Step 3 — Pretraining-exposure audit

```bash
python tests/overlap_audit.py \
    --flip <FLIP_DIR> --train-dir <CORPUS>/train --val-dir <CORPUS>/val \
    --call 0.80 --null-reps 5 --out overlap_audit
```

Writes `candidates.csv` (the overlap call per test protein) and the JSON the
figures use. The `0.80` threshold comes from the shuffled null and is used
everywhere a single number is needed.

> **Note:** this step reads the corpus and can't run until the data is
> released. The `overlap_audit/` directory ships pre-computed so everything
> downstream works without it.

### Step 4 — Restricted re-run (overlap removal)

```bash
bash tests/run_rerun.sh --dry     # check paths, discover arms, run nothing
bash tests/run_rerun.sh           # the real thing
```

Always `--dry` first -- it verifies all twelve directories exist and discovers
which probe heads the predictions tables actually have.

### Step 5 — Species stratification

```bash
python tests/organism_table.py \
    --res-dir res_voxel_mixed_split_e75 \
    --emb-dir mt_voxel_mixed_split_e75 \
    --uniprot overlap_audit/uniprot.json \
    --out organisms.csv
```

Prints a ready `stack_check --restrict-to` command for each organism group
large enough to carry an interval (≥200 proteins). Compare each group against
the full mixed-split value, not against other groups.

### Step 6 — Figures

```bash
python tests/make_figures.py --audit overlap_audit --rerun rerun
```

---

## A few things to know

- **Batch size is 1.** The voxel tensor's first dimension is the residue
  count, not the protein count.
- **`masked_seqs` and `labels` arrive without BOS.** `preprocess_sequence`
  adds it inside the training step. Code that feeds the encoder directly
  needs to prepend BOS itself, or every position shifts by one silently.
- **Labels are `-100` at unmasked positions** (including padding) and the
  original token at masked positions.
- **Embedding caches must carry a `sequence` field.** `stack_check` refuses
  a cache without one, because joining on row index risks silent misalignment.

---

## Not in this repository (yet)

The pretraining corpus and the analyses that depend on it. Both land together.
Until then, the appendices on field reconstruction, the descriptor breakdown,
and the frame assumption are not independently checkable.

---

## Citation

```bibtex
@inproceedings{<KEY>,
  title     = {<PAPER TITLE>},
  author    = {<AUTHORS>},
  booktitle = {<VENUE>},
  year      = {<YEAR>}
}
```

## License

MIT. See [LICENSE](LICENSE).
