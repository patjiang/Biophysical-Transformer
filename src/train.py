#!/usr/bin/env python
"""Ablation harness for the voxel reconstruction objective.

Both arms live in this one file, differing only by ``--no-voxel``::

    python train.py --seed=0 --epochs=80               # arm A (voxel on)
    python train.py --seed=0 --epochs=80 --no-voxel    # arm B (voxel off)

Compare checkpoints on the epoch grid (``--ckpt-every``), not by val_loss --
the two arms log different quantities there.  Run ``verify_ablation.py``
first to confirm bit-identical encoder init across arms.
"""

import argparse
import math
import os
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.plugins.environments import LightningEnvironment

# ----------------------------------------------------------------- arguments

p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--no-voxel", action="store_true",
               help="ARM B: drop the voxel reconstruction objective. The only "
                    "difference between the two arms.")
p.add_argument("--seed", type=int, default=0,
               help="seeds init, masking, shuffling and dataloader workers. "
                    "The two arms MUST use the same value.")
p.add_argument("--epochs", type=int, default=40,
               help="both arms must use the same value: it sets the cosine "
                    "horizon as well as the length.")
p.add_argument("--ckpt-every", type=int, default=5,
               help="write an unconditional checkpoint every N epochs, so the "
                    "arms can be compared on a matched epoch grid.")
p.add_argument("--steps-per-epoch", type=int, default=None,
               help="pin the optimizer steps/epoch used to size the cosine "
                    "schedule. Set this if the two arms run on different "
                    "numbers of GPUs -- otherwise the schedules differ.")
p.add_argument("--devices", default="auto",
               help="'auto' or an integer. Both arms should use the same count.")
p.add_argument("--resume", default=None,
               help="continue THIS arm from one of its own checkpoints.")
p.add_argument("--out", default=None, help="override the output directory")
p.add_argument("--deterministic", action="store_true",
               help="torch.use_deterministic_algorithms(warn_only). OFF by "
                    "default, matching run.py: it swaps in slower kernels and "
                    "would make this run's numerics differ from every previous "
                    "one. Reproducibility of the INIT comes from --seed and is "
                    "checked exactly by verify_ablation.py; reproducibility of "
                    "the data order is checked empirically there too.")
p.add_argument("--smoke", action="store_true",
               help="4 train / 2 val batches x 2 epochs; exercises both "
                    "staging branches and the abort callback in ~2 minutes.")
a = p.parse_args()

ARM = "novox" if a.no_voxel else "voxel"
SAVE_DIR = a.out or (f"chkpts_smoke_{ARM}" if a.smoke
                     else f"chkpts_abl_{ARM}_s{a.seed}")
MAX_EPOCHS = 2 if a.smoke else a.epochs
NUM_WORKERS = 8

BASE_LR = 1e-4
DECODER_LR_FACTOR = 10.0  # inert in arm B (no decoder grad)
N_ACC = 4                 # must match SE3PLMLightning.n_acc
WARMUP_STEPS = 2000       # optimizer steps (~3 epochs at 700 steps/epoch)

# Seed BEFORE datasets, loaders, and model.  workers=True gives each
# dataloader worker a derived seed so masking RNG is reproducible across
# arms.  This is the line run.py never had.
pl.seed_everything(a.seed, workers=True)

# 'high' (TF32), not 'medium': loss terms run in fp32 via
# autocast(enabled=False), must match across arms.
torch.set_float32_matmul_precision('high')

from data import *
from model import *

print("=" * 70)
print(f"  ABLATION ARM {'B (no voxel)' if a.no_voxel else 'A (voxel)'}"
      f"   seed {a.seed}   {MAX_EPOCHS} epochs   -> {SAVE_DIR}/")
print("=" * 70)

# ------------------------------------------------------------------- data

train_dataset = PreprocessedBiophysicalDataset('../kstruct_pt_preprocessed/train')
vali_dataset = PreprocessedBiophysicalDataset('../kstruct_pt_preprocessed/val')

train_loader = ChainedDataLoader(
    train_dataset,
    batch_size=1,
    num_workers=NUM_WORKERS,
    collate_fn=MLMCollate(mask=True, device='cpu'),
    shuffle=True
)

vali_loader = ChainedDataLoader(
    vali_dataset,
    batch_size=1,
    num_workers=NUM_WORKERS // 2,
    collate_fn=MLMCollate(mask=False, device='cpu')
)

# Cosine horizon in optimizer steps.  Must be identical across arms or the
# LR schedules diverge and confound the ablation.  Use --steps-per-epoch
# to pin it when arms run on different hardware.
if a.steps_per_epoch:
    STEPS_PER_EPOCH = a.steps_per_epoch
    _src = f"--steps-per-epoch={a.steps_per_epoch} (pinned)"
else:
    try:
        _batches = len(train_loader)
        _devices = max(torch.cuda.device_count(), 1)
        STEPS_PER_EPOCH = max(1, math.ceil(_batches / _devices / N_ACC))
        _src = (f"len(train_loader)={_batches} / {_devices} device(s) / "
                f"n_acc={N_ACC}")
    except TypeError:
        STEPS_PER_EPOCH = 700
        _src = "fallback (loader has no length); measured ~700 previously"
TOTAL_OPT_STEPS = STEPS_PER_EPOCH * MAX_EPOCHS

print(f"[run] {STEPS_PER_EPOCH} optimizer steps/epoch  [{_src}]")
print(f"[run] {TOTAL_OPT_STEPS} total over {MAX_EPOCHS} epochs")
if not a.steps_per_epoch:
    print(f"[run] NOTE record this number. If the other arm prints a "
          f"different one, re-run it with --steps-per-epoch={STEPS_PER_EPOCH}.")

# -------------------------------------------------------------- fail fast


class AbortOnNonFinite(Callback):
    """Halt training on the first non-finite loss.

    bf16 has no GradScaler, so a NaN propagates to every downstream
    parameter on the very next step.

    Parameters
    ----------
    check_params_every : int
        Scan all parameters every this many global steps.  0 to disable.
    """

    def __init__(self, check_params_every=500):
        super().__init__()
        self.check_params_every = check_params_every

    @staticmethod
    def _loss_of(outputs):
        """Extract the loss tensor from Lightning's output format."""
        if isinstance(outputs, dict):
            return outputs.get("loss")
        return outputs if torch.is_tensor(outputs) else None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Raise on non-finite loss or (periodically) non-finite parameters."""
        loss = self._loss_of(outputs)
        if loss is not None and not torch.isfinite(loss).all():
            raise RuntimeError(
                f"[abort] non-finite training loss ({loss}) at epoch "
                f"{trainer.current_epoch}, global_step {trainer.global_step}, "
                f"batch_idx {batch_idx}.\n"
                f"        Under bf16 there is no GradScaler to skip this step, "
                f"so continuing would NaN the weights.")
        if (self.check_params_every
                and trainer.global_step % self.check_params_every == 0):
            for name, q in pl_module.named_parameters():
                if not torch.isfinite(q).all():
                    raise RuntimeError(
                        f"[abort] non-finite PARAMETER {name} at global_step "
                        f"{trainer.global_step}.")


class StepTimer(Callback):
    """Log median seconds per training batch so arm cost is measured, not assumed.

    Uses the median rather than the mean because protein length varies by
    an order of magnitude at batch_size=1 and the mean is dragged by the
    tail.

    Parameters
    ----------
    warmup : int
        Ignore the first *warmup* batches (compilation, caching).
    report_every : int
        Print a timing summary every this many batches.
    """

    def __init__(self, warmup=50, report_every=500):
        super().__init__()
        self.warmup, self.report_every, self.t, self.d = warmup, report_every, None, []

    def on_train_batch_start(self, *args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        import time
        self.t = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        import time
        if self.t is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.d.append(time.perf_counter() - self.t)
        n = len(self.d)
        if n > self.warmup and n % self.report_every == 0:
            w = sorted(self.d[self.warmup:])
            med = w[len(w) // 2]
            print(f"[timing] {ARM}: median {med * 1000:.0f} ms/batch over "
                  f"{len(w)} batches  (p10 {w[len(w) // 10] * 1000:.0f}, "
                  f"p90 {w[len(w) * 9 // 10] * 1000:.0f})")


# -------------------------------------------------------------- callbacks

# Epoch-grid checkpoints for cross-arm comparison.  val_loss means
# different things in the two arms, so don't use it to select checkpoints.
checkpoint_grid = ModelCheckpoint(
    dirpath=SAVE_DIR,
    filename="se3plm_grid_{epoch:02d}",
    every_n_epochs=max(1, a.ckpt_every),
    save_top_k=-1,
    save_on_train_epoch_end=True,
)

checkpoint_cb = ModelCheckpoint(
    dirpath=SAVE_DIR,
    filename="se3plm_{epoch:02d}-{val_loss:.4f}",
    save_top_k=15,
    monitor="val_loss",
    mode="min",
)

checkpoint_last = ModelCheckpoint(
    dirpath=SAVE_DIR,
    filename="se3plm_last_{epoch:02d}",
    save_top_k=1,
    monitor="epoch",
    mode="max",
)

# ------------------------------------------------------------------ model
#
# Every value is copied from run.py.  verify_ablation.py diffs this dict
# against run.py's SE3PLMLightning(...) call, so drift is caught by a
# check rather than by a reader.

HPARAMS = dict(
    learning_rate=BASE_LR,
    decoder_lr_factor=DECODER_LR_FACTOR,
    weight_decay=1e-4,
    optimizer_type="adamw",
    cyclic_lr=False,
    cyclic_gamma_steps=TOTAL_OPT_STEPS,
    warmup_steps=WARMUP_STEPS,
    n_acc=N_ACC,

    # Rotation-invariant objective only; voxel-wise weights are zero.
    w_pattern=0.0,
    w_amplitude=0.0,
    w_occupancy=0.0,
    w_highfreq=0.0,
    w_wasserstein=0.0,
    w_invariant=1.0,
    w_inv_mean=0.5,
    w_inv_energy=0.5,
    w_inv_tv=0.25,
    inv_mask_sphere=True,
    occ_rel=3.0,
    recon_weight=1.0,

    # Collapse guard (replaces encoder_variance_floor).
    lambda_collapse=0.1,
    lambda_collapse_latent=0.1,
    collapse_target_std=1.0,
    collapse_cov_weight=1.0,
    collapse_target_share=0.25,
    collapse_share_weight=1.0,
    collapse_pr_target=20.0,

    # Copy term off; identity skip on.
    mlm_copy_frac=0.0,
    mlm_copy_tau=0.1,
    id_skip=True,
    w_cross=0.0,

    # Masked voxel reconstruction.
    mask_in_step=True,
    mlm_mask_rate=0.2,
    mlm_span_residues=3,
    mlm_replace_frac=0.2,
    w_masked_recon=3.0,

    # iBOT self-distillation with local teacher.
    lambda_ibot=1.0,
    ibot_prototypes=256,
    ibot_queue_length=8192,
    ibot_student_temp=0.1,
    ibot_teacher_temp=0.05,
    lambda_gram=0.0,
    ibot_teacher_band=2,

    # InteractionTokenBlock (not tuning knobs; see preflight.py).
    itb_token_init=0.1,
    itb_n_tokens=32,
    itb_n_heads=2,
    itb_layer_scale=0.3,
    rank_log_every=1,
    mlm_bias_path="mlm_unigram_bias.pt",
)

model = SE3PLMLightning(no_voxel=a.no_voxel, **HPARAMS)

# ----------------------------------------------------------------- logging


def setup_loggers(save_dir):
    """Create timestamped TensorBoard and CSV loggers under ``save_dir``."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return [TensorBoardLogger(save_dir=save_dir, name="tb_logs", version=timestamp),
            CSVLogger(save_dir=save_dir, name="csv_logs", version=timestamp)]


devices = a.devices if a.devices == "auto" else int(a.devices)

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="auto",
    devices=devices,
    plugins=[LightningEnvironment()],
    default_root_dir=SAVE_DIR,
    callbacks=[checkpoint_grid, checkpoint_cb, checkpoint_last,
               LearningRateMonitor(logging_interval="step"),
               AbortOnNonFinite(), StepTimer()],
    precision='bf16-mixed',   # NOT fp16: GradScaler silently swallows NaNs
    log_every_n_steps=1,
    logger=setup_loggers(SAVE_DIR),
    strategy='ddp_find_unused_parameters_true',
    profiler=None if a.smoke else "simple",
    deterministic=('warn' if a.deterministic else None),
    **({'limit_train_batches': 4, 'limit_val_batches': 2,
        'num_sanity_val_steps': 0} if a.smoke else {}),
)

if a.resume:
    print(f"[run] resuming {ARM} from {a.resume}")
    if ARM not in os.path.basename(os.path.dirname(a.resume) or ""):
        print(f"[run] WARNING {a.resume} does not look like it came from this "
              f"arm ({ARM}). Resuming across arms silently invalidates the "
              f"comparison -- check this before continuing.")

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=vali_loader,
            ckpt_path=a.resume)
