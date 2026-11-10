"""
BioT LightningModule: networks, staging, optimisation and logging.
"""

import dataclasses

import contextlib
import copy
import inspect
import math
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from blocks import (
    TransformerStackEnc, FactorizedMLMHead, InteractionTokenBlock,
    VoxelizeTrunk, collapse_penalty, effective_rank,
)
import loss as L
from span_masking import span_mask_tokens

PAD_TOKEN_ID = 8000
MASK_TOKEN_ID = 8001
BOS_TOKEN_ID = 8002

_BLOCKS_PATCH = """
blocks.py / collapse_penalty needs a `pr_target` kwarg.  Replace the
existing pr_ref line (do NOT leave the old one above the new ones, or
`iso` is read before assignment):

    def collapse_penalty(z, ..., pr_target=None):
        ...
        iso    = max(D * dof / (dof + D), 1.0 + 1e-6)
        pr_ref = float(pr_target) if pr_target is not None else iso
        pr_ref = max(min(pr_ref, iso), 1.0 + 1e-6)   # clamp to iso ceiling
        ...

The clamp prevents pr_target from asking for a participation ratio the
sample size can't reach (which would make spread_loss unsatisfiable).
Keep the F.relu -- without it, PR above pr_ref hands out a bonus for
super-isotropy, and a lower pr_ref widens the range where that happens.
"""


class BioTLightning(pl.LightningModule):
    """SE3 protein language model with staged reconstruction training.

    Stage 1 (epochs < ``recon_start_epoch``): MLM + DINO + iBOT only.
    Stage 2: adds the rotation-invariant voxel reconstruction objective,
    the decoder (``self.b``, ``self.i``), the identity skip, and the
    latent collapse guard.

    When ``no_voxel=True`` the model stays in stage 1 for the entire run,
    giving the control arm of the voxel ablation.
    """

    def __init__(self,
                 learning_rate: float = 1e-4,
                 decoder_lr_factor: float = 10.0,
                 weight_decay: float = 1e-4,
                 optimizer_type: str = "adamw",
                 adam_beta1: float = 0.9,
                 adam_beta2: float = 0.95,
                 cyclic_lr: bool = False,
                 max_lr_factor: float = 30.0,
                 step_size_up: int = 5000,
                 cyclic_gamma_steps: int = None,
                 warmup_steps: int = 2000,
                 min_lr_factor: float = 0.05,
                 lambda_dino: float = 0.1,
                 mlm_weight: float = 2.0,
                 # Copy term: CE at unmasked positions, hinged at tau so it
                 # acts as a floor rather than rewarding ever-more margin.
                 mlm_copy_frac: float = 0.2,
                 mlm_copy_tau: float = 0.1,
                 num_prototypes: int = 128,
                 teacher_momentum: float = 0.999,
                 dino_warmup_steps: int = 60000,
                 vocab_size: int = 8003,
                 queue_length: int = 4096,
                 latent_dim: int = 64,
                 encoder_dropout: float = 0.0,
                 # -- Reconstruction weights --
                 # All terms in loss.py are O(1), so these are directly
                 # comparable.  pattern_loss = 1.0 means residue-independent
                 # output (sphere/blob).
                 w_pattern: float = 1.0,
                 w_amplitude: float = 0.5,
                 w_occupancy: float = 0.5,
                 w_highfreq: float = 0.1,
                 w_wasserstein: float = 0.0,
                 occ_rel: float = 3.0,
                 pyramid_levels: int = 4,
                 # Rotation-invariant objective (invariant_loss.py).
                 # Voxel-wise weights should be 0 when this is on: any
                 # nonzero value pulls toward the sphere.
                 w_invariant: float = 0.0,
                 w_inv_mean: float = 0.5,
                 w_inv_energy: float = 0.5,
                 w_inv_tv: float = 0.25,
                 inv_mask_sphere: bool = True,
                 # Cross-residue terms.  OFF: requires absolute orientation
                 # in the protein frame, which sequence alone can't supply.
                 w_cross: float = 0.0,
                 w_cross_mean: float = 0.5,
                 cross_window: int = 10,
                 recon_weight: float = 1.0,
                 mlm_bias_path: str = None,
                 mlm_bias_scale: float = 0.25,
                 # -- MLM masking --
                 mlm_mask_rate: float = 0.15,
                 mlm_span_residues: int = 1,
                 # Re-mask in training_step instead of using the collate.
                 mask_in_step: bool = False,
                 # Extra reconstruction term over masked rows only (where
                 # the skip carries E[MASK] and the decoder has only z).
                 w_masked_recon: float = 0.0,
                 # Fraction of selected spans corrupted instead of masked.
                 mlm_replace_frac: float = 0.0,
                 # -- iBOT self-distillation --
                 lambda_ibot: float = 0.0,
                 ibot_prototypes: int = 256,
                 ibot_queue_length: int = 8192,
                 ibot_student_temp: float = 0.1,
                 ibot_teacher_temp: float = 0.05,
                 # Gram anchoring (OFF: gram_corr stayed at 0.04).
                 lambda_gram: float = 0.0,
                 # Per-layer half-width of the iBOT teacher's attention
                 # band.  0 = full attention.
                 ibot_teacher_band: int = 0,
                 # -- Identity skip --
                 # Zero-gated skip from token embedding into the decoder.
                 # Exact no-op at init (gate=0), safe to resume from any
                 # checkpoint.
                 id_skip: bool = True,
                 id_gate_init: float = 0.0,
                 # -- Collapse guard (replaces encoder_variance_floor) --
                 # Two weights: encoder output and latent (self.i output),
                 # both within-sequence.
                 lambda_collapse: float = 0.1,
                 lambda_collapse_latent: float = 0.1,
                 collapse_target_std: float = 1.0,
                 collapse_cov_weight: float = 1.0,
                 # Rank floor replacing the isotropy target.  None restores
                 # the old isotropy target.
                 collapse_pr_target: float = 20.0,
                 # Share term: the only term that can see a protein-level
                 # constant (within-centring deletes it from scale/spread).
                 collapse_target_share: float = 0.25,
                 collapse_share_weight: float = 1.0,
                 # -- InteractionTokenBlock --
                 # Not tuning knobs; run preflight.py before changing these.
                 itb_token_init: float = 0.1,
                 itb_n_tokens: int = 32,
                 itb_n_heads: int = 2,
                 itb_layer_scale: float = 0.3,
                 rank_log_every: int = 1,
                 # -- Ablation switch --
                 # Removes the voxel objective for the whole run.  The
                 # encoder, MLM, DINO, and iBOT are unchanged.
                 no_voxel: bool = False,
                 # -- Staged training --
                 recon_start_epoch: int = 1,
                 n_acc: int = 4,
                 me_max_over_ema: bool = True,
                 **kwargs):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.lambda_dino = lambda_dino
        self.recon_w = L.ReconWeights(
            pattern=w_pattern, amplitude=w_amplitude, occupancy=w_occupancy,
            highfreq=w_highfreq, wasserstein=w_wasserstein,
            occ_rel=occ_rel, pyramid_levels=pyramid_levels,
            invariant=w_invariant, inv_mean=w_inv_mean,
            inv_energy=w_inv_energy, inv_tv=w_inv_tv,
            inv_mask_sphere=inv_mask_sphere,
            cross=w_cross, cross_mean=w_cross_mean, cross_window=cross_window)
        # Masked rows aren't contiguous, so no cross terms in the masked-only
        # reconstruction term.
        self.recon_w_masked = dataclasses.replace(self.recon_w, cross=0.0)
        self.teacher_momentum = teacher_momentum
        self.dino_warmup_steps = dino_warmup_steps
        self._lr_group_index = {}

        # Fail loudly if blocks.py doesn't support pr_target yet.
        self._cp_pr_target_ok = (
            'pr_target' in inspect.signature(collapse_penalty).parameters)
        if collapse_pr_target is not None and not self._cp_pr_target_ok:
            raise RuntimeError(
                "collapse_pr_target=%r was requested but blocks.py's "
                "collapse_penalty() does not accept `pr_target`, so the guard "
                "would silently keep its isotropy target and fight the "
                "identity skip.\n%s\nPass collapse_pr_target=None to run "
                "against the old target on purpose."
                % (collapse_pr_target, _BLOCKS_PATCH))

        self.student_encoder = TransformerStackEnc(
            input_dim=latent_dim, output_dim=latent_dim,
            num_layers=5, vocab_size=vocab_size, dropout=encoder_dropout)
        if encoder_dropout > 0:
            print(f"[init] WARNING encoder_dropout={encoder_dropout}: "
                  "enc_between_ratio logged in train() mode will be inflated "
                  "by dropout noise. Check it under eval() before trusting it.")
        # Scale embedding init down from N(0,1) so initial logits aren't huge.
        # Must happen BEFORE the teacher deepcopy.
        nn.init.normal_(self.student_encoder.embedding.weight, mean=0.0, std=0.02)

        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        self.teacher_encoder.requires_grad_(False)

        self.mlm_head = FactorizedMLMHead(self.student_encoder.embedding,
                                          latent_dim=latent_dim)
        self._init_mlm_bias(mlm_bias_path, vocab_size, mlm_bias_scale)

        self.dino = L.DinoLoss(latent_dim=latent_dim,
                               num_prototypes=num_prototypes,
                               queue_length=queue_length,
                               me_max_over_ema=me_max_over_ema)
        # Built only when on, so checkpoints without it load cleanly.
        self.ibot = (L.ResidueDistillLoss(
            latent_dim=latent_dim, num_prototypes=ibot_prototypes,
            queue_length=ibot_queue_length, student_temp=ibot_student_temp,
            teacher_temp=ibot_teacher_temp) if lambda_ibot > 0 else None)
        self._kmer_codec = None
        if ibot_teacher_band > 0:
            if self.ibot is None:
                raise ValueError("ibot_teacher_band > 0 needs lambda_ibot > 0")
            self._check_banded_mirror()
        self.i = InteractionTokenBlock(
            latent_dim, n_tokens=itb_n_tokens, n_heads=itb_n_heads,
            layer_scale=itb_layer_scale, token_init=itb_token_init)
        self.b = VoxelizeTrunk()

        # --- Identity skip ---
        # id_proj gets a normal init (not zeros): at gate=0 the gradient
        # w.r.t. id_proj is identically 0, so the gate opens first and the
        # projection only learns once there's a path for it.  Zero-init on
        # BOTH would stall permanently.
        if id_skip:
            self.id_proj = nn.Linear(latent_dim, latent_dim, bias=False)
            self.id_gate = nn.Parameter(torch.full((), float(id_gate_init)))
            print(f"[init] identity skip ON, gate init {id_gate_init:g} "
                  f"({'exact no-op at step 0' if id_gate_init == 0.0 else 'NOT a no-op'})")
        else:
            self.id_proj, self.id_gate = None, None
            print("[init] identity skip OFF (old graph)")

        self.step = 0

        if no_voxel:
            inert = {k: v for k, v in (
                ('w_invariant', w_invariant), ('w_inv_mean', w_inv_mean),
                ('w_inv_energy', w_inv_energy), ('w_inv_tv', w_inv_tv),
                ('w_masked_recon', w_masked_recon), ('w_cross', w_cross),
                ('lambda_gram', lambda_gram),
                ('lambda_collapse_latent', lambda_collapse_latent),
            ) if v}
            print("\n" + "=" * 70
                  + "\n[init] NO-VOXEL CONTROL ARM. The voxel objective is off "
                    "for the whole run."
                  + "\n       Live: MLM"
                    f"(w={mlm_weight:g}) DINO(w={lambda_dino:g}) "
                    f"iBOT(w={lambda_ibot:g}, band={ibot_teacher_band}) "
                    f"enc-collapse(w={lambda_collapse:g})"
                  + "\n       Off:  reconstruction, masked reconstruction, the "
                    "identity skip, self.i, self.b,"
                  + "\n             the latent collapse guard, and the whole "
                    "voxel validation path."
                  + (f"\n       INERT (configured but unreachable): "
                     f"{', '.join(f'{k}={v:g}' for k, v in inert.items())}"
                     if inert else "")
                  + "\n       val_loss is MLM CE here (~8.8) against the voxel "
                    "arm's recon total (~0.3)."
                  + "\n       The two are NOT comparable: select checkpoints by "
                    "EPOCH, not by val_loss.\n"
                  + "=" * 70 + "\n")

        # Build the codec eagerly so an invertibility failure shows up at
        # construction, not at the first training step.
        if mlm_replace_frac > 0:
            try:
                self._codec()
            except RuntimeError as e:
                print(f"[init] WARNING {e}; the codec will be built on first use")

    def _init_mlm_bias(self, mlm_bias_path, vocab_size, mlm_bias_scale):
        """Load and rescale the unigram bias for the MLM head."""
        if mlm_bias_path and Path(mlm_bias_path).exists():
            b = torch.load(mlm_bias_path, map_location="cpu").float()
            if b.shape[0] != vocab_size:
                print(f"[init] WARNING bias shape {b.shape} != {vocab_size}, skipped")
                return
            p = b.exp()
            p = p / p.sum()
            # Only the spread competes with learned logits; specials
            # (8000-8002) are left untouched to suppress PAD/MASK.
            n_real = vocab_size - 3
            mu = b[:n_real].mean()
            std_before = b[:n_real].std().item()
            b[:n_real] = mu + mlm_bias_scale * (b[:n_real] - mu)
            self.mlm_head.bias.data.copy_(b)
            ce = -(p * torch.log_softmax(b, dim=0)).sum()
            print(f"[init] MLM bias: real-token std {std_before:.3f} -> "
                  f"{b[:n_real].std().item():.3f} (scale {mlm_bias_scale}), "
                  f"special offset {(b[n_real:].mean() - mu).item():.1f} nats, "
                  f"expected initial CE ~= {ce:.4f}")
        elif mlm_bias_path:
            print(f"[init] WARNING {mlm_bias_path} not found; bias stays zero "
                  f"(initial CE = ln(V) = {math.log(vocab_size):.4f})")

    # ------------------------------------------------- checkpoint compat

    def on_load_checkpoint(self, checkpoint):
        """Remap legacy keys and handle missing/extra keys for compatibility.

        Covers: pre-loss.py-split weight names, removed discriminator keys,
        identity skip (id_proj/id_gate) present in one direction but not the
        other, and iBOT keys likewise.  All insertions use init values so a
        resume is safe by construction.
        """
        sd = checkpoint.get("state_dict")
        if sd is None:
            return
        new_sd, n, n_drop = L.remap_legacy_state_dict(sd)
        if n:
            print(f"[ckpt] remapped {n} pre-split key(s).")
        if n_drop:
            print(f"[ckpt] dropped {n_drop} key(s) for removed modules "
                  "(discriminator, adaptive loss weights).")
        own = self.state_dict()
        skip_keys = ('id_proj.weight', 'id_gate')
        added = [k for k in skip_keys if k in own and k not in new_sd]
        for k in added:
            new_sd[k] = own[k].clone()
        if added:
            print(f"[ckpt] identity skip absent from checkpoint; inserted "
                  f"{added} at init values (gate "
                  f"{float(own['id_gate']):g} -> forward pass unchanged).")
        stale = [k for k in skip_keys if k in new_sd and k not in own]
        for k in stale:
            new_sd.pop(k)
        if stale:
            print(f"[ckpt] identity skip is OFF; dropped {stale} from the "
                  "checkpoint.")
        ib_own = [k for k in own if k.startswith('ibot.')]
        ib_add = [k for k in ib_own if k not in new_sd]
        for k in ib_add:
            new_sd[k] = own[k].clone()
        if ib_add:
            print(f"[ckpt] ibot absent from checkpoint; inserted {len(ib_add)} "
                  "tensor(s) at init values.")
        ib_stale = [k for k in new_sd if k.startswith('ibot.') and k not in own]
        for k in ib_stale:
            new_sd.pop(k)
        if ib_stale:
            print(f"[ckpt] ibot is OFF; dropped {len(ib_stale)} tensor(s) from "
                  "the checkpoint.")
        checkpoint["state_dict"] = new_sd

    # ----------------------------------------- banded iBOT teacher

    @staticmethod
    def _band_mask(T, n_reg, band, device):
        """Build a ``(T, T)`` bool attention mask for ``[BOS, registers, body]``.

        Body rows attend only to body columns within +/-band; BOS and
        register rows attend everywhere (their outputs aren't used as
        targets).  Every row keeps at least itself.
        """
        idx = torch.arange(T, device=device)
        body = idx >= n_reg + 1
        near = (idx[:, None] - idx[None, :]).abs() <= band
        return torch.where(body[:, None], body[None, :] & near,
                           torch.ones(T, T, dtype=torch.bool, device=device))

    @torch.no_grad()
    def _teacher_forward(self, seq, band=0):
        """Teacher encoder forward with optional banded attention.

        Mirrors ``TransformerStackEnc.forward`` step for step; with
        ``band=0`` the computation is identical (verified at init by
        ``_check_banded_mirror``).  Returns ``(B, 1+L, C)`` with BOS at 0.
        """
        enc = self.teacher_encoder
        x = enc.embedding(seq)
        regs = enc.registers.expand(x.shape[0], -1, -1).to(x.dtype)
        x = torch.cat([x[:, 0:1, :], regs, x[:, 1:, :]], dim=1)
        mask = (self._band_mask(x.shape[1], enc.num_reg, band, x.device)
                if band > 0 else None)
        for layer in enc.layers1:
            x = layer(x, mask=mask)
        for layer in enc.layers2:
            x = layer(x, mask=mask)
        return torch.cat([x[:, 0:1, :], x[:, enc.num_reg + 1:, :]], dim=1)

    def _check_banded_mirror(self):
        """Assert that ``_teacher_forward(band=0)`` reproduces the encoder."""
        enc = self.teacher_encoder
        for name in ('embedding', 'registers', 'layers1', 'layers2', 'num_reg'):
            if not hasattr(enc, name):
                raise RuntimeError(
                    f"ibot_teacher_band: the encoder has no `{name}`; "
                    "_teacher_forward mirrors TransformerStackEnc.forward and "
                    "has to be updated for this encoder.")
        if 'mask' not in inspect.signature(enc.layers1[0].forward).parameters:
            raise RuntimeError("ibot_teacher_band: encoder layers take no "
                               "`mask`; the banded teacher needs it.")
        was = enc.training
        enc.eval()
        g = torch.Generator().manual_seed(0)
        seq = torch.randint(0, self.hparams.vocab_size - 3, (1, 24), generator=g)
        seq[0, 0] = BOS_TOKEN_ID
        with torch.no_grad():
            a, b = enc(seq), self._teacher_forward(seq, band=0)
        enc.train(was)
        if a.shape != b.shape or not torch.allclose(a, b, atol=1e-5, rtol=1e-5):
            raise RuntimeError(
                "ibot_teacher_band: _teacher_forward(band=0) does not reproduce "
                "TransformerStackEnc.forward (max diff "
                f"{float((a - b).abs().max()) if a.shape == b.shape else 'shape'})"
                "; the mirror is stale for this encoder.")
        print(f"[init] banded iBOT teacher: +-{self.hparams.ibot_teacher_band} per "
              f"layer over {len(enc.layers1) + len(enc.layers2)} layers = "
              f"+-{self.hparams.ibot_teacher_band * (len(enc.layers1) + len(enc.layers2))}"
              " residues; mirror of the encoder forward verified")

    def _gram(self, z, body_tokens, tdat):
        """Gram anchoring loss and correlation between z and descriptor Gram.

        Returns ``(loss, corr)``, both zero when the protein has fewer than
        three real residues.
        """
        real = (body_tokens != PAD_TOKEN_ID).reshape(-1)
        zero = torch.zeros((), device=z.device)
        if int(real.sum()) < 3:
            return zero, zero
        with torch.amp.autocast('cuda', enabled=False):
            t = tdat.unsqueeze(1) if tdat.dim() == 4 else tdat
            t = t[real].float() * L.sphere_mask(t.shape[-1], t.device)
            desc = L._descriptor(t.shape[-1], self.recon_w.inv_l_max,
                                 self.recon_w.inv_shells, t.device)
            with torch.no_grad():
                rows = L.descriptor_rows(t[:, 0], desc)
            zb = z[:, 1:, :].reshape(-1, z.shape[-1])[real]
            return L.gram_losses(zb, rows)

    def _codec(self):
        """Build or return the KmerCodec for corrupted spans."""
        if self.hparams.mlm_replace_frac <= 0:
            return None
        if self._kmer_codec is None:
            from span_masking import KmerCodec
            try:
                from data import seqs_to_tens
            except ImportError as e:
                raise RuntimeError(
                    "mlm_replace_frac > 0 needs data.seqs_to_tens to build the "
                    "3-mer codec, and `data` could not be imported") from e
            self._kmer_codec = KmerCodec.from_seqs_to_tens(seqs_to_tens)
            print(f"[init] 3-mer codec built from data.seqs_to_tens; corrupted "
                  f"spans at {self.hparams.mlm_replace_frac:g} of selected spans")
        return self._kmer_codec

    # ------------------------------------------------------------- staging

    def _recon_active(self):
        """Whether the reconstruction objective is live this epoch."""
        if self.hparams.get('no_voxel', False):
            return False
        return self.current_epoch >= self.hparams.recon_start_epoch

    def on_train_epoch_start(self):
        self._rank_cache = {}
        if (self.current_epoch == self.hparams.recon_start_epoch
                and not self.hparams.get('no_voxel', False)):
            print(f"\n--- Epoch {self.current_epoch}: reconstruction enabled ---\n")

    # ------------------------------------------------------------- forward

    def _identity_skip(self, body_tokens, dtype):
        """``id_gate * id_proj(embedding(tok))`` for body tokens, ``(B, L-1, C)``.

        Reads the student embedding (shared with the MLM head).  At masked
        positions this injects ``id_gate * id_proj(E[MASK])`` -- a
        residue-independent constant -- not the true identity.  That's
        required: leaking the true token at a masked position would gut
        MLM pressure.

        Returns ``None`` when the skip is off.
        """
        if self.id_proj is None:
            return None
        e = self.student_encoder.embedding(body_tokens)
        return self.id_gate * self.id_proj(e.to(dtype))

    def _forward_full(self, seq):
        """Full forward: encoder -> self.i -> identity skip -> decoder.

        Returns ``(z, latent_core, latent, recon)``.

        ``latent_core`` is self.i's output (what the collapse guard reads).
        ``latent`` is ``latent_core + skip`` (what the decoder consumes).
        They're separate because guarding the post-skip tensor would
        penalise the anisotropy the skip exists to reintroduce and drive
        ``id_gate`` back to zero.  With the skip off they're the same object.
        """
        z = self.student_encoder(seq)
        bos = z[:, 0, :]
        zr = z[:, 1:, :]
        bos_expanded = repeat(bos, 'b c -> (b l) 1 c', l=z.shape[1] - 1)
        with torch.amp.autocast('cuda', enabled=False):
            latent_core = self.i(zr.to(self._i_dtype()))
            skip = self._identity_skip(seq[:, 1:], latent_core.dtype)
            latent = latent_core if skip is None else latent_core + skip
        recon = self.b(rearrange(latent, 'b l c -> (b l) 1 c'), bos_expanded)
        return z, latent_core, latent, recon

    def _i_dtype(self):
        """Dtype for self.i: its own parameters' dtype.

        Under bf16-mixed training this is fp32 (autocast disabled around
        self.i).  Under ``model.to(bfloat16)`` in eval harnesses it's bf16.
        """
        return next(self.i.parameters()).dtype

    def forward(self, seq, pred=False):
        if pred:
            return self.student_encoder(seq)
        z, _, _, recon = self._forward_full(seq)
        return z, recon

    def forward_int(self, seq, pred=True, include_skip=True):
        """Return the decoder's input ``(B, L-1, C)``.

        ``include_skip=True`` gives what the decoder actually consumes;
        ``False`` gives self.i's output alone.
        """
        z = self.student_encoder(seq)
        with torch.amp.autocast('cuda', enabled=False):
            zr = self.i(z[:, 1:, :].to(self._i_dtype()))
            if include_skip:
                skip = self._identity_skip(seq[:, 1:], zr.dtype)
                if skip is not None:
                    zr = zr + skip
        return zr

    # ------------------------------------------------------------- utils

    def preprocess_sequence(self, seqs):
        """Prepend BOS token."""
        bos = torch.full((seqs.shape[0], 1), BOS_TOKEN_ID,
                         device=seqs.device, dtype=seqs.dtype)
        return torch.cat([bos, seqs], dim=1)

    def _guard_kwargs(self, mask):
        """Build kwargs for ``collapse_penalty``."""
        kw = dict(target_std=self.hparams.collapse_target_std,
                  cov_weight=self.hparams.collapse_cov_weight,
                  target_share=self.hparams.collapse_target_share,
                  share_weight=self.hparams.collapse_share_weight,
                  mask=mask)
        if self._cp_pr_target_ok:
            kw['pr_target'] = self.hparams.collapse_pr_target
        return kw

    def _copy_ce(self, logits_flat, target_flat, select, tau=None):
        """CE at positions where the model was shown the real token.

        Returns ``(loss, raw_ce, n_tokens, frac_active)``.  ``loss`` is the
        hinged training term; ``raw_ce`` is plain mean CE for logging.
        All but ``n_tokens`` are ``None`` when the selection is empty.
        """
        n = int(select.sum())
        if n == 0:
            return None, None, 0, None
        ce = F.cross_entropy(logits_flat[select], target_flat[select],
                             reduction='none')
        raw = ce.mean()
        if not tau:
            return raw, raw, n, torch.ones((), device=ce.device)
        return (F.relu(ce - tau).mean(), raw, n,
                (ce.detach() > tau).float().mean())

    def _sync_ctx(self):
        """No-sync context manager for gradient accumulation steps."""
        if self.trainer.num_devices > 1:
            return getattr(self.trainer.model, "no_sync", contextlib.nullcontext)()
        return contextlib.nullcontext()

    @torch.no_grad()
    def _update_teacher_ema(self):
        """EMA update of the teacher encoder and distillation heads."""
        m = self.teacher_momentum
        for sp, tp in zip(self.student_encoder.parameters(),
                          self.teacher_encoder.parameters()):
            tp.data.mul_(m).add_(sp.data, alpha=1 - m)
        self.dino.update_teacher(m)
        if self.ibot is not None:
            self.ibot.update_teacher(m)

    def _group_lrs(self, opt, device):
        """Live learning rate per parameter group for logging."""
        pg = getattr(opt, "optimizer", opt).param_groups
        return {f'lr_{k}': torch.tensor(float(pg[i]['lr']), device=device)
                for k, i in self._lr_group_index.items() if i < len(pg)}

    # ------------------------------------------------------------- train

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None
        V = self.hparams.vocab_size
        N_ACC = self.hparams.n_acc
        recon_active = self._recon_active()

        opt_g = self.optimizers()
        if batch_idx % N_ACC == 0:
            opt_g.zero_grad()

        tdat, masked_seqs, labels, _, full_seqs = batch
        if self.hparams.mask_in_step:
            masked_seqs, labels = span_mask_tokens(
                full_seqs, pad_id=PAD_TOKEN_ID, mask_id=MASK_TOKEN_ID,
                mask_rate=self.hparams.mlm_mask_rate,
                span_residues=self.hparams.mlm_span_residues,
                replace_frac=self.hparams.mlm_replace_frac, codec=self._codec())
        bos_label = torch.full((labels.shape[0], 1), -100,
                               dtype=torch.int64, device=labels.device)
        labels_flat = torch.cat([bos_label, labels], dim=1).view(-1)

        masked_seqs = self.preprocess_sequence(masked_seqs)
        full_seqs = self.preprocess_sequence(full_seqs)

        if not (masked_seqs[:, 1:] != PAD_TOKEN_ID).any():
            # All-padding batch: collapse_penalty's share term would divide
            # by zero and propagate NaN into Adam's moments.
            print(f"[warn] all-padding batch at idx {batch_idx}; skipped.")
            return None

        if recon_active:
            z, latent_core, latent, rec = self._forward_full(masked_seqs)
        else:
            # Stage 1: skip self.i / self.b entirely.  Don't add a
            # 0.0 * touch term -- that populates zero grads and makes
            # AdamW apply weight decay, driving the decoder to zero.
            z = self.forward(masked_seqs, pred=True)
            latent_core = latent = rec = None

        zero = torch.zeros((), device=z.device)

        mlm_logits = self.mlm_head(z)
        logits_flat = mlm_logits.view(-1, V).float()
        valid = labels_flat != -100
        raw_mlm_loss = (F.cross_entropy(logits_flat, labels_flat,
                                        reduction='mean')
                        if valid.any() else 0.0 * torch.mean(mlm_logits))

        # Copy term at positions the model was SHOWN (disjoint from `valid`
        # by construction).  The `shown == target` clause ensures correctness
        # under any masking scheme.
        full_flat = full_seqs.reshape(-1)
        shown_flat = masked_seqs.reshape(-1)
        copy_sel = (~valid
                    & (full_flat != PAD_TOKEN_ID)
                    & (full_flat != BOS_TOKEN_ID)
                    & (shown_flat == full_flat))
        copy_loss, copy_raw, n_copy, copy_active = self._copy_ce(
            logits_flat, full_flat, copy_sel, tau=self.hparams.mlm_copy_tau)
        copy_w = self.hparams.mlm_weight * self.hparams.mlm_copy_frac

        with torch.no_grad():
            teacher_z = self.teacher_encoder(full_seqs)
        dino_loss = self.dino(z, teacher_z)

        # iBOT: student predicts teacher's per-residue prototype assignment
        # at masked/corrupted positions.
        if self.ibot is not None:
            real_flat = (full_flat != PAD_TOKEN_ID) & (full_flat != BOS_TOKEN_ID)
            band = self.hparams.ibot_teacher_band
            t_ibot = self._teacher_forward(full_seqs, band) if band > 0 else teacher_z
            ibot_loss, ibot_m = self.ibot(z, t_ibot, valid, real_flat)
            ibot_metrics = {'train_ibot_loss': ibot_loss.detach(), **ibot_m}
        else:
            ibot_loss, ibot_metrics = zero, {}

        # Gram anchoring (encoder-only, both stages).
        if self.hparams.lambda_gram > 0:
            gram_loss, gram_corr = self._gram(z, masked_seqs[:, 1:], tdat)
            gram_metrics = {'train_gram_loss': gram_loss.detach(),
                            'gram_corr': gram_corr}
        else:
            gram_loss, gram_metrics = zero, {}

        # Collapse guard on encoder output and self.i output (pre-skip).
        # Padding rows are excluded: they're near-duplicates that would
        # drag within-mean toward the pad embedding.
        pad_mask = masked_seqs[:, 1:] != PAD_TOKEN_ID
        gkw = self._guard_kwargs(pad_mask)
        collapse_loss, cmet = collapse_penalty(z[:, 1:, :], prefix='enc', **gkw)
        if latent_core is not None:
            lat_collapse, lmet = collapse_penalty(latent_core, prefix='lat', **gkw)
        else:
            # Stage 1: the latent guard is off (no reconstruction signal
            # means it would just push toward whiteness).
            lat_collapse, lmet = torch.zeros((), device=z.device), {}

        rc = (L.reconstruction_losses(rec, tdat, self.recon_w, None)
              if recon_active else L.zero_recon(z.device))

        # Masked-row reconstruction: the skip carries E[MASK] at these rows,
        # so this term reaches the decoder only through z.
        rc_m, n_masked_res = None, 0
        if recon_active and self.hparams.w_masked_recon > 0:
            m_rows = labels.reshape(-1) != -100
            n_masked_res = int(m_rows.sum())
            if n_masked_res >= 2:
                rc_m = L.reconstruction_losses(rec[m_rows], tdat[m_rows],
                                               self.recon_w_masked, None,
                                               with_diagnostics=False)

        total_g_loss = (self.hparams.mlm_weight * raw_mlm_loss
                        + self.lambda_dino * dino_loss
                        + self.hparams.lambda_collapse * collapse_loss
                        + self.hparams.lambda_collapse_latent * lat_collapse)
        if copy_loss is not None and copy_w != 0.0:
            total_g_loss = total_g_loss + copy_w * copy_loss
        if self.ibot is not None:
            total_g_loss = total_g_loss + self.hparams.lambda_ibot * ibot_loss
        if self.hparams.lambda_gram > 0:
            total_g_loss = total_g_loss + self.hparams.lambda_gram * gram_loss
        if recon_active:
            total_g_loss = (total_g_loss
                            + self.hparams.recon_weight * rc['recon_total'])
            if rc_m is not None:
                total_g_loss = (total_g_loss
                                + self.hparams.recon_weight
                                * self.hparams.w_masked_recon
                                * rc_m['recon_total'])

        accumulating = (batch_idx + 1) % N_ACC != 0
        ctx = self._sync_ctx() if accumulating else contextlib.nullcontext()
        with ctx:
            self.manual_backward(total_g_loss / N_ACC)
        if not accumulating:
            mods = [(self.student_encoder, 4.0), (self.mlm_head, 4.0),
                    (self.dino.student_dino_head, 4.0)]
            if self.ibot is not None:
                mods.append((self.ibot.student_head, 4.0))
            if recon_active:
                mods += [(self.b, 4.0), (self.i, 4.0)]
                if self.id_proj is not None:
                    mods += [(self.id_proj, 4.0)]
            for mod, mx in mods:
                torch.nn.utils.clip_grad_norm_(mod.parameters(), max_norm=mx)
            if self.id_gate is not None and self.id_gate.grad is not None:
                torch.nn.utils.clip_grad_norm_([self.id_gate], max_norm=1.0)
            opt_g.step()
            self._update_teacher_ema()
            sched = self.lr_schedulers()
            if sched is not None:
                (sched[0] if isinstance(sched, list) else sched).step()

        # ---- diagnostics ----
        with torch.no_grad():
            proto_ent = self.dino.prototype_entropy(z[:, 0, :])

            if valid.any():
                pred = logits_flat.argmax(-1)
                mlm_acc = (pred[valid] == labels_flat[valid]).float().mean()
                n_masked = valid.sum().float()
            else:
                mlm_acc = n_masked = zero

            copy_metrics = {
                'train_n_copy': torch.tensor(float(n_copy), device=z.device)}
            local_metrics = {}
            if copy_loss is not None:
                copy_ce_d = copy_raw.detach()
                local_metrics['train_copy_ce'] = copy_ce_d
                local_metrics['train_copy_ppl'] = torch.exp(
                    copy_ce_d.clamp(max=20.0))
                local_metrics['train_copy_active'] = copy_active

            h = self.mlm_head.norm(self.mlm_head.act(self.mlm_head.dense(z)))
            learned_logit_std = F.linear(
                h.float(), self.mlm_head._tied[0].weight.float()).std()
            bias_std = self.mlm_head.bias[:V - 3].std()

            td = tdat.unsqueeze(1) if tdat.dim() == 4 else tdat
            tgt_ratio = (td.reshape(td.shape[0], -1).std(0).mean()
                         / td.std().clamp(min=1e-8))
            rec_ratio = ((rec.reshape(rec.shape[0], -1).std(0).mean()
                          / rec.std().clamp(min=1e-8)) if recon_active else zero)
            lrs = self._group_lrs(opt_g, z.device)

            if self.id_gate is not None:
                gate_v = self.id_gate.detach().abs().reshape(())
                if latent_core is not None:
                    sk = latent - latent_core
                    skip_share = (sk.pow(2).mean()
                                  / latent.pow(2).mean().clamp(min=1e-12))
                else:
                    skip_share = zero
            else:
                gate_v = skip_share = zero

            if self.step % max(self.hparams.rank_log_every, 1) == 0:
                zr3 = z[:, 1:, :]
                self._rank_cache = {
                    'enc_eff_rank': effective_rank(zr3, mask=pad_mask,
                                                   center='within'),
                    'enc_eff_rank_global': effective_rank(zr3, mask=pad_mask,
                                                          center='global'),
                }
                if latent_core is not None:
                    self._rank_cache.update({
                        'lat_eff_rank': effective_rank(
                            latent_core, mask=pad_mask, center='within'),
                        'lat_eff_rank_global': effective_rank(
                            latent_core, mask=pad_mask, center='global'),
                    })
                    if latent is not latent_core:
                        self._rank_cache['lat_skip_eff_rank'] = effective_rank(
                            latent, mask=pad_mask, center='within')
            rank_metrics = getattr(self, '_rank_cache', {})

            masked_metrics = {}
            if recon_active and self.hparams.w_masked_recon > 0:
                masked_metrics['train_n_masked_res'] = torch.tensor(
                    float(n_masked_res), device=z.device)
            if rc_m is not None:
                local_metrics['inv_pattern_masked'] = rc_m['inv_pattern'].detach()
                local_metrics['recon_masked'] = rc_m['recon_total'].detach()

        # DDP: every rank must log the same synced keys on every step, or
        # NCCL hangs.  Conditional/optional keys go to local_metrics below.
        self.log_dict({
            **masked_metrics,
            **ibot_metrics,
            **gram_metrics,
            'stage_recon_active': torch.tensor(float(recon_active), device=z.device),
            'train_loss': total_g_loss,
            'train_raw_mlm_loss': raw_mlm_loss,
            'train_mlm_acc': mlm_acc,
            'train_n_masked': n_masked,
            **copy_metrics,
            'mlm_learned_logit_std': learned_logit_std,
            'mlm_bias_std': bias_std,
            'train_dino_loss': dino_loss,
            'id_gate': gate_v,
            'id_skip_share': skip_share,
            'pattern_loss': rc['pattern_loss'],
            'amp_loss': rc['amp_loss'],
            'occ_loss': rc['occ_loss'],
            'train_hgf_loss': rc['hgf_loss'],
            'train_wa_loss': rc['wa_loss'],
            'inv_pattern': rc['inv_pattern'],
            'inv_mean': rc['inv_mean'],
            'inv_energy': rc['inv_energy'],
            'inv_tv': rc['inv_tv'],
            'inv_cross': rc['inv_cross'],
            'inv_cross_mean': rc['inv_cross_mean'],
            'recon_total': rc['recon_total'],
            'rec_pattern_cos': rc['rec_pattern_cos'],
            'rec_blob_frac': rc['rec_blob_frac'],
            'tgt_blob_frac': rc['tgt_blob_frac'],
            'rec_radial_r2': rc['rec_radial_r2'],
            'tgt_radial_r2': rc['tgt_radial_r2'],
            'rec_rim_energy': rc['rec_rim_energy'],
            'tgt_rim_energy': rc['tgt_rim_energy'],
            'prototype_entropy': proto_ent,
            'prototype_marginal_entropy': self.dino.marginal_entropy(),
            'enc_between_ratio': enc_ratio,
            'collapse_loss': collapse_loss,
            **({'collapse_loss_latent': lat_collapse} if lmet else {}),
            **cmet,
            **lmet,
            **rank_metrics,
            'rec_between_ratio': rec_ratio,
            'tgt_between_ratio': tgt_ratio,
            'ratio_gap': tgt_ratio - rec_ratio,
            **lrs,
        }, sync_dist=True, batch_size=masked_seqs.shape[0])
        if local_metrics:
            self.log_dict(local_metrics, sync_dist=False,
                          batch_size=masked_seqs.shape[0])

        self.step += 1
        return total_g_loss

    # ------------------------------------------------------------- val

    def _stage1_val_mlm(self, seqs, batch_idx):
        """Masked-LM validation for stage 1.

        Masking uses a fixed seed per batch index so the same positions are
        held out every epoch.  Uses the same span scheme as training.
        """
        V = self.hparams.vocab_size
        body = seqs[:, 1:].clone()
        gen = torch.Generator().manual_seed(1234 + batch_idx)
        body, labels_body = span_mask_tokens(
            body, pad_id=PAD_TOKEN_ID, mask_id=MASK_TOKEN_ID,
            mask_rate=self.hparams.mlm_mask_rate,
            span_residues=self.hparams.mlm_span_residues,
            generator=gen, replace_frac=self.hparams.mlm_replace_frac,
            codec=self._codec())

        masked = torch.cat([seqs[:, :1], body], dim=1)
        bos_lbl = torch.full((seqs.shape[0], 1), -100,
                             dtype=torch.long, device=seqs.device)
        labels_flat = torch.cat([bos_lbl, labels_body], dim=1).view(-1)

        z = self.forward(masked, pred=True)
        logits = self.mlm_head(z)
        valid = labels_flat != -100
        if not valid.any():
            return None, None, None
        loss = F.cross_entropy(logits.view(-1, V).float(), labels_flat,
                               reduction='mean')
        acc = (logits.view(-1, V).argmax(-1)[valid]
               == labels_flat[valid]).float().mean()
        return loss, acc, L.between_ratio(z[:, 1:, :])

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return None
        V = self.hparams.vocab_size
        tdat, seqs, _ = batch
        seqs = self.preprocess_sequence(seqs)

        if not self._recon_active():
            loss, acc, ratio = self._stage1_val_mlm(seqs, batch_idx)
            if loss is None:
                return None
            self.log_dict({'val_stage1_mlm_loss': loss.detach(),
                           'val_stage1_mlm_acc': acc.detach(),
                           'val_enc_between_ratio': ratio.detach()},
                          sync_dist=True, batch_size=seqs.shape[0])
            self.log('val_loss', loss.detach(), prog_bar=True,
                     sync_dist=True, batch_size=seqs.shape[0])
            return loss

        z, latent_core, latent, rec = self._forward_full(seqs)
        rc = L.reconstruction_losses(rec, tdat, self.recon_w, None)

        # Masked-row and iBOT validation on a fixed-seed hidden set.
        v_masked = {}
        if self.hparams.lambda_gram > 0:
            with torch.no_grad():
                v_gl, v_gc = self._gram(z, seqs[:, 1:], tdat)
            v_masked = {'val_gram_loss': v_gl, 'val_gram_corr': v_gc}
        if self.hparams.w_masked_recon > 0 or self.ibot is not None:
            with torch.no_grad():
                body = seqs[:, 1:].clone()
                gen = torch.Generator().manual_seed(4321 + batch_idx)
                body, v_lab = span_mask_tokens(
                    body, pad_id=PAD_TOKEN_ID, mask_id=MASK_TOKEN_ID,
                    mask_rate=self.hparams.mlm_mask_rate,
                    span_residues=self.hparams.mlm_span_residues,
                    generator=gen, replace_frac=self.hparams.mlm_replace_frac,
                    codec=self._codec())
                v_rows = v_lab.reshape(-1) != -100
                hid_in = torch.cat([seqs[:, :1], body], dim=1)
                z_hid, _, _, rec_hid = self._forward_full(hid_in)
                if self.hparams.w_masked_recon > 0 and int(v_rows.sum()) >= 2:
                    hid = L.reconstruction_losses(
                        rec_hid[v_rows], tdat[v_rows], self.recon_w_masked,
                        None, with_diagnostics=False)
                    seen = L.reconstruction_losses(
                        rec[v_rows], tdat[v_rows], self.recon_w_masked,
                        None, with_diagnostics=False)
                    v_masked.update({
                        'val_inv_pattern_masked': hid['inv_pattern'],
                        'val_inv_pattern_masked_ref': seen['inv_pattern'],
                        'val_recon_masked': hid['recon_total']})
                if self.ibot is not None:
                    bos_l = torch.full((seqs.shape[0], 1), -100,
                                       dtype=torch.long, device=seqs.device)
                    v_sel = torch.cat([bos_l, v_lab], 1).reshape(-1) != -100
                    v_flat_t = seqs.reshape(-1)
                    v_real = ((v_flat_t != PAD_TOKEN_ID)
                              & (v_flat_t != BOS_TOKEN_ID))
                    band = self.hparams.ibot_teacher_band
                    v_t = (self._teacher_forward(seqs, band) if band > 0
                           else self.teacher_encoder(seqs))
                    v_ib, v_ibm = self.ibot(z_hid, v_t, v_sel, v_real,
                                            update_queue=False)
                    v_masked['val_ibot_loss'] = v_ib
                    v_masked['val_ibot_agree'] = v_ibm['ibot_agree']

        val_loss = rc['recon_total']
        if not torch.isfinite(val_loss):
            # Don't substitute a fake zero -- that's what caused ten epochs
            # of bad checkpoint selection last time.
            print(f"\n[WARNING] non-finite val_loss at batch {batch_idx}; "
                  f"terms: { {k: float(v) for k, v in rc.items()} }")
            torch.save({'seqs': seqs, 'tdat': tdat, 'rec': rec, 'recon': rc},
                       f'nan_dump_val_batch_{batch_idx}.pt')

        with torch.no_grad():
            # Copy test on unmasked input (the deployment condition).
            v_flat = seqs.reshape(-1)
            v_sel = ((v_flat != PAD_TOKEN_ID) & (v_flat != BOS_TOKEN_ID))
            v_logits = self.mlm_head(z).view(-1, V).float()
            _, v_copy_ce, v_n_copy, _ = self._copy_ce(v_logits, v_flat, v_sel)
            v_copy_metrics = {
                'val_n_copy': torch.tensor(float(v_n_copy), device=z.device)}
            if v_copy_ce is not None:
                v_copy_metrics['val_copy_ce'] = v_copy_ce
                v_copy_metrics['val_copy_ppl'] = torch.exp(
                    v_copy_ce.clamp(max=20.0))
                v_copy_metrics['val_copy_acc'] = (
                    v_logits[v_sel].argmax(-1) == v_flat[v_sel]).float().mean()

            rec_ratio = (rec.reshape(rec.shape[0], -1).std(0).mean()
                         / rec.std().clamp(min=1e-8))
            td = tdat.unsqueeze(1) if tdat.dim() == 4 else tdat
            tgt_ratio = (td.reshape(td.shape[0], -1).std(0).mean()
                         / td.std().clamp(min=1e-8))
            enc_ratio = L.between_ratio(z[:, 1:, :])
            v_pad = seqs[:, 1:] != PAD_TOKEN_ID
            zr3 = z[:, 1:, :]
            v_enc_rank = effective_rank(zr3, mask=v_pad, center='within')
            v_lat_rank = effective_rank(latent_core, mask=v_pad, center='within')
            v_enc_rank_g = effective_rank(zr3, mask=v_pad, center='global')
            v_lat_rank_g = effective_rank(latent_core, mask=v_pad, center='global')
            vkw = self._guard_kwargs(v_pad)
            _, v_cmet = collapse_penalty(zr3, prefix='enc', **vkw)
            _, v_lmet = collapse_penalty(latent_core, prefix='lat', **vkw)
            if latent is not latent_core:
                v_skip_share = (latent - latent_core).pow(2).mean() \
                    / latent.pow(2).mean().clamp(min=1e-12)
                v_lat_skip_rank = effective_rank(latent, mask=v_pad,
                                                 center='within')
            else:
                v_skip_share = torch.zeros((), device=z.device)
                v_lat_skip_rank = v_lat_rank

            v_cross_sep = {}
            by_sep = rc.get('inv_cross_by_sep')
            if by_sep is not None and self.hparams.w_cross > 0:
                for d, v in enumerate(by_sep.tolist(), start=1):
                    if v == v:
                        v_cross_sep[f'val_inv_cross_d{d}'] = torch.tensor(
                            v, device=z.device)

        self.log_dict({
            'val_pattern_loss': rc['pattern_loss'].detach(),
            'val_amp_loss': rc['amp_loss'].detach(),
            'val_occ_loss': rc['occ_loss'].detach(),
            'val_hgf_loss': rc['hgf_loss'].detach(),
            'val_wa_loss': rc['wa_loss'].detach(),
            'val_inv_pattern': rc['inv_pattern'].detach(),
            'val_inv_mean': rc['inv_mean'].detach(),
            'val_inv_energy': rc['inv_energy'].detach(),
            'val_inv_tv': rc['inv_tv'].detach(),
            'val_inv_cross': rc['inv_cross'].detach(),
            'val_inv_cross_mean': rc['inv_cross_mean'].detach(),
            **v_cross_sep,
            **v_masked,
            'val_pattern_cos': rc['rec_pattern_cos'].detach(),
            'val_rec_blob_frac': rc['rec_blob_frac'].detach(),
            'val_tgt_blob_frac': rc['tgt_blob_frac'].detach(),
            'val_rec_radial_r2': rc['rec_radial_r2'].detach(),
            'val_tgt_radial_r2': rc['tgt_radial_r2'].detach(),
            'val_rec_rim_energy': rc['rec_rim_energy'].detach(),
            'val_tgt_rim_energy': rc['tgt_rim_energy'].detach(),
            **v_copy_metrics,
            'val_id_gate': (self.id_gate.detach().abs().reshape(())
                            if self.id_gate is not None
                            else torch.zeros((), device=z.device)),
            'val_id_skip_share': v_skip_share,
            'val_enc_between_ratio': enc_ratio,
            'val_enc_eff_rank': v_enc_rank,
            'val_lat_eff_rank': v_lat_rank,
            'val_lat_skip_eff_rank': v_lat_skip_rank,
            'val_enc_eff_rank_global': v_enc_rank_g,
            'val_lat_eff_rank_global': v_lat_rank_g,
            'val_enc_scale': v_cmet['enc_scale'],
            'val_enc_spread': v_cmet['enc_spread'],
            'val_enc_share': v_cmet['enc_share'],
            'val_enc_pr': v_cmet['enc_pr'],
            'val_lat_scale': v_lmet['lat_scale'],
            'val_lat_spread': v_lmet['lat_spread'],
            'val_lat_share': v_lmet['lat_share'],
            'val_lat_pr': v_lmet['lat_pr'],
            'val_rec_between_ratio': rec_ratio,
            'val_ratio_gap': tgt_ratio - rec_ratio,
        }, sync_dist=True, prog_bar=False, batch_size=seqs.shape[0])
        self.log('val_loss', val_loss.detach(), prog_bar=True,
                 sync_dist=True, batch_size=seqs.shape[0])
        return val_loss

    def test_step(self, batch, batch_idx):
        tdat, seqs, tmsk = batch
        seqs = self.preprocess_sequence(seqs)
        z, rec = self(seqs)
        return {'embeddings': z.detach(), 'sequences': seqs,
                'reconstruction': rec.detach()}

    def predict_step(self, batch, batch_idx):
        return self(self.preprocess_sequence(batch), True)

    # ------------------------------------------------------------- optim

    def configure_optimizers(self):
        """Set up AdamW with four parameter groups and a warmup+cosine schedule.

        Groups: encoder decay, encoder no-decay, decoder decay, decoder
        no-decay.  The decoder (self.b, self.i, identity skip) trains at
        ``learning_rate * decoder_lr_factor``.
        """
        teacher = ['teacher_encoder', 'teacher_dino_head', 'ibot.teacher_head']
        NO_DECAY_NAMES = ('embedding', 'tokens', 'registers',
                          'pos_bias', 'to_query', 'cond_proj', 'cond_fuse',
                          'id_proj', 'id_gate')
        DECODER_PREFIXES = ('b.', 'i.', 'id_proj', 'id_gate')

        buckets = {'enc_decay': [], 'enc_no_decay': [],
                   'dec_decay': [], 'dec_no_decay': []}
        for n, p in self.named_parameters():
            if any(t in n for t in teacher) or not p.requires_grad:
                continue
            head = 'dec' if n.startswith(DECODER_PREFIXES) else 'enc'
            tail = ('_no_decay' if (p.ndim <= 1
                                    or any(t in n for t in NO_DECAY_NAMES))
                    else '_decay')
            buckets[head + tail].append(p)

        base = self.hparams.learning_rate
        dec_lr = base * self.hparams.decoder_lr_factor
        wd = self.hparams.weight_decay
        spec = [('enc_decay', base, wd), ('enc_no_decay', base, 0.0),
                ('dec_decay', dec_lr, wd), ('dec_no_decay', dec_lr, 0.0)]
        spec = [(k, lr, w) for k, lr, w in spec if buckets[k]]
        if not any(k.startswith('dec') for k, _, _ in spec):
            print("[optim] WARNING no parameters matched the decoder prefixes "
                  f"{DECODER_PREFIXES}; decoder_lr_factor is doing nothing. "
                  "Check the attribute names of self.b / self.i.")

        self._lr_group_index = {k: i for i, (k, _, _) in enumerate(spec)}

        print(f"[optim] encoder lr {base:.2e}   decoder lr {dec_lr:.2e} "
              f"({self.hparams.decoder_lr_factor:g}x)")
        for k, lr, w in spec:
            n_t = len(buckets[k])
            n_p = sum(q.numel() for q in buckets[k])
            print(f"[optim]   {k:<13} {n_t:>3} tensors  {n_p / 1e6:>6.2f}M  "
                  f"lr {lr:.2e}  wd {w}")

        if self.hparams.optimizer_type.lower() != "adamw":
            print("[optim] WARNING optimizer_type is not adamw. Classic Adam "
                  "folds weight decay into the gradient, and Adam's "
                  "normalisation then decays low-gradient parameters at ~lr "
                  "per step. That is what drove context_to_query.weight to "
                  "std 1.4e-05, 5300x below its init.")
        Opt = torch.optim.AdamW if self.hparams.optimizer_type.lower() == "adamw" \
            else torch.optim.Adam
        opt_g = Opt([{'params': buckets[k], 'lr': lr, 'weight_decay': w}
                     for k, lr, w in spec],
                    lr=base,
                    betas=(self.hparams.adam_beta1, self.hparams.adam_beta2))

        total = self.hparams.cyclic_gamma_steps
        if self.hparams.cyclic_lr:
            if total:
                gamma = math.exp(-1.0 / total)
            else:
                gamma = 0.9999
                print("[optim] WARNING cyclic_lr with no cyclic_gamma_steps: "
                      "falling back to gamma=0.9999, whose cycle amplitude "
                      "decays to ~3e-6 by step 127k. Pass the run's total "
                      "optimizer-step count to size it properly.")
            print(f"[optim] CyclicLR exp_range gamma {gamma:.8f} "
                  f"(amplitude e^-1 at step {total or 10000})")
            sched = torch.optim.lr_scheduler.CyclicLR(
                opt_g, step_size_up=self.hparams.step_size_up,
                base_lr=[lr for _, lr, _ in spec],
                max_lr=[lr * self.hparams.max_lr_factor for _, lr, _ in spec],
                mode='exp_range', gamma=gamma, cycle_momentum=False)
            return [opt_g], [{"scheduler": sched, "interval": "step"}]

        # Default: linear warmup then cosine to min_lr_factor.
        warm = max(int(self.hparams.warmup_steps), 1)
        horizon = total if total else 200_000
        if not total:
            print(f"[optim] WARNING no cyclic_gamma_steps given; cosine "
                  f"horizon defaults to {horizon} optimizer steps. Pass the "
                  f"run's real step count so the decay lands where you want.")
        floor_f = self.hparams.min_lr_factor

        def lr_lambda(step):
            if step < warm:
                return (step + 1) / warm
            t = min(1.0, (step - warm) / max(horizon - warm, 1))
            return floor_f + (1.0 - floor_f) * 0.5 * (1.0 + math.cos(math.pi * t))

        print(f"[optim] warmup {warm} steps, cosine over {horizon} steps "
              f"to {floor_f:g}x base")
        sched = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda)
        return [opt_g], [{"scheduler": sched, "interval": "step"}]
