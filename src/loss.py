"""Loss terms for BioT.
Run ``audit_objective.py`` after any change here.
"""

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# blocks is imported lazily (inside reconstruction_losses) to avoid pulling
# the full SE3 convolution stack just for wa_loss, which defaults to weight 0.


# ----------------------------------------------------------------- helpers

def between_ratio(t):
    """Ratio of across-position std to overall std, for ``(B, L, C)`` tensors.

    1.0 = every position differs as much as the tensor varies overall;
    0.0 = every position carries the same value.
    """
    flat = t.reshape(-1, t.shape[-1]) if t.dim() > 2 else t
    if flat.shape[0] < 2:
        return flat.new_zeros(())
    return flat.std(0).mean() / flat.std().clamp(min=1e-8)


def soft_occupancy(x, scale):
    """Occupancy mask in ``[0, 1)``.  *scale* is set relative to the field's
    own spread by ``occupancy_loss``."""
    return torch.tanh(torch.relu(x) / scale)


def _match_shapes(rec, tdat, tmsk=None):
    """Bring rec/target to ``(B, 1, D, H, W)`` float and apply an optional mask."""
    rec_f = rec.float()
    tdat_f = tdat.unsqueeze(1).float() if tdat.dim() == 4 else tdat.float()
    if tmsk is not None:
        mask = tmsk.unsqueeze(1).float() if tmsk.dim() == 4 else tmsk.float()
        rec_f, tdat_f = rec_f * mask, tdat_f * mask
    return rec_f, tdat_f


def _overlap(a, b, eps=1e-12):
    """Bounded overlap: ``1 - 2<a,b> / (||a||^2 + ||b||^2)``.

    0 iff ``a == b``, bounded in ``[0, 2]``, stationary at ``a == b``,
    and finite gradient as ``||a|| -> 0``.

    Classic soft Dice (L1 denominator) is *not* stationary at ``a == b``
    for non-binary masks -- it carries a standing gradient that inflates
    whatever it scores.
    """
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    num = 2.0 * (a * b).sum(1)
    den = (a * a).sum(1) + (b * b).sum(1) + eps
    return 1.0 - num / den


# ------------------------------------------------------- reconstruction

@dataclass
class ReconWeights:
    """Reconstruction term weights.  All terms are O(1).

    ``pattern`` is the headline term: it scores residue-centred fields, so
    any residue-independent output (sphere, blob) scores exactly 1.0.
    The others calibrate absolute scale, support, and detail.
    """
    pattern: float = 1.0
    amplitude: float = 0.5
    occupancy: float = 0.5
    highfreq: float = 0.1
    wasserstein: float = 0.0
    # Rotation-invariant terms (invariant_loss.py).  The voxel-wise terms
    # above compare in the target's frame; under random orientation each
    # has the sphere as its optimum.  These compare frame-free two-point
    # statistics instead.  To switch: set voxel-wise weights to 0,
    # invariant to 1.0.
    invariant: float = 0.0
    inv_mean: float = 0.5
    inv_energy: float = 0.5
    inv_tv: float = 0.25
    inv_l_max: int = 6
    inv_shells: int = 12
    # Cross-residue terms (invariant_loss.cross_residue_losses).
    # OFF by default: requires absolute orientation in the protein frame.
    cross: float = 0.0
    cross_mean: float = 0.5
    cross_window: int = 10
    # Restrict to the inscribed sphere (the region the descriptor covers).
    inv_mask_sphere: bool = True
    # Occupancy scale as a multiple of the target's own spread.
    occ_rel: float = 3.0
    pyramid_levels: int = 4

    def as_dict(self):
        return asdict(self)


def pattern_loss(rec, tgt, eps=1e-12):
    """Residue-specific pattern match -- the headline term.

    Both fields are centred on the batch mean first, so this scores only
    what distinguishes one residue from another.  A residue-independent
    output scores exactly 1.0 (no calibration needed); below 1.0 means
    the decoder is producing useful residue-specific structure.
    """
    if rec.shape[0] < 2:
        return rec.new_zeros(())
    a = rec - rec.mean(0, keepdim=True)
    b = tgt - tgt.mean(0, keepdim=True)
    return _overlap(a, b, eps).mean()


def amplitude_loss(rec, tgt, levels=4, eps=1e-12):
    """Multi-scale overlap on raw (uncentred) fields.

    Weights ``[2^(levels-1), ..., 1]``: full resolution weighted most.
    """
    w = [2 ** (levels - 1 - i) for i in range(levels)]
    tot, r, t = rec.new_zeros(()), rec, tgt
    for i in range(levels):
        tot = tot + w[i] * _overlap(r, t, eps).mean()
        if i < levels - 1:
            r, t = F.avg_pool3d(r, 2), F.avg_pool3d(t, 2)
    return tot / sum(w)


def occupancy_loss(rec, tgt, occ_rel=3.0, eps=1e-12):
    """Sign-symmetric overlap of soft occupancy masks at the feature scale.

    The scale is ``occ_rel`` x the target's own ``mean|x - median|``
    (detached), so it tracks the data rather than a hard-coded constant.
    Both signs are scored: the positive lobe alone is minimised by a
    positive blob everywhere.
    """
    with torch.no_grad():
        flat = tgt.reshape(tgt.shape[0], -1)
        med = flat.median(dim=1).values.view(-1, *([1] * (tgt.dim() - 1)))
        scale = (tgt - med).abs().mean().clamp(min=1e-20) * max(occ_rel, 1e-6)
    rc, tc = rec - med, tgt - med
    pos = _overlap(soft_occupancy(rc, scale), soft_occupancy(tc, scale), eps)
    neg = _overlap(soft_occupancy(-rc, scale), soft_occupancy(-tc, scale), eps)
    return 0.5 * (pos + neg).mean()


def highfreq_loss(rec, tgt, k=3, eps=1e-12):
    """Overlap of the high-pass content -- the term that asks for detail."""
    def hp(v):
        return v - F.avg_pool3d(v, k, stride=1, padding=k // 2)
    return _overlap(hp(rec), hp(tgt), eps).mean()


_SHELL_CACHE = {}


def _shell_index(grid, device):
    """Integer-radius shell per voxel, cached per ``(grid, device)``."""
    key = (grid, str(device))
    if key not in _SHELL_CACHE:
        import numpy as np
        idx = np.indices((grid,) * 3).reshape(3, -1).T - (grid - 1) / 2.0
        r = np.rint(np.linalg.norm(idx, axis=1)).astype(np.int64)
        _SHELL_CACHE[key] = (torch.tensor(r, device=device), int(r.max()) + 1)
    return _SHELL_CACHE[key]


@torch.no_grad()
def radial_r2(x):
    """Fraction of variance explained by radius alone (mean over batch).

    1.0 = a perfect sphere.  Frame-free, so this is the sphere monitor
    under the rotation-invariant objective (where ``pattern_loss`` stays
    near 1.0 for a correct shape in the model's own frame).
    """
    v = x.reshape(x.shape[0], -1).float()
    sh, n_sh = _shell_index(x.shape[-1], x.device)
    N = v.shape[0]
    sums = torch.zeros(N, n_sh, device=v.device).index_add_(1, sh, v)
    cnt = torch.zeros(n_sh, device=v.device).index_add_(
        0, sh, torch.ones_like(sh, dtype=torch.float32))
    pred = (sums / cnt.clamp(min=1))[:, sh]
    ss_res = (v - pred).pow(2).sum(1)
    ss_tot = (v - v.mean(1, keepdim=True)).pow(2).sum(1).clamp(min=1e-30)
    return (1.0 - ss_res / ss_tot).mean()


@torch.no_grad()
def rim_energy(x, r_frac=2.0 / 3.0):
    """Fraction of field energy beyond ``r_frac`` of the half-width.

    Frame-free.  Rim noise lowers both ``radial_r2`` and blob fraction
    (reads as "structure" on both), so this is the monitor that catches it.
    """
    v = x.reshape(x.shape[0], -1).float().pow(2)
    sh, n_sh = _shell_index(x.shape[-1], x.device)
    rim = (sh.float() >= r_frac * (x.shape[-1] - 1) / 2.0).float()
    return ((v * rim).sum(1) / v.sum(1).clamp(min=1e-30)).mean()


@torch.no_grad()
def recon_diagnostics(rec, tgt):
    """Logging-only diagnostics (not backwarded)."""
    r = rec.reshape(rec.shape[0], -1)
    t = tgt.reshape(tgt.shape[0], -1)
    rc, tc = r - r.mean(0, keepdim=True), t - t.mean(0, keepdim=True)
    cos = F.cosine_similarity(rc, tc, dim=1, eps=1e-12).mean()
    blob_r = (r.mean(0).pow(2).sum() * r.shape[0]
              / r.pow(2).sum().clamp(min=1e-20))
    blob_t = (t.mean(0).pow(2).sum() * t.shape[0]
              / t.pow(2).sum().clamp(min=1e-20))
    return {'rec_pattern_cos': cos,
            'rec_blob_frac': blob_r, 'tgt_blob_frac': blob_t,
            'rec_radial_r2': radial_r2(rec), 'tgt_radial_r2': radial_r2(tgt),
            'rec_rim_energy': rim_energy(rec), 'tgt_rim_energy': rim_energy(tgt),
            'rec_rms': r.std(), 'tgt_rms': t.std()}


_DESC_CACHE = {}
_MASK_CACHE = {}


def sphere_mask(grid, device, r_max=None):
    """``(1, 1, G, G, G)`` float mask of the inscribed sphere, cached.

    Use for visualisation too: outside it the decoder is unconstrained.
    """
    key = (grid, str(device), r_max)
    if key not in _MASK_CACHE:
        import numpy as np
        c = (grid - 1) / 2.0
        idx = np.indices((grid,) * 3).astype(np.float64) - c
        r = np.sqrt((idx ** 2).sum(0))
        m = (r <= (r_max or c) + 0.5).astype(np.float32)
        _MASK_CACHE[key] = torch.tensor(m, device=device).view(1, 1, *m.shape)
    return _MASK_CACHE[key]


def _descriptor(grid, l_max, n_shells, device):
    """Cached ``SHDescriptor`` (~44 MB at defaults, built once)."""
    key = (grid, l_max, n_shells, str(device))
    if key not in _DESC_CACHE:
        from invariant_loss import SHDescriptor
        _DESC_CACHE[key] = SHDescriptor(grid=grid, l_max=l_max,
                                        n_shells=n_shells).to(device)
    return _DESC_CACHE[key]


def reconstruction_losses(rec, tdat, w: ReconWeights, tmsk=None,
                          with_diagnostics=True):
    """Compute the full reconstruction objective.

    Returns a dict of every term plus ``recon_total`` (the weighted sum
    and the only entry that should be backwarded).  Voxel-wise terms are
    always computed as diagnostics regardless of their weight.
    """
    with torch.amp.autocast('cuda', enabled=False):
        if (w.invariant > 0 or w.cross > 0) and w.inv_mask_sphere:
            sm = sphere_mask(rec.shape[-1], rec.device)
            if tmsk is None:
                tmsk = sm
            else:
                tmsk = (tmsk.unsqueeze(1) if tmsk.dim() == 4 else tmsk).float() * sm
        r, t = _match_shapes(rec, tdat, tmsk)
        out = {
            'pattern_loss': pattern_loss(r, t),
            'amp_loss': amplitude_loss(r, t, w.pyramid_levels),
            'occ_loss': occupancy_loss(r, t, w.occ_rel),
            'hgf_loss': highfreq_loss(r, t),
        }
        if w.wasserstein > 0:
            from blocks import wa_loss
            out['wa_loss'] = wa_loss(r, t, denorm=False)
        else:
            out['wa_loss'] = r.new_zeros(())
        if w.invariant > 0 and r.shape[0] >= 2:
            from invariant_loss import invariant_losses
            d = _descriptor(r.shape[-1], w.inv_l_max, w.inv_shells, r.device)
            inv = invariant_losses(r, t, d, w_pattern=1.0, w_mean=w.inv_mean,
                                   w_energy=w.inv_energy, w_tv=w.inv_tv)
            out['inv_pattern'] = inv['inv_pattern']
            out['inv_mean'] = inv['inv_mean']
            out['inv_energy'] = inv['inv_energy']
            out['inv_tv'] = inv['inv_tv']
        else:
            out['inv_pattern'] = r.new_zeros(())
            out['inv_mean'] = r.new_zeros(())
            out['inv_energy'] = r.new_zeros(())
            out['inv_tv'] = r.new_zeros(())
        if w.cross > 0 and r.shape[0] >= 3:
            from invariant_loss import cross_residue_losses
            d = _descriptor(r.shape[-1], w.inv_l_max, w.inv_shells, r.device)
            cx = cross_residue_losses(r, t, d, window=w.cross_window)
            out['inv_cross'] = cx['inv_cross']
            out['inv_cross_mean'] = cx['inv_cross_mean']
            out['inv_cross_by_sep'] = cx['inv_cross_by_sep']
        else:
            out['inv_cross'] = r.new_zeros(())
            out['inv_cross_mean'] = r.new_zeros(())
            out['inv_cross_by_sep'] = r.new_full((max(w.cross_window, 1),),
                                                 float('nan'))
        out['recon_total'] = (w.pattern * out['pattern_loss']
                              + w.amplitude * out['amp_loss']
                              + w.occupancy * out['occ_loss']
                              + w.highfreq * out['hgf_loss']
                              + w.wasserstein * out['wa_loss']
                              + w.invariant * (out['inv_pattern']
                                               + w.inv_mean * out['inv_mean']
                                               + w.inv_energy * out['inv_energy']
                                               + w.inv_tv * out['inv_tv'])
                              + w.cross * (out['inv_cross']
                                           + w.cross_mean * out['inv_cross_mean']))
        if with_diagnostics:
            out.update(recon_diagnostics(r, t))
        return out


# ------------------------------------------------- gram anchoring

def descriptor_rows(fields, desc):
    """Compute rotation-invariant descriptors for one protein's residues.

    Returns ``(N, F)`` with each degree block scaled to unit RMS over the
    protein so ``l=0`` (12 values) isn't drowned by ``l=1..6`` (144 each).
    """
    d = desc(fields)
    out = []
    for l in sorted(d):
        b = d[l].reshape(fields.shape[0], -1).float()
        out.append(b / b.pow(2).mean().sqrt().clamp(min=1e-12))
    return torch.cat(out, 1)


def gram_losses(z, rows):
    """Gram anchoring of z to the target descriptors within one protein.

    Matches the residue-to-residue similarity matrix of z to that of the
    target descriptors.  Both sides are centred and L2-normalised per
    residue, so a protein-wide constant contributes nothing on either side.

    Parameters
    ----------
    z : Tensor, shape ``(N, C)``
    rows : Tensor, shape ``(N, F)``
        From ``descriptor_rows``.  N >= 3.

    Returns
    -------
    loss : Tensor
        Mean squared difference of off-diagonal cosines.
    corr : Tensor
        Pearson correlation (detached); 0 = unrelated, 1 = identical.
    """
    N = z.shape[0]
    zc = F.normalize(z.float() - z.float().mean(0, keepdim=True), dim=-1)
    tc = F.normalize(rows - rows.mean(0, keepdim=True), dim=-1)
    off = ~torch.eye(N, dtype=torch.bool, device=z.device)
    gs, gt = (zc @ zc.T)[off], (tc @ tc.T)[off]
    loss = (gs - gt).pow(2).mean()
    with torch.no_grad():
        a, b = gs - gs.mean(), gt - gt.mean()
        corr = (a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-12)
    return loss, corr


def zero_recon(device, with_diagnostics=True):
    """Return a dict of zero tensors matching ``reconstruction_losses`` keys."""
    z = torch.zeros((), device=device)
    out = {k: z for k in ('pattern_loss', 'amp_loss', 'occ_loss',
                          'hgf_loss', 'wa_loss', 'inv_pattern', 'inv_mean',
                          'inv_energy', 'inv_tv', 'inv_cross',
                          'inv_cross_mean', 'recon_total')}
    if with_diagnostics:
        out.update({k: z for k in ('rec_pattern_cos', 'rec_blob_frac',
                                   'tgt_blob_frac', 'rec_radial_r2',
                                   'tgt_radial_r2', 'rec_rim_energy',
                                   'tgt_rim_energy', 'rec_rms', 'tgt_rms')})
    return out


# -------------------------------------------------------------------- DINO

class DinoLoss(nn.Module):
    """BOS-level self-distillation (DINO).

    Cross-entropy to a Sinkhorn-balanced momentum teacher, plus me_max
    and KoLeo regularisers.  Owns its prototypes and both queues; the
    encoder stays on the LightningModule, which passes ``teacher_z`` in.
    """

    def __init__(self, latent_dim=64, num_prototypes=128, queue_length=4096,
                 student_temp=0.15, teacher_temp=0.05,
                 me_max_over_ema=True, me_max_weight=0.5, koleo_weight=0.1):
        super().__init__()
        self.student_dino_head = nn.Linear(latent_dim, num_prototypes)
        self.teacher_dino_head = nn.Linear(latent_dim, num_prototypes)
        self.teacher_dino_head.load_state_dict(self.student_dino_head.state_dict())
        self.teacher_dino_head.requires_grad_(False)

        self.num_prototypes = num_prototypes
        self.queue_length = queue_length
        self.student_queue_size = queue_length
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.me_max_over_ema = me_max_over_ema
        self.me_max_weight = me_max_weight
        self.koleo_weight = koleo_weight

        self.register_buffer("queue", torch.zeros(queue_length, latent_dim))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("student_queue", torch.zeros(queue_length, latent_dim))
        self.register_buffer("student_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("student_prob_ema",
                             torch.ones(num_prototypes) / num_prototypes)

    @torch.no_grad()
    def _enqueue_teacher(self, keys):
        bs = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + bs > self.queue_length:
            ptr = 0
        self.queue[ptr:ptr + bs] = keys
        self.queue_ptr[0] = (ptr + bs) % self.queue_length

    @torch.no_grad()
    def _enqueue_student(self, features):
        ptr = int(self.student_queue_ptr)
        for i in range(features.shape[0]):
            self.student_queue[(ptr + i) % self.student_queue_size] = features[i].detach()
        self.student_queue_ptr[0] = ptr + features.shape[0]

    @torch.no_grad()
    def update_teacher(self, momentum):
        """EMA update of the teacher head."""
        for sp, tp in zip(self.student_dino_head.parameters(),
                          self.teacher_dino_head.parameters()):
            tp.data.mul_(momentum).add_(sp.data, alpha=1 - momentum)

    def sinkhorn_knopp(self, teacher_output, n_iter=3):
        """Balanced assignment via Sinkhorn-Knopp normalisation."""
        logits = teacher_output.float() / self.teacher_temp
        logits = logits - logits.max(dim=1, keepdim=True)[0]
        Q = torch.exp(logits.clamp(min=-20.0, max=20.0))
        B, K = Q.shape
        Q = Q / (Q.sum() + 1e-8)
        for _ in range(n_iter):
            Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-8)
            Q = Q / K
            Q = Q / (Q.sum(dim=0, keepdim=True) + 1e-8)
            Q = Q / B
        return Q * B

    def _koleo(self, current_bos):
        """KoLeo regulariser: current batch vs history (rectangular cdist).

        A square cdist over ``[current; history]`` would need
        ``fill_diagonal_`` (in-place on an autograd tensor) and its
        zero-diagonal yields NaN in the backward.
        """
        ptr = int(self.student_queue_ptr)
        if ptr == 0:
            return torch.zeros((), device=current_bos.device)
        valid = min(ptr, self.student_queue_size)
        cur = F.normalize(current_bos.float(), dim=-1, p=2)
        hist = F.normalize(self.student_queue[:valid].detach().float(), dim=-1, p=2)
        d = torch.cdist(cur, hist)
        return -torch.log(d.min(dim=1).values.clamp(min=1e-4)).mean()

    def forward(self, student_z, teacher_z):
        student_bos = F.layer_norm(student_z[:, 0, :], (student_z.size(-1),))
        student_logits = self.student_dino_head(student_bos) / self.student_temp
        student_log_probs = F.log_softmax(student_logits, dim=-1)

        with torch.no_grad():
            teacher_bos = F.layer_norm(teacher_z, (teacher_z.size(-1),))[:, 0, :]
            ptr = int(self.queue_ptr)
            combined = (torch.cat([teacher_bos, self.queue[:ptr]], dim=0)
                        if ptr > 0 else teacher_bos)
            probs_all = self.sinkhorn_knopp(self.teacher_dino_head(combined))
            teacher_probs = probs_all[:teacher_bos.shape[0]].clamp(min=1e-7)
            teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True)
            self._enqueue_teacher(teacher_bos.detach())

        ce_loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()

        batch_avg = student_log_probs.exp().mean(dim=0)
        if self.me_max_over_ema:
            # Spread assignments across the running distribution, not just
            # one sample.  Snapshot before in-place update so the graph
            # never references mutated storage.
            avg_probs = 0.99 * self.student_prob_ema.clone() + 0.01 * batch_avg
            with torch.no_grad():
                self.student_prob_ema.mul_(0.99).add_(batch_avg.detach(), alpha=0.01)
        else:
            avg_probs = batch_avg
        me_max_loss = (avg_probs * torch.log(avg_probs + 1e-8)).sum()

        koleo_loss = self._koleo(student_bos)
        self._enqueue_student(student_bos)

        loss = (ce_loss + self.me_max_weight * me_max_loss
                + self.koleo_weight * koleo_loss)
        if not torch.isfinite(loss):
            return torch.zeros((), device=student_z.device, requires_grad=True)
        return loss

    @torch.no_grad()
    def prototype_entropy(self, bos):
        """Per-sample assignment entropy on the trained path."""
        x = F.layer_norm(bos, (bos.size(-1),))
        p = F.softmax(self.student_dino_head(x) / self.student_temp, dim=-1)
        return -(p * torch.log(p + 1e-8)).sum(dim=-1).mean()

    @torch.no_grad()
    def marginal_entropy(self):
        """Entropy of the running assignment distribution across proteins.

        Low per-sample + high marginal = healthy.
        Low on both = prototype collapse.
        """
        q = self.student_prob_ema
        return -(q * torch.log(q + 1e-8)).sum()


class ResidueDistillLoss(nn.Module):
    """Per-residue self-distillation at masked/corrupted positions (iBOT).

    The teacher encodes the full sequence; a per-residue head plus
    Sinkhorn-KN turns each teacher feature into a balanced assignment
    over ``num_prototypes``.  The student predicts that assignment at
    every position it didn't see correctly -- MLM with a learned semantic
    vocabulary where prototypes group residues by contextual role.

    Sinkhorn balances over the current protein's selected residues plus
    a cross-protein queue.

    Parameters
    ----------
    latent_dim : int
    num_prototypes : int
    queue_length : int
    student_temp, teacher_temp : float
    enqueue_per_protein : int
        Max residues enqueued per protein (random subset if exceeded).
    """

    def __init__(self, latent_dim=64, num_prototypes=256, queue_length=8192,
                 student_temp=0.1, teacher_temp=0.05, enqueue_per_protein=256):
        super().__init__()
        self.student_head = nn.Linear(latent_dim, num_prototypes)
        self.teacher_head = nn.Linear(latent_dim, num_prototypes)
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        self.teacher_head.requires_grad_(False)
        self.num_prototypes = num_prototypes
        self.queue_length = queue_length
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.enqueue_per_protein = enqueue_per_protein
        self.register_buffer("queue", torch.zeros(queue_length, latent_dim))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_fill", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update_teacher(self, momentum):
        """EMA update of the teacher head."""
        for sp, tp in zip(self.student_head.parameters(),
                          self.teacher_head.parameters()):
            tp.data.mul_(momentum).add_(sp.data, alpha=1 - momentum)

    @torch.no_grad()
    def _enqueue(self, feats):
        n = min(feats.shape[0], self.queue_length)
        if n == 0:
            return
        if feats.shape[0] > n:
            feats = feats[torch.randperm(feats.shape[0], device=feats.device)[:n]]
        ptr = int(self.queue_ptr)
        idx = (torch.arange(n, device=feats.device) + ptr) % self.queue_length
        self.queue[idx] = feats
        self.queue_ptr[0] = (ptr + n) % self.queue_length
        self.queue_fill[0] = min(int(self.queue_fill) + n, self.queue_length)

    @staticmethod
    def _sinkhorn(logits, n_iter=3):
        logits = logits - logits.max(dim=1, keepdim=True)[0]
        Q = torch.exp(logits.clamp(min=-20.0, max=20.0))
        B, K = Q.shape
        Q = Q / (Q.sum() + 1e-8)
        for _ in range(n_iter):
            Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-8)
            Q = Q / K
            Q = Q / (Q.sum(dim=0, keepdim=True) + 1e-8)
            Q = Q / B
        return Q * B

    def forward(self, student_z, teacher_z, select, real, update_queue=True):
        """Compute the iBOT loss at selected positions.

        Parameters
        ----------
        student_z, teacher_z : Tensor, ``(B, 1+L, C)``
            Encoder outputs with BOS at position 0.
        select : Tensor, ``(B*(1+L),)`` bool
            Positions to train (masked and corrupted body positions).
        real : Tensor, ``(B*(1+L),)`` bool
            Non-PAD, non-BOS positions (the queue feed pool).
        update_queue : bool
            False for validation (don't contaminate the queue).

        Returns
        -------
        loss : Tensor
        metrics : dict
            ``ibot_agree``, ``ibot_student_entropy``, ``ibot_teacher_entropy``
            (all detached).
        """
        C = student_z.shape[-1]
        dev = student_z.device
        if not bool(select.any()):
            z = torch.zeros((), device=dev)
            return z, {'ibot_agree': z, 'ibot_student_entropy': z,
                       'ibot_teacher_entropy': z}
        s = F.layer_norm(student_z.reshape(-1, C)[select].float(), (C,))
        with torch.no_grad():
            t_all = F.layer_norm(teacher_z.reshape(-1, C).float(), (C,))
            t_sel = t_all[select]
            fill = int(self.queue_fill)
            combined = (torch.cat([t_sel, self.queue[:fill]], 0) if fill > 0
                        else t_sel)
            q_all = self._sinkhorn(self.teacher_head(combined) / self.teacher_temp)
            q = q_all[:t_sel.shape[0]].clamp(min=1e-7)
            q = q / q.sum(dim=-1, keepdim=True)
            if update_queue:
                pool = t_all[real]
                k = min(self.enqueue_per_protein, pool.shape[0])
                if k < pool.shape[0]:
                    pool = pool[torch.randperm(pool.shape[0], device=dev)[:k]]
                self._enqueue(pool)
        log_p = F.log_softmax(self.student_head(s) / self.student_temp, dim=-1)
        loss = -(q * log_p).sum(dim=-1).mean()
        with torch.no_grad():
            p = log_p.exp()
            q_bar = q.mean(0)
            metrics = {
                'ibot_agree': (log_p.argmax(-1) == q.argmax(-1)).float().mean(),
                'ibot_student_entropy': -(p * log_p).sum(-1).mean(),
                'ibot_teacher_entropy': -(q_bar * torch.log(q_bar + 1e-8)).sum(),
            }
        if not torch.isfinite(loss):
            loss = torch.zeros((), device=dev, requires_grad=True)
        return loss, metrics


# ------------------------------------------------- old-checkpoint key remap

LEGACY_KEY_MAP = {
    "queue": "dino.queue",
    "queue_ptr": "dino.queue_ptr",
    "student_queue": "dino.student_queue",
    "student_queue_ptr": "dino.student_queue_ptr",
    "student_prob_ema": "dino.student_prob_ema",
    "student_dino_head.weight": "dino.student_dino_head.weight",
    "student_dino_head.bias": "dino.student_dino_head.bias",
    "teacher_dino_head.weight": "dino.teacher_dino_head.weight",
    "teacher_dino_head.bias": "dino.teacher_dino_head.bias",
    "loss_log_vars": "adaptive.loss_log_vars",
    "log_var_min": "adaptive.log_var_min",
}

DROP_PREFIXES = ("discriminator.", "adaptive.")


def remap_legacy_state_dict(sd):
    """Rename pre-split keys and drop removed modules.

    Returns ``(new_sd, n_renamed, n_dropped)``.
    """
    out, n = {}, 0
    for k, v in sd.items():
        if k in LEGACY_KEY_MAP:
            out[LEGACY_KEY_MAP[k]] = v
            n += 1
        else:
            out[k] = v
    dropped = [k for k in out if k.startswith(DROP_PREFIXES)]
    for k in dropped:
        del out[k]
    return out, n, len(dropped)
