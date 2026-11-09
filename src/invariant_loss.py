"""Rotation-invariant reconstruction loss.

Compares rec and target through their spherical-harmonic cross-shell power
spectra -- statistics that don't change when either field is rotated about
the residue centre.  A correct shape in any frame scores ~0; a sphere
scores the maximum.

The descriptor covers the inscribed sphere (``r <= 16.5``, 18,853 of
35,937 voxels).  The corners are invisible here, so
``ReconWeights.inv_mask_sphere`` masks them when this objective is on.

Self-test::

    python invariant_loss.py            # full test suite
    python invariant_loss.py --cross    # cross-residue term only
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _overlap(a, b, eps=1e-24):
    """``1 - 2<a,b>/(|a|^2+|b|^2)`` per sample.  0 iff ``a == b``, bounded ``[0, 2]``.

    eps is 1e-24 (not loss.py's 1e-12) because descriptors are quadratic
    in a field at ~1e-2, so ``|a|^2`` here is ~1e-9 and a larger eps would
    floor a perfect match at ~5e-4.
    """
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    num = 2.0 * (a * b).sum(1)
    den = (a * a).sum(1) + (b * b).sum(1)
    return 1.0 - num / (den + eps)


def real_sph_harm(l_max, theta, phi):
    """Real spherical harmonics up to *l_max* at polar *theta*, azimuth *phi*.

    Returns ``(n_points, (l_max+1)^2)``, ordered l-major, m from ``-l..l``.
    """
    from scipy.special import lpmv
    ct = np.cos(theta)
    out = np.zeros((theta.shape[0], (l_max + 1) ** 2), dtype=np.float64)
    k = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            am = abs(m)
            norm = math.sqrt((2 * l + 1) / (4 * math.pi)
                             * math.factorial(l - am) / math.factorial(l + am))
            P = lpmv(am, l, ct)
            if m > 0:
                y = math.sqrt(2) * norm * P * np.cos(am * phi)
            elif m < 0:
                y = math.sqrt(2) * norm * P * np.sin(am * phi)
            else:
                y = norm * P
            out[:, k] = y
            k += 1
    return out


class SHDescriptor(nn.Module):
    """Precomputed linear map from voxels to ``(shell, lm)`` coefficients,
    plus the invariant cross-shell power built from them.

    Parameters
    ----------
    grid : int
        Cube side length (33).
    l_max : int
        Highest harmonic degree (6 = up to 12 angular lobes per shell).
    n_shells : int
        Gaussian radial shells from *r_min* to *r_max*.
    sigma : float
        Shell width in voxels.
    r_min : float
        Inner shell radius.  Shells below ~2 hold too few voxels to
        resolve ``l=6`` and introduce lattice artefacts under rotation.
    """

    def __init__(self, grid=33, l_max=6, n_shells=12, sigma=1.0,
                 r_min=2.0, r_max=None):
        super().__init__()
        self.grid, self.l_max, self.n_shells = grid, l_max, n_shells
        c = (grid - 1) / 2.0
        r_max = r_max or c
        idx = np.indices((grid,) * 3).reshape(3, -1).T.astype(np.float64) - c
        r = np.linalg.norm(idx, axis=1)
        keep = r <= r_max + 0.5
        self.register_buffer('keep_idx',
                             torch.tensor(np.nonzero(keep)[0]))
        xyz, r = idx[keep], r[keep]
        rs = np.clip(r, 1e-9, None)
        theta = np.arccos(np.clip(xyz[:, 2] / rs, -1, 1))
        phi = np.arctan2(xyz[:, 1], xyz[:, 0])
        Y = real_sph_harm(l_max, theta, phi)
        Y[r < 0.5, 1:] = 0.0                        # centre voxel: l=0 only
        centres = np.linspace(r_min, r_max, n_shells)
        W = np.exp(-(r[None, :] - centres[:, None]) ** 2 / (2 * sigma ** 2))
        W = W / W.sum(1, keepdims=True) * 4 * math.pi
        M = (W[:, None, :] * Y.T[None, :, :])
        self.register_buffer('M', torch.tensor(
            M.reshape(n_shells * Y.shape[1], -1), dtype=torch.float32))
        self.n_lm = Y.shape[1]

    def coefficients(self, f):
        """``f: (N, G, G, G)`` or ``(N, 1, G, G, G)`` -> ``(N, S, n_lm)``."""
        if f.dim() == 5:
            f = f[:, 0]
        x = f.reshape(f.shape[0], -1)[:, self.keep_idx].float()
        return (x @ self.M.T).view(f.shape[0], self.n_shells, self.n_lm)

    def forward(self, f, compress=True):
        """Invariant descriptors per degree *l*.

        Returns a dict: ``l=0`` -> ``(N, S)`` shell means;
        ``l >= 1`` -> ``(N, S, S)`` cross-shell power, amplitude-compressed
        as ``sign(P) * sqrt(|P|)`` so the gradient doesn't vanish at the
        sphere (where all ``l >= 1`` coefficients are zero).
        """
        c = self.coefficients(f)
        out = {0: c[:, :, 0]}
        for l in range(1, self.l_max + 1):
            cl = c[:, :, l * l:(l + 1) * (l + 1)]
            P = cl @ cl.transpose(1, 2)
            out[l] = P / torch.sqrt(P.abs() + 1e-12) if compress else P
        return out


def invariant_losses(rec, tgt, desc: SHDescriptor, w_pattern=1.0,
                     w_mean=0.5, w_energy=0.5, w_tv=0.25):
    """Rotation-invariant reconstruction losses for one protein's residues.

    Parameters
    ----------
    rec, tgt : Tensor, ``(N, G, G, G)`` or ``(N, 1, G, G, G)``
    desc : SHDescriptor

    Returns
    -------
    dict
        ``inv_pattern`` (residue-centred, 1.0 = residue-independent),
        ``inv_mean`` (raw calibration), ``inv_energy`` (radial energy
        profile), ``inv_tv`` (excess total variation), ``inv_total``,
        and per-degree diagnostics.
    """
    if rec.dim() == 5:
        rec = rec[:, 0]
    if tgt.dim() == 5:
        tgt = tgt[:, 0]
    dr, dt = desc(rec), desc(tgt)
    pat, mean = [], []
    for l in dr:
        a, b = dr[l], dt[l]
        mean.append(_overlap(a, b).mean())
        ac = a - a.mean(0, keepdim=True)
        bc = b - b.mean(0, keepdim=True)
        pat.append(_overlap(ac, bc).mean())
    inv_pattern = torch.stack(pat).mean()
    inv_mean = torch.stack(mean).mean()
    e_rec = desc.coefficients(rec * rec)[:, :, 0]
    e_tgt = desc.coefficients(tgt * tgt)[:, :, 0]
    inv_energy = _overlap(e_rec, e_tgt).mean()
    inv_tv = tv_excess(rec, tgt, desc)
    return {'inv_pattern': inv_pattern, 'inv_mean': inv_mean,
            'inv_energy': inv_energy, 'inv_tv': inv_tv,
            'inv_total': (w_pattern * inv_pattern + w_mean * inv_mean
                          + w_energy * inv_energy + w_tv * inv_tv),
            'inv_pattern_by_l': torch.stack(pat).detach(),
            'inv_mean_by_l': torch.stack(mean).detach()}


# ----------------------------------------- cross-residue terms

CROSS_KNEE = 1.0


def _cross_compress(X, knee):
    """``X / sqrt(|X| + knee)``: signed sqrt above the knee, linear below.

    A data-scale knee prevents near-zero cross-products from dominating
    the gradient (which they do at the per-residue knee of 1e-12, since
    cross-products cross zero routinely).
    """
    return X / torch.sqrt(X.abs() + knee)


def _cross_terms(cr, ct, l_max, window, knee_scale=CROSS_KNEE):
    """Coefficient-level cross-residue overlaps per separation.

    Returns ``(centred, raw)`` lists of per-separation losses.
    """
    N = cr.shape[0]
    cen, raw = [], []
    for d in range(1, window + 1):
        if N - d < 2:
            break
        pc, pr = [], []
        for l in range(1, l_max + 1):
            sl = slice(l * l, (l + 1) * (l + 1))
            Xr = cr[:-d, :, sl] @ cr[d:, :, sl].transpose(1, 2)
            Xt = ct[:-d, :, sl] @ ct[d:, :, sl].transpose(1, 2)
            # One knee from the target for both sides, so a perfect
            # reconstruction compresses identically and scores 0.
            knee = (knee_scale * Xt.abs().mean()).detach().clamp(min=1e-12) \
                if knee_scale > 0 else Xt.new_tensor(1e-12)
            xr, xt = _cross_compress(Xr, knee), _cross_compress(Xt, knee)
            pr.append(_overlap(xr, xt).mean())
            pc.append(_overlap(xr - xr.mean(0, keepdim=True),
                               xt - xt.mean(0, keepdim=True)).mean())
        cen.append(torch.stack(pc).mean())
        raw.append(torch.stack(pr).mean())
    return cen, raw


def cross_residue_losses(rec, tgt, desc: SHDescriptor, window=10):
    """Relative orientation between residues of one protein.

    For every pair ``(j, k)`` with ``1 <= k-j <= window`` and every
    degree ``l >= 1``, compares the shell-by-shell inner products of the
    two residues' SH coefficients.  Exactly invariant under a shared
    rotation of the whole protein; changes when one residue's environment
    turns relative to another.

    Returns ``inv_cross`` (centred, 1.0 = residue-independent),
    ``inv_cross_mean`` (raw), and ``inv_cross_by_sep`` (per-separation
    diagnostic, NaN where the protein is too short).
    """
    if rec.dim() == 5:
        rec = rec[:, 0]
    if tgt.dim() == 5:
        tgt = tgt[:, 0]
    N = rec.shape[0]
    by_sep = rec.new_full((window,), float('nan'))
    if N < 3:
        z = rec.new_zeros(())
        return {'inv_cross': z, 'inv_cross_mean': z, 'inv_cross_by_sep': by_sep}
    cr, ct = desc.coefficients(rec), desc.coefficients(tgt)
    cen, raw = _cross_terms(cr, ct, desc.l_max, window)
    cen, raw = torch.stack(cen), torch.stack(raw)
    by_sep[:cen.shape[0]] = cen.detach()
    return {'inv_cross': cen.mean(), 'inv_cross_mean': raw.mean(),
            'inv_cross_by_sep': by_sep}


# ----------------------------------------- total variation

def isotropic_tv(f, mask):
    """Sum of ``|grad f|`` (L2 of forward differences) over the sphere, per field."""
    if f.dim() == 5:
        f = f[:, 0]
    dx = f[:, 1:, :-1, :-1] - f[:, :-1, :-1, :-1]
    dy = f[:, :-1, 1:, :-1] - f[:, :-1, :-1, :-1]
    dz = f[:, :-1, :-1, 1:] - f[:, :-1, :-1, :-1]
    g = torch.sqrt(dx * dx + dy * dy + dz * dz + 1e-20)
    m = mask[0, 0, :-1, :-1, :-1]
    return (g * m).reshape(f.shape[0], -1).sum(1)


def roughness(f, mask):
    """``TV / sqrt(energy)`` over the sphere, per field.

    Scale-free: invariant to a global rescaling of the field.
    """
    if f.dim() == 5:
        f = f[:, 0]
    e = ((f * f) * mask[0, 0]).reshape(f.shape[0], -1).sum(1)
    return isotropic_tv(f, mask) / torch.sqrt(e + 1e-30)


def tv_excess(rec, tgt, desc):
    """One-sided excess roughness: ``relu(rough(rec) - rough(tgt)) / rough(tgt)``.

    Penalises axis-aligned striping (a blind spot of second-order spectral
    invariants) without charging reconstructions that are as smooth as or
    smoother than the target.  Capped at 2 so a wild early output can't
    dominate.
    """
    mask = _sphere_mask_like(rec, desc)
    rr, rt = roughness(rec, mask), roughness(tgt, mask)
    return (torch.relu(rr - rt) / rt.clamp(min=1e-20)).clamp(max=2.0).mean()


_SPH = {}


def _sphere_mask_like(x, desc):
    key = (desc.grid, str(x.device))
    if key not in _SPH:
        m = torch.zeros(desc.grid ** 3, device=x.device)
        m[desc.keep_idx.to(x.device)] = 1.0
        _SPH[key] = m.view(1, 1, desc.grid, desc.grid, desc.grid)
    return _SPH[key]


# ---------------------------------------------------------------- rotations

def rotate_cube(x, k):
    """*k*-th of the 24 proper cube rotations (exact voxel permutation).

    ``x: (N, G, G, G)``.
    """
    import itertools
    rots = []
    for perm in itertools.permutations(range(3)):
        sp = np.linalg.det(np.eye(3)[list(perm)])
        for flips in itertools.product((1, -1), repeat=3):
            if sp * np.prod(flips) > 0:
                rots.append((perm, tuple(i for i, f in enumerate(flips)
                                         if f < 0)))
    perm, flips = rots[k % 24]
    y = x.permute(0, *[p + 1 for p in perm])
    if flips:
        y = torch.flip(y, dims=[f + 1 for f in flips])
    return y.contiguous()


def rotate_continuous(x, R):
    """Rotate ``(N, G, G, G)`` by 3x3 *R* about the grid centre, trilinear."""
    N, G = x.shape[0], x.shape[-1]
    A = torch.zeros(N, 3, 4, dtype=x.dtype, device=x.device)
    A[:, :, :3] = torch.as_tensor(R, dtype=x.dtype, device=x.device)
    grid = F.affine_grid(A, (N, 1, G, G, G), align_corners=True)
    return F.grid_sample(x.unsqueeze(1), grid, mode='bilinear',
                         padding_mode='zeros', align_corners=True)[:, 0]


def random_rotation(gen):
    """Random rotation matrix from a unit quaternion."""
    q = torch.randn(4, generator=gen)
    q = q / q.norm()
    w, x, y, z = q.tolist()
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# ---------------------------------------------------------------- self-test

def _self_test():
    """Invariance checks and candidate ranking on synthetic targets."""
    import loss as L
    from audit_objective import synthetic, build_candidates
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    torch.set_num_threads(2)
    desc = SHDescriptor()
    t = synthetic(n_res=24, n_prot=1)[0]
    w = L.ReconWeights()
    print("=" * 76)
    print("INVARIANT LOSS  self-test on synthetic targets (24 residues)")
    print(f"  descriptor: l_max {desc.l_max}, {desc.n_shells} shells, "
          f"{desc.M.shape[0]} coefficients from {desc.M.shape[1]} voxels")
    print("=" * 76)

    # 1. Exact invariance under the 24 cube rotations
    base = invariant_losses(t, t, desc)['inv_total']
    worst = 0.0
    for k in range(24):
        v = float(invariant_losses(rotate_cube(t, k), t, desc)['inv_total'])
        worst = max(worst, v)
    print(f"\n1. exact invariance: inv_total(R t, t) over 24 cube rotations  "
          f"max {worst:.2e}   (identity {float(base):.2e})")

    # 2. Invariance under continuous rotation (analytic + trilinear)
    G = t.shape[-1]
    ax = torch.arange(G).float()
    XX, YY, ZZ = torch.meshgrid(ax, ax, ax, indexing='ij')

    def blobs(C, A, S):
        f = torch.zeros(G, G, G)
        for ce, a, sg in zip(C, A, S):
            f += a * torch.exp(-((XX - ce[0]) ** 2 + (YY - ce[1]) ** 2
                                 + (ZZ - ce[2]) ** 2) / (2 * sg * sg))
        return f
    rng = np.random.default_rng(0)
    spec = [(rng.normal(16, 4, (5, 3)),
             rng.choice([-1, 1], 5) * rng.uniform(0.01, 0.03, 5),
             rng.uniform(1.2, 2.0, 5)) for _ in range(24)]
    TA = torch.stack([blobs(C, A, S) for C, A, S in spec])
    TAr = torch.stack([blobs((random_rotation(gen) @ (C - 16.0).T).T + 16.0,
                             A, S) for C, A, S in spec])
    an = float(invariant_losses(TAr, TA, desc)['inv_total'])
    an_old = float(L.reconstruction_losses(TAr.unsqueeze(1), TA, w)
                   ['pattern_loss'])
    vals = [float(invariant_losses(rotate_continuous(t, random_rotation(gen)),
                                   t, desc)['inv_total']) for _ in range(8)]
    print(f"2. continuous rotation, ANALYTIC (lattice error only): inv_total "
          f"{an:.4f}   <- the training floor")
    print(f"   same fields under the OLD pattern_loss:               "
          f"{an_old:.4f}   <- correct shape, wrong frame")
    print(f"   continuous rotation via trilinear resampling:          mean "
          f"{np.mean(vals):.4f}  (adds interpolation blur)")

    # 3. Candidate ranking
    cands = build_candidates(t, gen)
    rot_perfect = torch.stack([rotate_continuous(t[i:i + 1],
                                                 random_rotation(gen))[0]
                               for i in range(t.shape[0])])
    rot_feat = torch.stack([rotate_continuous(cands['features only'][i:i + 1],
                                              random_rotation(gen))[0]
                            for i in range(t.shape[0])])
    fixed = t[3:4].expand_as(t).contiguous()
    Gs = t.shape[-1]
    ys = torch.arange(Gs).float().view(1, 1, Gs, 1)
    stripes = torch.cos(math.pi * ys) * (ys > 8).float() * (ys < 15).float()
    striped = t + 0.15 * float(t.std()) * stripes.expand_as(t)
    ordered = [("perfect", cands["perfect"]),
               ("perfect, each residue ROTATED", rot_perfect),
               ("noise 20%", cands["noise 20%"]),
               ("features only", cands["features only"]),
               ("features only, ROTATED", rot_feat),
               ("noise 50%", cands["noise 50%"]),
               ("amp x2", cands["amp x2"]),
               ("rim relocated", cands["rim relocated"]),
               ("perfect + axis stripes", striped),
               ("residues shuffled", cands["residues shuffled"]),
               ("one fixed shape for all", fixed),
               ("protein blob (resid mean)", cands["protein blob (resid mean)"]),
               ("SPHERE (observed)", cands["SPHERE (observed)"]),
               ("zeros", cands["zeros"])]
    print("\n3. RANKING   old = voxel-wise objective   new = invariant")
    print(f"   {'candidate':<32}{'old pattern':>12}{'old total':>11}"
          f"{'new inv_pattern':>17}{'new inv_mean':>14}{'inv_energy':>12}"
          f"{'inv_tv':>8}{'new total':>11}")
    print("   " + "-" * 117)
    rows = {}
    for nm, c in ordered:
        old = L.reconstruction_losses(c.unsqueeze(1), t, w, None)
        new = invariant_losses(c, t, desc)
        rows[nm] = (float(old['pattern_loss']), float(old['recon_total']),
                    float(new['inv_pattern']), float(new['inv_mean']),
                    float(new['inv_total']), float(new['inv_energy']),
                    float(new['inv_tv']))
        o = rows[nm]
        print(f"   {nm:<32}{o[0]:>12.4f}{o[1]:>11.4f}{o[2]:>17.4f}"
              f"{o[3]:>14.4f}{o[5]:>12.4f}{o[6]:>8.3f}{o[4]:>11.4f}")

    def chk(name, ok, detail=""):
        print(f"   [{'PASS' if ok else 'FAIL'}] {name}  {detail}")

    print()
    r = rows
    chk("OLD: correct shape in wrong frame punished like noise 50%",
        r["perfect, each residue ROTATED"][0] > 0.5 * r["noise 50%"][0],
        f"old pattern {r['perfect, each residue ROTATED'][0]:.3f} vs "
        f"{r['noise 50%'][0]:.3f}")
    chk("NEW: rotated-perfect near perfect, far below sphere",
        r["perfect, each residue ROTATED"][4] < 0.1 * r["SPHERE (observed)"][4],
        f"{r['perfect, each residue ROTATED'][4]:.4f} vs sphere "
        f"{r['SPHERE (observed)'][4]:.4f}")
    chk("NEW: rotated features beat sphere by a wide margin",
        r["features only, ROTATED"][4] < 0.5 * r["SPHERE (observed)"][4],
        f"{r['features only, ROTATED'][4]:.4f} vs {r['SPHERE (observed)'][4]:.4f}")
    chk("NEW: sphere, blob and zeros all score inv_pattern == 1.0",
        all(abs(r[k][2] - 1.0) < 1e-3 for k in
            ("SPHERE (observed)", "protein blob (resid mean)", "zeros")),
        str([round(r[k][2], 4) for k in ("SPHERE (observed)",
                                         "protein blob (resid mean)", "zeros")]))
    chk("NEW: one fixed shape for all also scores inv_pattern == 1.0",
        abs(r["one fixed shape for all"][2] - 1.0) < 1e-3,
        f"{r['one fixed shape for all'][2]:.4f}")
    chk("NEW: monotone in noise", r["perfect"][4] < r["noise 20%"][4]
        < r["noise 50%"][4], "")
    chk("NEW: rim relocation caught by inv_energy",
        r["rim relocated"][5] > 0.2 and r["rim relocated"][4]
        > 3 * r["perfect, each residue ROTATED"][4],
        f"inv_energy {r['rim relocated'][5]:.4f}")
    chk("NEW: inv_energy is itself invariant (rotated-perfect < 0.05)",
        r["perfect, each residue ROTATED"][5] < 0.05,
        f"{r['perfect, each residue ROTATED'][5]:.4f}")
    chk("NEW: striping nearly free to spectral terms (the hole) ...",
        r["perfect + axis stripes"][2] + r["perfect + axis stripes"][3]
        + r["perfect + axis stripes"][5] < 0.3,
        f"pattern+mean+energy "
        f"{r['perfect + axis stripes'][2] + r['perfect + axis stripes'][3] + r['perfect + axis stripes'][5]:.4f}")
    chk("NEW: ... and inv_tv charges it",
        r["perfect + axis stripes"][6] > 0.1,
        f"inv_tv {r['perfect + axis stripes'][6]:.3f}")
    chk("NEW: inv_tv is one-sided: rotated-perfect, features, amp x2 all ~0",
        max(r["perfect, each residue ROTATED"][6], r["amp x2"][6],
            r["features only"][6]) < 0.05,
        f"rot {r['perfect, each residue ROTATED'][6]:.3f} amp2 "
        f"{r['amp x2'][6]:.3f} feat {r['features only'][6]:.3f}")
    chk("NEW: shuffled residues score near the blob",
        r["residues shuffled"][2] > 0.8, f"{r['residues shuffled'][2]:.4f}")
    chk("NEW: 2x amplitude error costs ~0.2",
        0.15 < r["amp x2"][2] < 0.30, f"inv_pattern {r['amp x2'][2]:.4f}")
    chk("NEW: noise is cheap to spectral terms, inv_tv charges roughness",
        r["noise 50%"][2] + r["noise 50%"][3] + r["noise 50%"][5] < 0.1,
        f"spectral {r['noise 50%'][2] + r['noise 50%'][3] + r['noise 50%'][5]:.4f}"
        f"  inv_tv {r['noise 50%'][6]:.3f}")

    # 4. Gradient properties
    tv = t.clone().requires_grad_(True)
    invariant_losses(tv, t, desc)['inv_total'].backward()
    g_match = float(tv.grad.norm())
    sv = cands["SPHERE (observed)"].clone().requires_grad_(True)
    invariant_losses(sv, t, desc)['inv_total'].backward()
    g_sph = float(sv.grad.norm())
    zv = torch.zeros_like(t).requires_grad_(True)
    invariant_losses(zv, t, desc)['inv_total'].backward()
    g_zero = float(zv.grad.norm())
    print()
    chk("stationary at rec == target", g_match < 1e-5, f"|grad| {g_match:.2e}")
    chk("finite, nonzero gradient at the sphere",
        np.isfinite(g_sph) and g_sph > 1e-6, f"|grad| {g_sph:.2e}")
    chk("finite gradient at rec == 0", np.isfinite(g_zero), f"|grad| {g_zero:.2e}")
    gs = sv.grad.reshape(t.shape[0], -1)
    spec = float((gs - gs.mean(0, keepdim=True)).norm() / gs.norm())
    chk("sphere gradient is residue-specific (centred share > 0.5)", spec > 0.5,
        f"{spec:.3f}")
    print("=" * 76)


def _protein_crops(spec, centres, R=None, G=33):
    """Crops of one analytic field (Gaussian blobs), each centred on a
    residue.  With *R*, the whole protein is rotated first."""
    atoms, amps, sig = spec
    if R is not None:
        R = torch.as_tensor(R, dtype=torch.float32)
        atoms, centres = atoms @ R.T, centres @ R.T
    ax = torch.arange(G, dtype=torch.float32) - (G - 1) / 2.0
    grid = torch.stack(torch.meshgrid(ax, ax, ax, indexing='ij'), -1).reshape(-1, 3)
    out = []
    for c in centres:
        d2 = torch.cdist(grid + c, atoms).pow(2)
        out.append((torch.exp(-d2 / (2 * sig * sig)) @ amps).reshape(G, G, G))
    return torch.stack(out)


def _cross_self_test(n_res=24, window=10):
    """Cross-residue term self-test on a synthetic protein."""
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    desc = SHDescriptor()
    # Build a compact chain
    pos, tries = [torch.zeros(3)], 0
    while len(pos) < n_res:
        s = torch.randn(3, generator=gen)
        c = pos[-1] + 3.8 * s / s.norm()
        tries += 1
        if c.norm() < 10 and all((c - q).norm() > 3.5 for q in pos[:-1]):
            pos.append(c)
            tries = 0
        elif tries > 2000:
            pos, tries = [torch.zeros(3)], 0
    ca = torch.stack(pos)
    sc = ca + 2.5 * F.normalize(torch.randn(n_res, 3, generator=gen), dim=1)
    atoms = torch.cat([ca, sc])
    amps = torch.cat([0.01 * torch.randn(n_res, generator=gen),
                      0.03 * torch.randint(-1, 2, (n_res,), generator=gen).float()])
    spec = (atoms, amps, 1.5)
    t = _protein_crops(spec, ca)

    def both(rec):
        c = cross_residue_losses(rec, t, desc, window)
        p = invariant_losses(rec, t, desc)
        return float(c['inv_cross']), float(c['inv_cross_mean']), float(p['inv_pattern'])

    print("=" * 76)
    print(f"CROSS-RESIDUE TERM  self-test: {n_res} residues, pairs |i-j| <= {window}")
    print("=" * 76)
    print(f"   {'candidate':<46}{'inv_cross':>11}{'cross_mean':>12}{'inv_pattern':>13}")
    rows = {}
    rows['perfect'] = both(t)
    rows['whole protein, cube rotation (worst of 23)'] = max(
        (both(rotate_cube(t, k)) for k in range(1, 24)), key=lambda q: q[0])
    rows['whole protein, continuous rotation (analytic)'] = both(
        _protein_crops(spec, ca, random_rotation(gen)))
    rows['each crop its own cube rotation'] = both(torch.stack(
        [rotate_cube(t[i:i + 1], 1 + i % 23)[0] for i in range(n_res)]))
    rows['each crop its own continuous rotation'] = both(torch.stack(
        [_protein_crops((atoms - ca[i], amps, 1.5), ca[i:i + 1] * 0,
                        random_rotation(gen))[0] for i in range(n_res)]))
    rows['one field for every residue'] = both(t[5:6].expand_as(t).contiguous())
    rows['zeros'] = both(torch.zeros_like(t))
    for k, v in rows.items():
        print(f"   {k:<46}{v[0]:>11.4f}{v[1]:>12.4f}{v[2]:>13.4f}")

    def chk(name, ok, detail=""):
        print(f"   [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
        return ok

    r = rows
    ok = [
        chk("perfect scores 0", max(r['perfect'][:2]) < 1e-5, f"{r['perfect'][:2]}"),
        chk("exact under one cube rotation of the whole protein",
            max(r['whole protein, cube rotation (worst of 23)'][:2]) < 1e-4,
            f"{r['whole protein, cube rotation (worst of 23)'][:2]}"),
        chk("whole-protein rotation much cheaper than independent crop rotations",
            r['whole protein, continuous rotation (analytic)'][0]
            < 0.25 * r['each crop its own continuous rotation'][0],
            f"{r['whole protein, continuous rotation (analytic)'][0]:.4f} vs "
            f"{r['each crop its own continuous rotation'][0]:.4f}"),
        chk("independent crop rotations: invisible to inv_pattern, caught by inv_cross",
            r['each crop its own cube rotation'][2] < 1e-4
            and r['each crop its own cube rotation'][0] > 0.2,
            f"inv_pattern {r['each crop its own cube rotation'][2]:.2e}, "
            f"inv_cross {r['each crop its own cube rotation'][0]:.4f}"),
        chk("one field for every residue scores inv_cross == 1.0",
            abs(r['one field for every residue'][0] - 1.0) < 1e-4,
            f"{r['one field for every residue'][0]:.6f}"),
    ]
    rec = t.clone().mul_(0.5).requires_grad_(True)
    cross_residue_losses(rec, t, desc, window)['inv_cross'].backward()
    g = float(rec.grad.norm())
    ok.append(chk("gradient finite and non-zero", np.isfinite(g) and g > 0, f"|grad| {g:.2e}"))

    # Gradient concentration test: data-scale knee vs 1e-12 knee
    pred = torch.stack([rotate_cube(t[i:i + 1], 1 + i % 23)[0] for i in range(n_res)])
    ct = desc.coefficients(t)

    def spikes(knee_scale):
        cr = desc.coefficients(pred).detach().requires_grad_(True)
        cen, raw = _cross_terms(cr, ct, desc.l_max, window, knee_scale)
        (torch.stack(cen).mean() + 0.5 * torch.stack(raw).mean()).backward()
        gg = cr.grad.abs().flatten()
        gg = gg[gg > 0]
        top = gg.topk(max(1, gg.numel() // 1000)).values
        return (float(gg.max() / gg.median()),
                float(top.pow(2).sum() / gg.pow(2).sum()))
    old_r, old_share = spikes(0.0)
    new_r, new_share = spikes(CROSS_KNEE)
    print(f"   gradient largest/median:  old knee {old_r:,.0f}x   data knee {new_r:,.0f}x")
    print(f"   top-0.1% share of squared grad norm:  old {old_share:.1%}   new {new_share:.1%}")
    ok.append(chk("data knee reduces gradient concentration",
                  new_share * 2 < old_share and new_r < old_r,
                  f"share {old_share:.1%} -> {new_share:.1%}, "
                  f"peak {old_r:,.0f}x -> {new_r:,.0f}x"))
    print(f"   {sum(ok)}/{len(ok)} passed")
    print("=" * 76)
    return all(ok)


if __name__ == '__main__':
    import sys
    if '--cross' in sys.argv:
        sys.exit(0 if _cross_self_test() else 1)
    _self_test()
    _cross_self_test()
