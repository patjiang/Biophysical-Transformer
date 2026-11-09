"""SE3PLM building blocks.

Equivariant attention, the voxel decoder, the RoPE sequence encoder,
representation-collapse diagnostics, and voxel-level losses.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from se3cnn.image.convolution import SE3Convolution
except ImportError as _e:
    # collapse_penalty / effective_rank / InteractionTokenBlock are plain
    # torch and are imported by audit and probe scripts that don't need the
    # 3D-conv backend.  Fail at layer construction, not at module import.
    _SE3_ERR = _e

    def SE3Convolution(*a, **k):
        raise ImportError(
            "this layer needs the equivariant 3D-convolution backend, which "
            f"did not import: {_SE3_ERR}") from _SE3_ERR
import ot
from einops import rearrange
from tqdm import tqdm


# ------------------------------------------------- equivariant 3D attention

class SE3ConvolutionalSelfAttention3d(nn.Module):
    """SE(3)-equivariant gated self-attention over a voxel grid."""

    def __init__(self, scalar_ch, vector_ch, dimprod_scalar, dimprod_vector):
        super().__init__()
        self.scalar_ch = scalar_ch
        self.vector_ch = vector_ch
        self.Rs_in = [(scalar_ch, 0), (vector_ch, 1)]
        self.Rs_qk = [(dimprod_scalar, 0), (dimprod_vector, 1)]
        self.dw_conv_q = SE3Convolution(Rs_in=self.Rs_in, Rs_out=self.Rs_in, size=5, padding=2)
        self.dw_conv_k = SE3Convolution(Rs_in=self.Rs_in, Rs_out=self.Rs_in, size=5, padding=2)
        self.dw_conv_v = SE3Convolution(Rs_in=self.Rs_in, Rs_out=self.Rs_in, size=5, padding=2)
        self.pw_conv_q = SE3Convolution(Rs_in=self.Rs_in, Rs_out=self.Rs_qk, size=1)
        self.pw_conv_k = SE3Convolution(Rs_in=self.Rs_in, Rs_out=self.Rs_qk, size=1)
        Rs_gate_in = self.Rs_qk + self.Rs_qk
        self.pw_conv_gate = nn.Sequential(
            SE3Convolution(
                Rs_in=Rs_gate_in,
                Rs_out=[(scalar_ch + vector_ch, 0)],
                size=1
            ),
            nn.Sigmoid()
        )
        Rs_ffn_hidden = [(scalar_ch * 3, 0), (vector_ch * 3, 1)]
        self.pw_conv_ffn = nn.Sequential(
            SE3Convolution(Rs_in=self.Rs_in, Rs_out=Rs_ffn_hidden, size=1),
            SE3Convolution(Rs_in=Rs_ffn_hidden, Rs_out=self.Rs_in, size=1)
        )

    def forward(self, x):
        feat_q = self.dw_conv_q(x)
        feat_k = self.dw_conv_k(x)
        feat_v = self.dw_conv_v(x)
        q = self.pw_conv_q(feat_q)
        k = self.pw_conv_k(feat_k)
        qk_concat = torch.cat([q, k], dim=1)
        gate = self.pw_conv_gate(qk_concat)
        scalar_gate = gate[:, :self.scalar_ch]
        vector_gate = gate[:, self.scalar_ch:]
        scalar_v = feat_v[:, :self.scalar_ch]
        gated_scalar_v = scalar_v * scalar_gate
        vector_v = feat_v[:, self.scalar_ch:]
        B, _, H, W, D = vector_v.shape
        vector_v_reshaped = vector_v.view(B, self.vector_ch, 3, H, W, D)
        vector_gate_expanded = vector_gate.unsqueeze(2)
        gated_vector_v = (vector_v_reshaped * vector_gate_expanded).view(B, -1, H, W, D)
        gated_v = torch.cat([gated_scalar_v, gated_vector_v], dim=1)
        out_feat = self.pw_conv_ffn(gated_v)
        return x + out_feat


class EquivariantFeedForward(nn.Module):
    """Scalar/vector feed-forward with a gating mechanism."""

    def __init__(self, scalar_ch, vector_ch, mult=2, dropout=0.1):
        super().__init__()
        self.scalar_ch = scalar_ch
        self.vector_ch = vector_ch
        hidden_s = int(scalar_ch * mult)
        hidden_v = int(vector_ch * mult)
        self.s_up = nn.Conv3d(scalar_ch, hidden_s, 1)
        self.s_down = nn.Conv3d(hidden_s, scalar_ch, 1)
        self.act = nn.GELU()
        self.v_up_weight = nn.Parameter(torch.empty(hidden_v, vector_ch))
        self.v_down_weight = nn.Parameter(torch.empty(vector_ch, hidden_v))
        nn.init.kaiming_uniform_(self.v_up_weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.v_down_weight, a=5**0.5)
        self.gate_proj = nn.Conv3d(hidden_s, hidden_v, 1)
        self.dropout_s = nn.Dropout(dropout)

    def forward(self, x):
        x_s = x[:, :self.scalar_ch]
        x_v = x[:, self.scalar_ch:]
        h_s = self.act(self.s_up(x_s))
        B, _, D, H, W = x_v.shape
        x_v_view = x_v.view(B, self.vector_ch, 3, D, H, W)
        h_v = torch.einsum('bicdhw, oi -> bocdhw', x_v_view, self.v_up_weight)
        gates = torch.sigmoid(self.gate_proj(h_s))
        h_v_gated = h_v * gates.unsqueeze(2)
        out_v_view = torch.einsum('bicdhw, oi -> bocdhw', h_v_gated, self.v_down_weight)
        out_v = out_v_view.reshape(B, self.vector_ch * 3, D, H, W)
        out_s = self.s_down(h_s)
        out = torch.cat([self.dropout_s(out_s), out_v], dim=1)
        return out


# ----------------------------------------------------------- voxel losses

def get_random_projections_pt(n_features, n_projections, device):
    """Orthogonal random projections via QR decomposition."""
    A = torch.randn(n_features, n_projections, device=device)
    Q, _ = torch.linalg.qr(A)
    return Q


def wa_loss(ori, reconstruction, denorm=True, split_posneg=False, max_val=4981.8994140625):
    """Sliced-Wasserstein distance over axis-marginal projections.

    Sorts projections across the batch, so it is exactly invariant to
    permuting which residue gets which reconstruction.
    """
    with torch.amp.autocast('cuda', enabled=False):
        dev = ori.device
        scale_factor = max_val if denorm else 1.0

        def compute_axis_loss(t_ori, t_rec, pattern, projvec):
            flat_ori = rearrange(t_ori, pattern)
            flat_rec = rearrange(t_rec, pattern)
            dist = ot.sliced.max_sliced_wasserstein_distance(
                flat_ori, flat_rec, projections=projvec
            )
            return dist

        accumulated_loss = 0.0
        if split_posneg:
            with torch.no_grad():
                ori_pos_mass = torch.relu(ori).sum()
                ori_neg_mass = torch.relu(-ori).sum()
                eps = 1e-6
                total_mass = ori_pos_mass + ori_neg_mass + eps
                w_pos = total_mass / (2 * (ori_pos_mass + eps))
                w_neg = total_mass / (2 * (ori_neg_mass + eps))
                w_pos = torch.clamp(w_pos, min=0.1, max=10.0)
                w_neg = torch.clamp(w_neg, min=0.1, max=10.0)
                total_weight = w_pos + w_neg
                w_pos = w_pos / total_weight * 2.0
                w_neg = w_neg / total_weight * 2.0
            phases = [
                ("pos", lambda x: torch.relu(x), w_pos),
                ("neg", lambda x: torch.relu(-x), w_neg)
            ]
        else:
            phases = [("all", lambda x: x, 1.0)]

        for phase_name, op, weight in phases:
            curr_rec = op(reconstruction.float())
            with torch.no_grad():
                curr_ori = op(ori.float())
            dist = 0.0

            projvec = get_random_projections_pt(1089, 50, dev)
            dist += compute_axis_loss(curr_ori, curr_rec, 'b s x y z -> (b z) (s x y)', projvec)
            dist += compute_axis_loss(curr_ori, curr_rec, 'b s x y z -> (b y) (s x z)', projvec)
            dist += compute_axis_loss(curr_ori, curr_rec, 'b s x y z -> (b x) (s y z)', projvec)
            del projvec

            diagvec = get_random_projections_pt(33, 50, dev)
            dist += compute_axis_loss(curr_ori, curr_rec, 'b s x y z -> (b x y) (s z)', diagvec)
            dist += compute_axis_loss(curr_ori, curr_rec, 'b s x y z -> (b x z) (s y)', diagvec)
            dist += compute_axis_loss(curr_ori, curr_rec, 'b s x y z -> (b y z) (s x)', diagvec)
            del diagvec

            accumulated_loss += dist * weight
            del curr_rec, curr_ori

        return accumulated_loss * scale_factor


def laplacian_pyramid_loss(rec, target, levels=4):
    """L1 at each pyramid level, weighted ``[1, 2, 4, 8]`` coarse to fine."""
    loss = 0.0
    r, t = rec, target
    weights = [2 ** i for i in range(levels)]
    for i in range(levels):
        loss += F.l1_loss(r, t) * weights[i]
        if i < levels - 1:
            r = F.avg_pool3d(r, 2)
            t = F.avg_pool3d(t, 2)
    return loss / sum(weights)


# --------------------------------------------------------- tokenizer util

def seqs_to_tens(seqs):
    """Convert 3-mer string tokens to integer ids."""
    seqdict = {
        'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'E': 6, 'Q': 7, 'G': 8,
        'H': 9, 'I': 10, 'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16,
        'T': 17, 'W': 18, 'Y': 19, 'V': 20
    }
    olist = []
    for s in seqs:
        a = s.split('_')[0]
        if(len(a) != 3):
            olist.append(9000)
            continue
        uuid = seqdict.get(a[0], 40)*20**2 + seqdict.get(a[1], 40)*20 + seqdict.get(a[2], 40)
        olist.append(uuid)
    return torch.FloatTensor(olist).type(torch.float16)


# ---------------------------------------------------- equivariant decoder

class SE3Transformer(nn.Module):
    """SE(3)-equivariant transformer with checkpointed layers."""

    def __init__(self, scalar_ch, vector_ch, dimprod_scalar, dimprod_vector,
                 num_layers=1, ffn_mult=2, dropout=0.1):
        super().__init__()
        self.scalar_ch = scalar_ch
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'attn': SE3ConvolutionalSelfAttention3d(
                    scalar_ch, vector_ch, dimprod_scalar, dimprod_vector
                ),
                'norm_s': nn.InstanceNorm3d(scalar_ch, affine=True),
                'norm_v': nn.InstanceNorm3d(vector_ch * 3, affine=True),
                'ffn': EquivariantFeedForward(scalar_ch, vector_ch, ffn_mult, dropout)
            }))

    def forward(self, x):
        def _layer_fwd(x, attn, ffn, norm_s, norm_v, s_ch):
            x = attn(x)
            x_s = norm_s(x[:, :s_ch])
            x_v = norm_v(x[:, s_ch:])
            x_normed = torch.cat([x_s, x_v], dim=1)
            x = x + ffn(x_normed)
            return x

        for layer in self.layers:
            args = (x, layer['attn'], layer['ffn'], layer['norm_s'], layer['norm_v'], self.scalar_ch)
            if self.training:
                x = torch.utils.checkpoint.checkpoint(_layer_fwd, *args, use_reentrant=False)
            else:
                x = _layer_fwd(*args)
        return x


class SE3CrossTransformer(nn.Module):
    """SE(3)-equivariant cross-attention transformer."""

    def __init__(self, scalar_ch, vector_ch, dimprod_scalar, dimprod_vector,
                 context_dim, num_layers=1, n_heads=8, ffn_mult=2, dropout=0.1):
        super().__init__()
        self.scalar_ch = scalar_ch
        self.total_ch = scalar_ch + vector_ch * 3
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'se3_self_attn': SE3ConvolutionalSelfAttention3d(
                    scalar_ch, vector_ch, dimprod_scalar, dimprod_vector
                ),
                'norm_cross_s': nn.InstanceNorm3d(scalar_ch, affine=True),
                'norm_cross_v': nn.InstanceNorm3d(vector_ch * 3, affine=True),
                'mha': nn.MultiheadAttention(
                    embed_dim=self.total_ch,
                    kdim=context_dim,
                    vdim=context_dim,
                    num_heads=n_heads,
                    batch_first=False
                ),
                'norm_ffn_s': nn.InstanceNorm3d(scalar_ch, affine=True),
                'norm_ffn_v': nn.InstanceNorm3d(vector_ch * 3, affine=True),
                'ffn': EquivariantFeedForward(scalar_ch, vector_ch, ffn_mult, dropout)
            }))

    def forward(self, x, context):
        def _layer_fwd(x, context, se3_attn, mha, ffn,
                       norm_cs, norm_cv, norm_fs, norm_fv, s_ch):
            B, C, D, H, W = x.shape
            x = se3_attn(x)
            x_s = norm_cs(x[:, :s_ch])
            x_v = norm_cv(x[:, s_ch:])
            x_normed = torch.cat([x_s, x_v], dim=1)
            query = x_normed.flatten(2).permute(2, 0, 1)
            mha_out, _ = mha(query, context, context)
            mha_out = mha_out.permute(1, 2, 0).reshape(B, C, D, H, W)
            x = x + mha_out
            x_s = norm_fs(x[:, :s_ch])
            x_v = norm_fv(x[:, s_ch:])
            x_normed = torch.cat([x_s, x_v], dim=1)
            x = x + ffn(x_normed)
            return x

        for layer in self.layers:
            args = (
                x, context,
                layer['se3_self_attn'], layer['mha'], layer['ffn'],
                layer['norm_cross_s'], layer['norm_cross_v'],
                layer['norm_ffn_s'], layer['norm_ffn_v'],
                self.scalar_ch
            )
            if self.training:
                x = torch.utils.checkpoint.checkpoint(_layer_fwd, *args, use_reentrant=False)
            else:
                x = _layer_fwd(*args)
        return x


class VoxelizeTrunk(nn.Module):
    """Voxel decoder: latent + BOS context -> ``(B, 1, 33, 33, 33)`` field.

    Builds a 4^3 seed from the residue and global latents, upsamples
    through SE3 self-attention and cross-attention stages with FiLM
    conditioning, and refines to 33^3 with skip connections.
    """

    def __init__(self, context_dim=64, max_ctx_residues=256,
                 use_residue_pos=True):
        super().__init__()
        self.scalar_ch = 29
        self.vector_ch = 1
        self.context_dim = context_dim
        self.max_ctx_residues = max_ctx_residues
        self.use_residue_pos = use_residue_pos

        n_query = self.scalar_ch * 4 * 4 * 4
        self.context_to_query = nn.Linear(context_dim, n_query)
        self.residue_to_query = nn.Linear(context_dim, n_query)
        self.query_act = nn.GELU()
        # residue_to_query uses default init.  A previous small init
        # (std=0.02) combined with gradient-folded weight decay to drive
        # both query projections to near-zero, collapsing the seed to a
        # per-protein constant.
        nn.init.zeros_(self.residue_to_query.bias)

        self.cond_fuse = nn.Sequential(
            nn.Linear(context_dim * 2, context_dim),
            nn.LayerNorm(context_dim),
            nn.GELU(),
        )

        self.cond_proj_32 = nn.Linear(context_dim, 64)
        self.cond_proj_24 = nn.Linear(context_dim, 48)
        self.cond_proj_16 = nn.Linear(context_dim, 32)

        grid = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, 4),
            torch.linspace(-1, 1, 4),
            torch.linspace(-1, 1, 4),
            indexing='ij'
        ))
        self.register_buffer('coord_grid', grid)

        self.ctx_norm = nn.LayerNorm(context_dim)
        self.residue_pos_bias = nn.Parameter(torch.zeros(1, context_dim))
        self.sct1 = SE3CrossTransformer(
            scalar_ch=self.scalar_ch, vector_ch=self.vector_ch,
            dimprod_scalar=32, dimprod_vector=8,
            context_dim=context_dim, num_layers=1,
        )
        self.sct2 = SE3Transformer(
            scalar_ch=self.scalar_ch, vector_ch=self.vector_ch,
            dimprod_scalar=32, dimprod_vector=8, num_layers=1,
        )

        self.reduce_32_to_24 = nn.Conv3d(32, 24, 1)
        self.reduce_24_to_16 = nn.Conv3d(24, 16, 1)
        self.skip_reduce_32_to_24 = nn.Conv3d(32, 24, 1)
        self.skip_reduce_24_to_16 = nn.Conv3d(24, 16, 1)

        self.refine_16 = nn.Sequential(
            nn.Conv3d(24, 24, 3, 1, 1), nn.GroupNorm(8, 24), nn.GELU())
        self.refine_33 = nn.Sequential(
            nn.Conv3d(16, 16, 3, 1, 1), nn.GroupNorm(8, 16), nn.GELU())
        self.convout = nn.Sequential(
            nn.Conv3d(16, 16, 3, 1, 1), nn.GELU(),
            nn.Conv3d(16, 8, 3, 1, 1), nn.GELU(),
            nn.Conv3d(8, 1, 1), nn.Tanh())

        self.up0 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up2 = nn.Upsample(size=(33, 33, 33), mode='trilinear', align_corners=False)

    def _sinusoidal_pos(self, n, device, dtype):
        pos = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(0, self.context_dim, 2, device=device, dtype=torch.float32)
        ang = pos / (10000.0 ** (i / self.context_dim))
        pe = torch.zeros(n, self.context_dim, device=device, dtype=torch.float32)
        pe[:, 0::2] = ang.sin()
        pe[:, 1::2] = ang.cos()
        return pe.to(dtype)

    def se3_film(self, x, cond_proj, cond):
        """SE(3)-equivariant FiLM: scalar shift+scale, vector scale only."""
        params = cond_proj(cond)
        B, C, D, H, W = x.shape
        n_s = self.scalar_ch * 2
        params_s = params[:, :n_s]
        params_v = params[:, n_s:n_s + self.vector_ch]
        x_s = x[:, :self.scalar_ch]
        gamma_s, beta_s = params_s.chunk(2, dim=1)
        x_s = x_s * (1.0 + gamma_s.view(B, self.scalar_ch, 1, 1, 1)) \
              + beta_s.view(B, self.scalar_ch, 1, 1, 1)
        x_v = x[:, self.scalar_ch:].view(B, self.vector_ch, 3, D, H, W)
        x_v = x_v * (1.0 + params_v.view(B, self.vector_ch, 1, 1, 1, 1))
        x_v = x_v.view(B, self.vector_ch * 3, D, H, W)
        return torch.cat([x_s, x_v], dim=1)

    def standard_film(self, x, cond_proj, cond):
        """Standard FiLM conditioning (shift + scale)."""
        gamma_beta = cond_proj(cond)
        B, C = x.shape[:2]
        gamma, beta = gamma_beta.chunk(2, dim=1)
        return x * (1.0 + gamma.view(B, C, 1, 1, 1)) + beta.view(B, C, 1, 1, 1)

    def forward(self, x, global_context):
        xr = x.squeeze(1) if x.dim() == 3 else x
        gc = global_context.squeeze(1) if global_context.dim() == 3 else global_context
        if xr.shape[0] != gc.shape[0]:
            raise ValueError(f"residue count mismatch: x={xr.shape} gc={gc.shape}")
        B = xr.shape[0]

        latent = self.context_to_query(gc) + self.residue_to_query(xr)
        dynamic_latent = self.query_act(latent).view(B, self.scalar_ch, 4, 4, 4)
        grid_batch = self.coord_grid.unsqueeze(0).expand(B, -1, -1, -1, -1)
        query = torch.cat([dynamic_latent, grid_batch.to(dynamic_latent.dtype)], dim=1)
        cond = self.cond_fuse(torch.cat([gc, xr], dim=-1))
        ctx_res = self.ctx_norm(xr)
        if self.use_residue_pos:
            ctx_res = ctx_res + self._sinusoidal_pos(B, xr.device, xr.dtype)
        ctx_res = ctx_res + self.residue_pos_bias
        if B > self.max_ctx_residues:
            idx = torch.linspace(0, B - 1, self.max_ctx_residues,
                                 device=xr.device).long()
            ctx_res = ctx_res[idx]
        bos_row = self.ctx_norm(gc[:1])
        ctx = torch.cat([bos_row, ctx_res], dim=0)
        mha_context = ctx.unsqueeze(1).expand(-1, B, -1)

        x4 = self.sct1(query, mha_context)
        x4 = self.se3_film(x4, self.cond_proj_32, cond)
        x8 = self.up0(x4)
        x8 = self.sct2(x8)
        x8 = self.se3_film(x8, self.cond_proj_32, cond)
        skip8 = x8

        x16 = self.up1(x8)
        x16 = self.reduce_32_to_24(x16)
        x16 = self.refine_16(x16)
        x16 = self.standard_film(x16, self.cond_proj_24, cond)

        x33 = self.up2(x16)
        x33 = self.reduce_24_to_16(x33)
        x33 = self.refine_33(x33)
        x33 = self.standard_film(x33, self.cond_proj_16, cond)

        skip33 = self.up2(self.skip_reduce_24_to_16(
            self.skip_reduce_32_to_24(skip8)))
        return self.convout(x33 + skip33)


# ------------------------------------------------- RoPE sequence encoder

class RotaryPositionalEmbedding(nn.Module):
    """Cached rotary positional embedding."""

    def __init__(self, dim, max_seq_len=8192, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freqs", freqs)
        self.cache = None

    def forward(self, seq_len, device, dtype):
        if (self.cache is None or seq_len > self.cache.shape[0]
                or self.cache.device != device):
            t = torch.arange(seq_len, device=device, dtype=self.freqs.dtype)
            freqs = torch.outer(t, self.freqs)
            emb = freqs.repeat_interleave(2, dim=-1)
            self.cache = torch.stack([emb.cos(), emb.sin()], dim=-1)
        return self.cache[:seq_len].to(dtype=dtype)


def apply_rotary_pos_emb(x, sincos):
    """Apply RoPE to ``(B, n_heads, L, head_dim)``."""
    cos = sincos[..., 0].unsqueeze(0).unsqueeze(0)
    sin = sincos[..., 1].unsqueeze(0).unsqueeze(0)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    x_rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
    x_original = torch.stack([x1, x2], dim=-1).flatten(-2)
    return (x_original * cos) + (x_rotated * sin)


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with rotary positional embeddings."""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_p = dropout
        self.resid_dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        sincos = self.rope(L, x.device, q.dtype)
        q = apply_rotary_pos_emb(q, sincos)
        k = apply_rotary_pos_emb(k, sincos)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )
        out = out.to(dtype=x.dtype)
        out = out.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
        return self.resid_dropout(self.out_proj(out))


class RoPETransformerLayer(nn.Module):
    """Pre-norm transformer layer with RoPE attention."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = RoPEMultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        x_norm = self.norm1(x)
        attn_out = self.self_attn(x_norm, mask=mask)
        x = x + attn_out
        x_norm = self.norm2(x)
        ff_out = self.feed_forward(x_norm)
        x = x + ff_out
        return x


# ------------------------------------------------- representation collapse

def _center(x, mask=None, center='global'):
    """Centre and pool rows for a rank/collapse statistic.

    Parameters
    ----------
    center : ``'global'`` | ``'within'`` | ``'none'``
        ``'global'``: subtract mean over all rows.
        ``'within'``: subtract each sequence's own mean, then pool
        residuals.  Requires ``(B, L, D)`` input.
        ``'none'``: valid rows, uncentred (for the share term).

    Returns ``(rows, dof, n_groups)``.  *dof* accounts for the centring:
    ``N-1`` pooled, ``N-B`` after removing one mean per sequence.
    Sequences with fewer than 2 valid rows are dropped.
    """
    if center not in ('global', 'within', 'none'):
        raise ValueError(
            f"center must be 'global', 'within' or 'none', got {center!r}")

    if center == 'none':
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        if mask is not None:
            x = x[mask.reshape(-1).bool()]
        return x, max(x.shape[0] - 1, 0), 1

    if center == 'global':
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        if mask is not None:
            x = x[mask.reshape(-1).bool()]
        n = x.shape[0]
        if n < 2:
            return x[:0], 0, 0
        return x - x.mean(0, keepdim=True), n - 1, 1

    if x.dim() != 3:
        raise ValueError(f"center='within' needs (B, L, D), got {tuple(x.shape)}")
    B, Lx, D = x.shape
    m = (x.new_ones(B, Lx) if mask is None
         else mask.reshape(B, Lx).to(x.dtype))
    cnt = m.sum(1)
    keep = cnt >= 2
    if not bool(keep.any()):
        return x.new_zeros(0, D), 0, 0
    x, m, cnt = x[keep], m[keep], cnt[keep]
    mu = (x * m.unsqueeze(-1)).sum(1, keepdim=True) / cnt.view(-1, 1, 1)
    r = (x - mu) * m.unsqueeze(-1)
    rows = r.reshape(-1, D)[m.reshape(-1).bool()]
    n, b = int(cnt.sum().item()), int(keep.sum().item())
    return rows, max(n - b, 0), b


def effective_rank(x, eps=1e-12, mask=None, center='global',
                   amp_floor=1e-5):
    """Participation ratio of the centred ``(N, D)`` matrix.

    Not algebraic rank -- it discounts small singular values, so a true
    rank-K subspace reads below K.  Use ``center='within'`` for per-residue
    decoder diagnostics.
    """
    a, dof, _ = _center(x, mask=mask, center=center)
    if dof < 1 or a.shape[0] < 2:
        return (x if isinstance(x, torch.Tensor) else a).new_zeros(())
    a = a.float()
    amp = a.pow(2).mean().sqrt()
    if float(amp) < amp_floor:
        return a.new_zeros(())
    a = a / amp
    s2 = torch.linalg.svdvals(a) ** 2
    return s2.sum() ** 2 / s2.pow(2).sum().clamp(min=eps)


def collapse_penalty(z, target_std=1.0, cov_weight=1.0, mask=None,
                     eps=1e-6, center='within', prefix='collapse',
                     target_share=0.25, share_weight=1.0, pr_target=None):
    """Guard against representation collapse.  Three bounded, rotation-invariant
    terms on the within-sequence residual ``Zc = z - mean_L(z)``:

    * **scale** -- ``relu(target_std - rms(Zc))``: amplitude floor.
    * **spread** -- log deficit of the participation ratio against a
      reference (``pr_target`` or the isotropy ceiling): rank floor.
    * **share** -- ``relu(target_share - ms(Zc)/ms(z))``: fraction of the
      raw mean square that varies with the residue, which is the only term
      sensitive to a protein-level constant.

    Parameters
    ----------
    z : Tensor, ``(B, L, D)`` or ``(N, D)``
    mask : Tensor, optional
        Valid positions.  Pass it -- padding rows are near-duplicates that
        distort the within-mean.
    pr_target : float or None
        Participation-ratio floor.  None uses the isotropy ceiling
        ``D * dof / (dof + D)``.

    Returns ``(loss, metrics)``.  Metrics are detached and keyed with *prefix*.
    """
    dev = z.device.type if isinstance(z, torch.Tensor) else 'cpu'
    with torch.amp.autocast(dev, enabled=False):
        zc, dof, _ = _center(z, mask=mask, center=center)
        zero = z.new_zeros(())
        met = lambda **kw: {f'{prefix}_{k}': v for k, v in kw.items()}
        if dof < 1 or zc.shape[0] < 2:
            return z.sum() * 0.0, met(scale=zero, spread=zero, share=zero,
                                      pr=zero, n=zero)

        D = zc.shape[-1]
        zc = zc.float()
        ms_within = zc.pow(2).sum() / (zc.shape[0] * D)
        rms = torch.sqrt(ms_within * zc.shape[0] / max(dof, 1) + eps)
        scale_loss = F.relu(target_std - rms)

        # Trace-normalise before squaring so the clamp can't bind above
        # rms ~1e-4 and PR is scale-invariant.
        zn = zc / rms.clamp(min=1e-4)
        Cn = (zn.T @ zn) / dof
        pr = Cn.diagonal().sum().pow(2) / Cn.pow(2).sum().clamp(min=1e-12)
        iso    = max(D * dof / (dof + D), 1.0 + 1e-6)
        pr_ref = float(pr_target) if pr_target is not None else iso
        pr_ref = max(min(pr_ref, iso), 1.0 + 1e-6)
        spread_loss = F.relu(torch.log(pr_ref / pr.clamp(min=1.0))
                             / math.log(pr_ref))

        # Share: raw, uncentred mean square -- what a protein-level
        # constant can't hide from.
        zr, _, _ = _center(z, mask=mask, center='none')
        ms_total = zr.float().pow(2).sum() / (max(zr.shape[0], 1) * D) + eps
        share = ms_within / ms_total
        share_loss = F.relu(target_share - share) / max(target_share, 1e-6)

        loss = (scale_loss + cov_weight * spread_loss
                + share_weight * share_loss)
        return loss, met(scale=scale_loss.detach(),
                         spread=spread_loss.detach(),
                         share=share.detach(), pr=pr.detach(),
                         n=torch.tensor(float(zc.shape[0]), device=z.device))


class InteractionTokenBlock(nn.Module):
    """Residues pool into shared token summaries, then read back from them.

    The attention branch is effectively a per-protein constant generator
    on this encoder (residue share of branch output ~1e-9 across all tested
    configurations).  Residue identity reaches the decoder through the
    residual only, and through the identity skip when enabled.

    Run ``preflight.py`` before changing any of these defaults.
    """

    def __init__(self, d_model=64, n_tokens=32, n_heads=2, layer_scale=0.3,
                 token_init=5.0):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(1, n_tokens, d_model)
                                   * token_init)
        self.pool_attn = nn.MultiheadAttention(d_model, n_heads,
                                               batch_first=True)
        self.read_attn = nn.MultiheadAttention(d_model, n_heads,
                                               batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.gamma_attn = nn.Parameter(layer_scale * torch.ones(d_model))
        self.gamma_ff = nn.Parameter(layer_scale * torch.ones(d_model))

    def forward(self, z):
        t = self.tokens.expand(z.shape[0], -1, -1).to(z.dtype)
        zn = self.norm1(z)
        t, _ = self.pool_attn(t, zn, zn)
        attended, _ = self.read_attn(self.norm2(z), t, t)
        z = z + self.gamma_attn * attended
        z = z + self.gamma_ff * self.ff(self.norm3(z))
        return z


# --------------------------------------------------------- sequence encoder

class TransformerStackEnc(nn.Module):
    """RoPE transformer encoder with register tokens.

    Input: ``(B, L)`` token ids with BOS at position 0.
    Output: ``(B, 1+L_body, D)`` with BOS at position 0, registers removed.
    """

    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 64,
        num_layers: int = 6,
        nhead: int = 8,
        dropout: float = 0.1,
        batch_first: bool = True,
        vocab_size: int = 8003,
        num_reg: int = 4
    ):
        super().__init__()
        self.num_reg = num_reg
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.layers1 = nn.ModuleList([
            RoPETransformerLayer(
                d_model=input_dim,
                nhead=nhead,
                dim_feedforward=input_dim * 2,
                dropout=dropout
            ) for _ in range(num_layers // 2)
        ])
        self.registers = nn.Parameter(torch.zeros(1, num_reg, input_dim))
        nn.init.normal_(self.registers, std=0.02)
        e2l = (num_layers // 2) + 1 if num_layers % 2 != 0 else num_layers // 2
        self.layers2 = nn.ModuleList([
            RoPETransformerLayer(
                d_model=output_dim,
                nhead=nhead,
                dim_feedforward=output_dim * 2,
                dropout=dropout
            ) for _ in range(e2l)
        ])

    def forward(self, x: torch.Tensor):
        b, l = x.shape
        x = self.embedding(x)
        bos = x[:, 0:1, :]
        sentence = x[:, 1:, :]
        regs = self.registers.expand(b, -1, -1).to(x.dtype)
        x = torch.cat([bos, regs, sentence], dim=1)
        for layer in self.layers1:
            if self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        for layer in self.layers2:
            if self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        word_latents = x[:, self.num_reg + 1:, :]
        bos_latent = x[:, 0:1, :]
        return torch.cat([bos_latent, word_latents], dim=1)


# ------------------------------------------------------ bias initialisation

def compute_and_initialize_biases(
    train_loader,
    mlm_head,
    convout_module=None,
    vocab_size=8003,
    device="cuda"
):
    """Single-pass computation of the MLM unigram bias and reconstruction mean."""
    print("Computing unigram + reconstruction biases in single pass...")
    counts = torch.zeros(vocab_size, dtype=torch.long)
    total_tokens = 0
    running_mean = 0.0
    total_voxels = 0

    for batch in tqdm(train_loader, desc="Unified Bias Init"):
        if batch is None:
            continue
        tdat = batch[0]
        if not torch.isfinite(tdat).all():
            nan_count = (~torch.isfinite(tdat)).sum().item()
            print(f"Warning: Found {nan_count} non-finite voxels. Skipping them.")
            print(f"to remove: {tdat[-1]}")
        batch_count = tdat.numel()
        batch_mean = tdat.mean().item()
        new_total = total_voxels + batch_count
        running_mean += (batch_mean - running_mean) * (batch_count / new_total)
        total_voxels = new_total
        if len(batch) >= 5:
            full_seqs = batch[4]
            tokens = full_seqs.view(-1)
            valid_mask = tokens < (vocab_size - 3)
            valid_tokens = tokens[valid_mask]
            if valid_tokens.numel() > 0:
                counts += torch.bincount(valid_tokens.cpu(), minlength=vocab_size)
                total_tokens += valid_tokens.numel()

    probs = counts.float() / (total_tokens + 1e-8)
    log_probs = torch.log(probs + 1e-8)
    with torch.no_grad():
        mlm_head.bias.data.copy_(log_probs.to(device))
    entropy = -(probs * torch.log(probs + 1e-12)).sum().item()

    print(f"MLM baseline entropy: {entropy:.4f}")
    print(f"Total tokens processed: {total_tokens}")
    print(f"Unique tokens with counts > 0: {(counts > 0).sum().item()}")
    print(f"Reconstruction mean potential: {running_mean:.4f}")
    print(f"Total voxels processed: {total_voxels}")

    return {
        "mlm_entropy": entropy,
        "recon_mean_bias": running_mean,
        "total_tokens": total_tokens,
        "total_voxels": total_voxels,
    }


# -------------------------------------------------------- factorized MLM head

class FactorizedMLMHead(nn.Module):
    """Project latent -> embedding dim, then tie to the embedding matrix."""

    def __init__(self, embedding, latent_dim=64):
        super().__init__()
        vocab_size, embed_dim = embedding.weight.shape
        self._tied = [embedding]
        self.dense = nn.Linear(latent_dim, embed_dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(embed_dim)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        nn.init.normal_(self.dense.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.dense.bias)

    def forward(self, z):
        x = self.norm(self.act(self.dense(z)))
        return F.linear(x, self._tied[0].weight) + self.bias
