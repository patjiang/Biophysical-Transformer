import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from .blocks import *

class SE3Transformer(nn.Module):
    def __init__(self, scalar_ch, vector_ch, dimprod_scalar, dimprod_vector,
                 num_layers=1, ffn_mult=2, dropout=0.1):
        super().__init__()
        self.scalar_ch = scalar_ch
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'attn': SE3ConvolutionalSelfAttention3d(
                    scalar_ch,
                    vector_ch,
                    dimprod_scalar,
                    dimprod_vector
                ),
                'norm_s': nn.InstanceNorm3d(scalar_ch, affine=True),
                'norm_v': nn.InstanceNorm3d(vector_ch * 3, affine=True),
                'ffn': EquivariantFeedForward(
                    scalar_ch,
                    vector_ch,
                    ffn_mult,
                    dropout
                )
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
        def _layer_fwd(x, context, se3_attn, mha, ffn, norm_cs, norm_cv, norm_fs, norm_fv, s_ch):
            B, C, D, H, W = x.shape
            x = se3_attn(x)
            x_s, x_v = norm_cs(x[:, :s_ch]), norm_cv(x[:, s_ch:])
            x_normed = torch.cat([x_s, x_v], dim=1)
            
            query = x_normed.flatten(2).permute(2, 0, 1)
            mha_out, _ = mha(query, context, context)
            x = x + mha_out.permute(1, 2, 0).reshape(B, C, D, H, W)
            
            x_s, x_v = norm_fs(x[:, :s_ch]), norm_fv(x[:, s_ch:])
            x = x + ffn(torch.cat([x_s, x_v], dim=1))
            return x

        for layer in self.layers:
            args = (x, context, layer['se3_self_attn'], layer['mha'], layer['ffn'],
                    layer['norm_cross_s'], layer['norm_cross_v'], layer['norm_ffn_s'], layer['norm_ffn_v'], self.scalar_ch)
            if self.training:
                x = torch.utils.checkpoint.checkpoint(_layer_fwd, *args, use_reentrant=False)
            else:
                x = _layer_fwd(*args)
        return x

class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv3d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch)
        self.pointwise = nn.Conv3d(in_ch, out_ch, kernel_size=1)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))
        
class InteractionTokenBlock(nn.Module):
    def __init__(self, d_model=64, n_tokens=16, n_heads=2):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(1, n_tokens, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, z):
        B, L, C = z.shape
        t = self.tokens.expand(B, -1, -1)
        t, _ = self.attn(t, z, z)
        z, _ = self.attn(z, t, t)
        return self.ff(z)
        
class VoxelizeTrunk(nn.Module):
    def __init__(self, context_dim=64, tot_layers=3):
        super().__init__()
        self.scalar_ch, self.vector_ch = 29, 1
        self.context_to_query = nn.Sequential(nn.Linear(context_dim, self.scalar_ch * 64), nn.GELU())
        self.cond_proj_32 = nn.Linear(context_dim, 64)
        self.cond_proj_24 = nn.Linear(context_dim, 48)
        self.cond_proj_16 = nn.Linear(context_dim, 32)
        
        grid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 4), torch.linspace(-1, 1, 4), torch.linspace(-1, 1, 4), indexing='ij'))
        self.register_buffer('coord_grid', grid)
        
        self.sct1 = SE3CrossTransformer(self.scalar_ch, self.vector_ch, 32, 8, context_dim, num_layers=1)
        self.sct2 = SE3Transformer(self.scalar_ch, self.vector_ch, 32, 8, num_layers=1)
        
        self.reduce_32_to_24 = nn.Conv3d(32, 24, 1)
        self.reduce_24_to_16 = nn.Conv3d(24, 16, 1)
        self.refine_16 = nn.Sequential(nn.Conv3d(24, 24, 3, 1, 1), nn.GroupNorm(8, 24), nn.GELU())
        self.refine_33 = nn.Sequential(nn.Conv3d(16, 16, 3, 1, 1), nn.GroupNorm(8, 16), nn.GELU())
        self.convout = nn.Sequential(nn.Conv3d(16, 16, 3, 1, 1), nn.GELU(), nn.Conv3d(16, 8, 3, 1, 1), nn.GELU(), nn.Conv3d(8, 1, 1), nn.Tanh())
        
        self.up0 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up2 = nn.Upsample(size=(33, 33, 33), mode='trilinear', align_corners=False)

    def se3_film(self, x, cond_proj, cond):
        if cond.dim() == 3: cond = cond.squeeze(1)
        params = cond_proj(cond)
        B, _, D, H, W = x.shape
        gamma_s, beta_s = params[:, :self.scalar_ch*2].chunk(2, dim=1)
        x_s = x[:, :self.scalar_ch] * (1.0 + gamma_s.view(B, -1, 1, 1, 1)) + beta_s.view(B, -1, 1, 1, 1)
        gamma_v = params[:, self.scalar_ch*2:].view(B, self.vector_ch, 1, 1, 1, 1)
        x_v = (x[:, self.scalar_ch:].view(B, self.vector_ch, 3, D, H, W) * (1.0 + gamma_v)).view(B, -1, D, H, W)
        return torch.cat([x_s, x_v], dim=1)

    def standard_film(self, x, cond_proj, cond):
        if cond.dim() == 3: cond = cond.squeeze(1)
        gamma, beta = cond_proj(cond).chunk(2, dim=1)
        return x * (1.0 + gamma.view(x.size(0), -1, 1, 1, 1)) + beta.view(x.size(0), -1, 1, 1, 1)

    def forward(self, x, global_context):
        B = global_context.shape[0]
        query = torch.cat([self.context_to_query(global_context).view(B, self.scalar_ch, 4, 4, 4), 
                           self.coord_grid.unsqueeze(0).expand(B, -1, -1, -1, -1)], dim=1)
        
        mha_ctx = global_context.unsqueeze(0) if global_context.dim() == 2 else global_context.permute(1, 0, 2)
        x4 = self.se3_film(self.sct1(query, mha_ctx), self.cond_proj_32, global_context)
        x8 = self.se3_film(self.sct2(self.up0(x4)), self.cond_proj_32, global_context)
        
        x16 = self.standard_film(self.refine_16(self.reduce_32_to_24(self.up1(x8))), self.cond_proj_24, global_context)
        x33 = self.standard_film(self.refine_33(self.reduce_24_to_16(self.up2(x16))), self.cond_proj_16, global_context)
        
        skip33 = self.up2(self.reduce_24_to_16(self.reduce_32_to_24(x8)))
        return self.convout(x33 + skip33)

class TransformerStackEnc(nn.Module):
    def __init__(self, input_dim=512, output_dim=64, num_layers=6, nhead=8, dropout=0.1, vocab_size=8003, num_reg=4):
        super().__init__()
        self.num_reg = num_reg
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.registers = nn.Parameter(torch.zeros(1, num_reg, input_dim))
        nn.init.normal_(self.registers, std=0.02)
        
        self.layers1 = nn.ModuleList([RoPETransformerLayer(input_dim, nhead, input_dim*2, dropout) for _ in range(num_layers // 2)])
        self.input_projection = nn.Linear(input_dim, output_dim)
        e2l = (num_layers // 2) + (1 if num_layers % 2 != 0 else 0)
        self.layers2 = nn.ModuleList([RoPETransformerLayer(output_dim, nhead, output_dim*2, dropout) for _ in range(e2l)])

    def forward(self, x):
        b = x.shape[0]
        x = self.embedding(x).to(dtype=torch.bfloat16)
        x = torch.cat([x[:, 0:1, :], self.registers.expand(b, -1, -1).to(x.dtype), x[:, 1:, :]], dim=1)
        
        for layer in self.layers1:
            x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False) if self.training else layer(x)
        x = self.input_projection(x)
        for layer in self.layers2:
            x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False) if self.training else layer(x)
            
        return torch.cat([x[:, 0:1, :], x[:, self.num_reg + 1:, :]], dim=1)

class SingleScaleDiscriminator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        def cb(ic, oc, k, s, p, norm=True):
            l = [nn.Conv3d(ic, oc, k, s, p)]
            if norm: l.append(nn.InstanceNorm3d(oc, affine=True))
            l.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*l)
        self.model = nn.Sequential(cb(in_channels, 32, 4, 2, 1, False), cb(32, 64, 4, 2, 1), cb(64, 128, 4, 2, 1), cb(128, 64, 3, 1, 1), nn.Conv3d(64, 1, 3, 1, 1))
    def forward(self, x): return self.model(x)

class PatchDiscriminator3D(nn.Module):
    def __init__(self, in_channels=1, num_scales=2):
        super().__init__()
        self.scales = nn.ModuleList([SingleScaleDiscriminator(in_channels) for _ in range(num_scales)])
        self.downsample = nn.AvgPool3d(3, 2, 1)
    def forward(self, x):
        logits = []
        for i, model in enumerate(self.scales):
            if i > 0: x = self.downsample(x)
            logits.append(model(x).view(x.size(0), -1))
        return torch.cat(logits, dim=1)
