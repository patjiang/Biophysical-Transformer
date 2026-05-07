import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from .blocks import *

class VoxelizeTrunk(nn.Module):
    def __init__(self, context_dim=64, tot_layers=3):
        super().__init__()
        self.scalar_ch = 29
        self.vector_ch = 1
        self.context_to_query = nn.Sequential(
            nn.Linear(context_dim, self.scalar_ch * 4 * 4 * 4),
            nn.GELU()
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
        self.sct1 = SE3CrossTransformer(
            scalar_ch=self.scalar_ch, 
            vector_ch=self.vector_ch, 
            dimprod_scalar=32, 
            dimprod_vector=8, 
            context_dim=context_dim, 
            num_layers=1
        )
        self.sct2 = SE3Transformer(
            scalar_ch=self.scalar_ch, 
            vector_ch=self.vector_ch, 
            dimprod_scalar=32, 
            dimprod_vector=8, 
            num_layers=1
        )
        self.reduce_32_to_24 = nn.Conv3d(32, 24, 1) 
        self.reduce_24_to_16 = nn.Conv3d(24, 16, 1)
        self.refine_16 = nn.Sequential(
            nn.Conv3d(24, 24, 3, 1, 1),
            nn.GroupNorm(8, 24),
            nn.GELU()
        )
        self.refine_33 = nn.Sequential(
            nn.Conv3d(16, 16, 3, 1, 1),
            nn.GroupNorm(8, 16),
            nn.GELU()
        )
        self.convout = nn.Sequential(
            nn.Conv3d(16, 16, 3, 1, 1),
            nn.GELU(),
            nn.Conv3d(16, 8, 3, 1, 1),
            nn.GELU(),
            nn.Conv3d(8, 1, 1),
            nn.Tanh()
        )
        self.up0 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up2 = nn.Upsample(size=(33, 33, 33), mode='trilinear', align_corners=False)

    def se3_film(self, x, cond_proj, cond):
        if cond.dim() == 3 and cond.size(1) == 1:
            cond = cond.squeeze(1)
        params = cond_proj(cond)
        B, C, D, H, W = x.shape
        num_scalar_params = self.scalar_ch * 2
        num_vector_params = self.vector_ch
        params_s = params[:, :num_scalar_params]
        params_v = params[:, num_scalar_params:num_scalar_params + num_vector_params]
        x_s = x[:, :self.scalar_ch]
        gamma_s, beta_s = params_s.chunk(2, dim=1)
        gamma_s = gamma_s.view(B, self.scalar_ch, 1, 1, 1)
        beta_s = beta_s.view(B, self.scalar_ch, 1, 1, 1)
        x_s = x_s * (1.0 + gamma_s) + beta_s
        x_v = x[:, self.scalar_ch:]
        x_v = x_v.view(B, self.vector_ch, 3, D, H, W)
        gamma_v = params_v.view(B, self.vector_ch, 1, 1, 1, 1)
        x_v = x_v * (1.0 + gamma_v)
        x_v = x_v.view(B, self.vector_ch * 3, D, H, W)
        return torch.cat([x_s, x_v], dim=1)

    def standard_film(self, x, cond_proj, cond):
        if cond.dim() == 3 and cond.size(1) == 1:
            cond = cond.squeeze(1)
        gamma_beta = cond_proj(cond)
        B, C = x.shape[:2]
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.view(B, C, 1, 1, 1)
        beta = beta.view(B, C, 1, 1, 1)
        return x * (1.0 + gamma) + beta

    def forward(self, x, global_context):
        B = global_context.shape[0]
        dynamic_latent = self.context_to_query(global_context)
        dynamic_latent = dynamic_latent.view(B, self.scalar_ch, 4, 4, 4)
        grid_batch = self.coord_grid.unsqueeze(0).expand(B, -1, -1, -1, -1)
        query = torch.cat([dynamic_latent, grid_batch], dim=1)
        if global_context.dim() == 2:
            mha_context = global_context.unsqueeze(0) 
        else:
            mha_context = global_context.permute(1, 0, 2)
        x4 = self.sct1(query, mha_context) 
        x4 = self.se3_film(x4, self.cond_proj_32, global_context)
        x8 = self.up0(x4)
        x8 = self.sct2(x8)
        x8 = self.se3_film(x8, self.cond_proj_32, global_context)
        skip8 = x8 
        x16 = self.up1(x8)
        x16 = self.reduce_32_to_24(x16)
        x16 = self.refine_16(x16)
        x16 = self.standard_film(x16, self.cond_proj_24, global_context)
        x33 = self.up2(x16)
        x33 = self.reduce_24_to_16(x33)
        x33 = self.refine_33(x33)
        x33 = self.standard_film(x33, self.cond_proj_16, global_context)
        skip8_std = self.reduce_32_to_24(skip8)
        skip8_reduced = self.reduce_24_to_16(skip8_std)
        skip33 = self.up2(skip8_reduced)
        x33 += skip33
        out = self.convout(x33)
        return out

class FactorizedMLMHead(nn.Module):
    def __init__(self, embedding_weight, latent_dim=64):
        super().__init__()
        self.embedding_weight = embedding_weight 
        embed_dim = embedding_weight.size(1)
        
        self.projection = nn.Linear(latent_dim, embed_dim)
        self.norm = nn.LayerNorm(latent_dim)
        
        self.bias = nn.Parameter(torch.zeros(embedding_weight.size(0)))
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.projection.bias)

    def forward(self, z):
        x = self.norm(z)
        x = self.projection(x) 
        
        logits = F.linear(x, self.embedding_weight) + self.bias
        return logits

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=8192, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freqs", freqs)
        self.cache = None

    def forward(self, x):
        seq_len = x.shape[1]
        if self.cache is None or seq_len > self.cache.shape[0]:
            t = torch.arange(seq_len, device=x.device, dtype=self.freqs.dtype)
            freqs = torch.outer(t, self.freqs)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cache = torch.stack([emb.cos(), emb.sin()], dim=-1)
        return self.cache[:seq_len].to(dtype=x.dtype)

def apply_rotary_pos_emb_latent(x, sincos):
    cos = sincos[..., 0]
    sin = sincos[..., 1]
    
    if x.dim() == 4:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    x_rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
    x_original = torch.stack([x1, x2], dim=-1).flatten(-2)
    return (x_original * cos) + (x_rotated * sin)

class TuckerAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, tucker_ranks=(8, 128, 64), dropout=0.1, shared_kv=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.shared_kv = shared_kv
        r1, r2, r3 = tucker_ranks
        self.U1 = nn.Parameter(torch.Tensor(num_heads, r1))
        self.U2 = nn.Parameter(torch.Tensor(embed_dim, r2))
        self.U3 = nn.Parameter(torch.Tensor(embed_dim, r3))
        self.C = nn.Parameter(torch.Tensor(r1, r2, r3))
        self.U1_tilde = nn.Parameter(torch.Tensor(num_heads, r1))
        self.U2_tilde = nn.Parameter(torch.Tensor(embed_dim, r2))
        if not shared_kv:
            self.U3_tilde = nn.Parameter(torch.Tensor(embed_dim, r3))
        self.C_tilde = nn.Parameter(torch.Tensor(r1, r2, r3))
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout_p = dropout
        self.rope = RotaryPositionalEmbedding(r3)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in [self.U1, self.U2, self.U3, self.C, self.U1_tilde, self.U2_tilde, self.C_tilde]:
            nn.init.normal_(p, std=0.02)
        if not self.shared_kv:
            nn.init.normal_(self.U3_tilde, std=0.02)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        X_U2 = x @ self.U2
        K = x @ self.U3
        Q = torch.einsum('ijk, hi, blj -> bhlk', self.C, self.U1, X_U2)
        sincos = self.rope(K)
        Q = apply_rotary_pos_emb_latent(Q, sincos)
        K = apply_rotary_pos_emb_latent(K, sincos)
        scores = torch.einsum('bhlk, bmk -> bhlm', Q, K) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn = F.softmax(scores, dim=-1)
        if self.training and self.dropout_p > 0.0:
            attn = F.dropout(attn, p=self.dropout_p)
        V = K if self.shared_kv else (x @ self.U3_tilde)
        Y = torch.einsum('bhlm, bmk -> bhlk', attn, V)
        out = torch.einsum('bhlk, hi, ijk, dj -> bld', Y, self.U1_tilde, self.C_tilde, self.U2_tilde)
        return self.resid_dropout(out)
        
class RoPEMultiheadAttention(nn.Module):
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
        
        sincos = self.rope(q)
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
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, use_tucker=False, tucker_ranks=(8, 128, 64)):
        super().__init__()
        if use_tucker:
            self.self_attn = TuckerAttention(d_model, nhead, tucker_ranks=tucker_ranks, dropout=dropout)
        else:
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

class FactorizedSequenceEmbedding(nn.Module):
    def __init__(self, num_tokens=8002, factorized_dim=128, hidden_dim=256, max_len=5000):
        super().__init__()
        self.token_embedding = nn.Embedding(num_tokens, factorized_dim)
        self.projection = nn.Linear(factorized_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)

    def forward(self, x):
        tokens = self.token_embedding(x)
        h = self.projection(tokens)
        return self.dropout(self.norm(h))

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


class TransformerStackEnc(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 64,
        num_layers: int = 6,
        nhead: int = 8,
        dropout: float = 0.1,
        batch_first: bool = True,
        vocab_size: int = 8003,
        num_reg = 4 
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
        self.input_projection = nn.Linear(input_dim, output_dim)
        self.registers = nn.Parameter(torch.zeros(1, num_reg, input_dim))
        nn.init.normal_(self.registers, std=0.02)
        if(num_layers % 2 != 0):
            e2l = (num_layers // 2) + 1
        else:
            e2l = num_layers // 2
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
        x = self.embedding(x).to(dtype=torch.bfloat16)
        bos = x[:, 0:1, :]
        sentence = x[:, 1:, :]
        regs = self.registers.expand(b, -1, -1).to(x.dtype)
        x = torch.cat([bos, regs, sentence], dim=1)
        for layer in self.layers1:
            if self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
                
        x = self.input_projection(x)
        
        for layer in self.layers2:
            if self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
            
        word_latents = x[:, self.num_reg + 1:, :]
        bos_latent = x[:, 0:1, :]
        return torch.cat([bos_latent, word_latents], dim=1)



class SingleScaleDiscriminator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        
        def conv_block(in_c, out_c, k, s, p, use_norm=True):
            layers = [nn.Conv3d(in_c, out_c, k, s, p)]
            if use_norm:
                layers.append(nn.InstanceNorm3d(out_c, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            conv_block(in_channels, 32, 4, 2, 1, use_norm=False),
            conv_block(32, 64, 4, 2, 1, use_norm=True),
            conv_block(64, 128, 4, 2, 1, use_norm=True),
            conv_block(128, 64, 3, 1, 1, use_norm=True),
            nn.Conv3d(64, 1, 3, 1, 1)
        )

    def forward(self, x):
        return self.model(x)

class PatchDiscriminator3D(nn.Module):
    def __init__(self, in_channels=1, num_scales=2):
        super().__init__()
        self.num_scales = num_scales
        
        self.scales = nn.ModuleList([
            SingleScaleDiscriminator(in_channels) for _ in range(num_scales)
        ])
        
        self.downsample = nn.AvgPool3d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        B = x.shape[0]
        logits = []
        
        for i, model in enumerate(self.scales):
            if i > 0:
                x = self.downsample(x)
            
            out = model(x)
            logits.append(out.view(B, -1))
        
        return torch.cat(logits, dim=1)

class PatchDiscriminator3D_sm(nn.Module):
    def __init__(self, in_channels=1, num_scales=1):
        super().__init__()
        self.num_scales = num_scales

        def make_scale(in_c):
            return nn.Sequential(
                nn.Conv3d(in_c,  32, 4, 2, 1, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d( 32,  64, 4, 2, 1, bias=False),
                nn.InstanceNorm3d(64, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d( 64,   1, 3, 1, 1),
            )

        self.scales = nn.ModuleList([make_scale(in_channels) for _ in range(num_scales)])
        self.downsample = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        out = []
        for i, model in enumerate(self.scales):
            if i > 0:
                x = self.downsample(x)
            out.append(model(x).flatten(1))
        return torch.cat(out, dim=1)
