import torch
import torch.nn.functional as F
from blocks import *

class SE3Transformer(nn.Module):
    def __init__(self, channels, se3_dimprod, num_layers = 1, ffn_mult = 2, dropout = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                    SE3ConvolutionalSelfAttention3d(channels, se3_dimprod),
                    ConvFeedForward(channels, ffn_mult, dropout),
                    nn.InstanceNorm3d(channels),
                    nn.GELU()
                ]))

    def forward(self, x):
        def _layer_fwd(x, attention, ffn, norm, activation):
            x = attention(x)
            x = x + ffn(x)
            x = norm(x)
            x = activation(x)
            return x

        for attention, ffn, norm, activation in self.layers:
            if self.training:
                x.requires_grad_(True)
                x = torch.utils.checkpoint.checkpoint(_layer_fwd, x, attention, ffn, norm, activation, use_reentrant=False)
            else:
                x = _layer_fwd(x, attention, ffn, norm, activation)
        return x

class SE3CrossTransformer(nn.Module):
    def __init__(self, channels, se3_dimprod, num_layers = 1, ffn_mult = 2, dropout = 0.1, z_dim=64, n_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                    SE3ConvolutionalSelfAttention3d(
                        channels, 
                        se3_dimprod
                    ),
                    nn.MultiheadAttention(
                        embed_dim=channels,
                        kdim=z_dim,
                        vdim=z_dim,
                        num_heads=n_heads,
                        batch_first=False
                    ),
                    ConvFeedForward(channels, ffn_mult, dropout),
                    nn.LayerNorm(channels),
                    nn.InstanceNorm3d(channels)
                ]))

    def forward(self, x, context):
        def _layer_fwd(x, context, attention, mha, ffn, norm1, norm2):
            x = attention(x)
            B, C, D, H, W = x.shape
            query = x.flatten(2).permute(2, 0, 1)
            key, value = context.expand(-1, B, -1), context.expand(-1, B, -1)
            mha_o, _ = mha(norm1(query), key, value)
            x = query + mha_o
            x = x.permute(1, 2, 0).reshape(B, C, D, H, W)
            x = x + ffn(norm2(x))
            return x

        for attention, mha, ffn, norm1, norm2 in self.layers:
            if self.training:
                x.requires_grad_(True)
                x = torch.utils.checkpoint.checkpoint(_layer_fwd, x, context, attention, mha, ffn, norm1, norm2, use_reentrant=False)
            else:
                x = _layer_fwd(x, context, attention, mha, ffn, norm1, norm2)
        return x

class VoxelizeTrunk(nn.Module):
    def __init__(self, tot_layers=3):
        super().__init__()
        self.context_to_query = nn.Sequential(
            nn.Linear(64, 29 * 4 * 4 * 4),
            nn.GELU()
        )
        grid = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, 4),
            torch.linspace(-1, 1, 4),
            torch.linspace(-1, 1, 4),
            indexing='ij'
        ))
        self.register_buffer('coord_grid', grid)
        self.sct1 = SE3CrossTransformer(32, 64, num_layers=tot_layers // 2)
        self.up0 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        if tot_layers % 2 == 1:
            ncl = 1 + (tot_layers // 2)
        else:
            ncl = tot_layers // 2
        self.sct2 = SE3Transformer(32, 512, num_layers=2)
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.refine_16 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.GELU()
        )
        self.up2 = nn.Upsample(size=(33, 33, 33), mode='trilinear', align_corners=False)
        self.refine = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.GELU()
        )
        self.convout = nn.Sequential(
            nn.Conv3d(32, 1, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x, global_context):
        B = x.shape[0]
        dynamic_latent = self.context_to_query(global_context)
        dynamic_latent = dynamic_latent.view(B, 29, 4, 4, 4)
        query = torch.cat([
            dynamic_latent,
            self.coord_grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)
        ], dim=1)
        x = self.sct1(query, x)
        x = self.up0(x)
        x = self.sct2(x)
        x = self.up1(x)
        x = self.refine_16(x)
        x = self.up2(x)
        x = self.refine(x)
        x = self.convout(x)
        return x

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

def apply_rotary_pos_emb(x, sincos):
    cos = sincos[..., 0].unsqueeze(0).unsqueeze(2)
    sin = sincos[..., 1].unsqueeze(0).unsqueeze(2)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    x_rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
    x_original = torch.stack([x1, x2], dim=-1).flatten(-2)
    return (x_original * cos) + (x_rotated * sin)

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

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out = self.self_attn(x_norm)
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
        vocab_size: int = 8002 
    ):
        super().__init__()
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
        x = self.embedding(x).to(dtype=torch.bfloat16)
        for layer in self.layers1:
            if self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        x = self.input_projection(x)
        for layer in self.layers2:
            if self.training:
                output = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                output = layer(x)
            x = output
        return output

class PatchDiscriminator3D(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(in_channels, 16, 4, 2, 1)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv3d(16, 8, 4, 2, 1)),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv3d(8, 1, 4, 1, 0))
        )
    
    def forward(self, x):
        return self.model(x)
