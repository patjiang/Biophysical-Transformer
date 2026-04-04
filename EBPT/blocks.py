import torch
import torch.nn as nn
import torch.nn.functional as F
from se3cnn.image.convolution import SE3Convolution

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

class ConvolutionalSelfAttention3d(nn.Module):
    def __init__(self, in_ch, dimprod, scale_reduce=None):
        super().__init__()
        self.dw_conv = nn.Conv3d(in_ch, in_ch, 5, padding=2)
        if scale_reduce is None:
            self.pw_conv_1 = nn.Conv3d(in_ch, dimprod, 1)
            self.pw_conv_2 = nn.Sequential(
                nn.Conv3d(dimprod, in_ch, 1),
                nn.Sigmoid()
            )
        else:
            scaled_hw = (dimprod // (scale_reduce**3))
            self.pw_conv_1 = nn.Sequential(
                nn.AvgPool3d(scale_reduce),
                nn.Conv3d(in_ch, scaled_hw, 1)
            )
            self.pw_conv_2 = nn.Sequential(
                nn.Conv3d(scaled_hw, in_ch, 1),
                nn.Sigmoid(),
                nn.Upsample(scale_factor=scale_reduce)
            )
        self.pw_conv_3 = nn.Sequential(
            nn.Conv3d(in_ch, 3*in_ch, 1),
            nn.Conv3d(3*in_ch, in_ch, 1)
        )

    def forward(self, x):
        q, k, v = self.dw_conv(x), self.dw_conv(x), self.dw_conv(x)
        q, k = self.pw_conv_1(q), self.pw_conv_1(k)
        # Spatial correlation proxy
        k = k.flatten(start_dim=1).T.reshape(q.shape)
        qk = self.pw_conv_2((q * k))
        qkv = v * qk
        out_feat = self.pw_conv_3(qkv)
        return x + out_feat

class SE3ConvolutionalSelfAttention3d(nn.Module):
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
        vector_gate_expanded = vector_gate.unsqueeze(2) # (B, vector_ch, 1, H, W, D)
        
        gated_vector_v = (vector_v_reshaped * vector_gate_expanded).view(B, -1, H, W, D)
        
        gated_v = torch.cat([gated_scalar_v, gated_vector_v], dim=1)
        out_feat = self.pw_conv_ffn(gated_v)
        
        return x + out_feat

class ConvFeedForward(nn.Module):
    def __init__(self, scalar_ch, vector_ch, ffn_mult, dropout):
        super().__init__()
        self.scalar_ch = scalar_ch
        self.vector_ch = vector_ch
        
        self.ffn_s = nn.Sequential(
            nn.Conv3d(scalar_ch, scalar_ch * ffn_mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv3d(scalar_ch * ffn_mult, scalar_ch, 1)
        )
        self.ffn_v = nn.Sequential(
            nn.Conv3d(vector_ch * 3, vector_ch * 3 * ffn_mult, 1, groups=vector_ch),
            nn.GELU(),
            nn.Conv3d(vector_ch * 3 * ffn_mult, vector_ch * 3, 1, groups=vector_ch)
        )

    def forward(self, x):
        x_s = x[:, :self.scalar_ch]
        x_v = x[:, self.scalar_ch:]
        return torch.cat([self.ffn_s(x_s), self.ffn_v(x_v)], dim=1)

class EquivariantFeedForward(nn.Module):
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
        
        # Scalar branch
        h_s = self.act(self.s_up(x_s))
        
        # Vector branch (Equivariant via Einstein Summation)
        B, _, D, H, W = x_v.shape
        x_v_view = x_v.view(B, self.vector_ch, 3, D, H, W)
        h_v = torch.einsum('bicdhw, oi -> bocdhw', x_v_view, self.v_up_weight)
        
        # Gated Linear Unit variant for vectors
        gates = torch.sigmoid(self.gate_proj(h_s))
        h_v_gated = h_v * gates.unsqueeze(2)
        
        out_v_view = torch.einsum('bicdhw, oi -> bocdhw', h_v_gated, self.v_down_weight)
        out_v = out_v_view.reshape(B, self.vector_ch * 3, D, H, W)
        
        out_s = self.s_down(h_s)
        return torch.cat([self.dropout_s(out_s), out_v], dim=1)
