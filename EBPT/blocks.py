import torch
import torch.nn as nn
import torch.nn.functional as F
from se3cnn.image.convolution import SE3Convolution

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
        vector_gate_expanded = vector_gate.unsqueeze(2)
        gated_vector_v = (vector_v_reshaped * vector_gate_expanded).view(B, -1, H, W, D)
        gated_v = torch.cat([gated_scalar_v, gated_vector_v], dim=1)
        out_feat = self.pw_conv_ffn(gated_v)
        return x + out_feat

class ConvFeedForward(nn.Module):
    def __init__(self, scalar_ch, vector_ch, ffn_mult, dropout):
        self.ffn_s = nn.Sequential(
            nn.Conv3d(scalar_ch, scalar_ch*ffn_mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv3d(scalar_ch*ffn_mult, scalar_ch, 1)
        )
        self.ffn_v = nn.Sequential(
            nn.Conv3d(vector_ch*3, vector_ch*3*ffn_mult, 1, groups=vector_ch),
            nn.Conv3d(vector_ch*3*ffn_mult, vector_ch*3, 1, groups=vector_ch)
        )
        
def FeedForward(dim, mult, dropout):
    return nn.Sequential(
        nn.Linear(dim, dim*mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim*mult, dim))

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
            args = (
                x,
                layer['attn'],
                layer['ffn'],
                layer['norm_s'],
                layer['norm_v'],
                self.scalar_ch
            )

            if self.training:
                x = torch.utils.checkpoint.checkpoint(
                    _layer_fwd, *args, use_reentrant=False
                )
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
                    scalar_ch,
                    vector_ch,
                    dimprod_scalar,
                    dimprod_vector
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
                'ffn': EquivariantFeedForward(
                    scalar_ch,
                    vector_ch,
                    ffn_mult,
                    dropout
                )
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
                x,
                context,
                layer['se3_self_attn'],
                layer['mha'],
                layer['ffn'],
                layer['norm_cross_s'],
                layer['norm_cross_v'],
                layer['norm_ffn_s'],
                layer['norm_ffn_v'],
                self.scalar_ch
            )

            if self.training:
                x = torch.utils.checkpoint.checkpoint(
                    _layer_fwd, *args, use_reentrant=False
                )
            else:
                x = _layer_fwd(*args)

        return x

class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_ch, in_ch, kernel_size=kernel_size,
            padding=padding, groups=in_ch
        )
        self.pointwise = nn.Conv3d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
