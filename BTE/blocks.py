import torch.nn as nn
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
    def __init__(self, in_ch, dimprod):
        super().__init__()
        self.dw_conv = SE3Convolution([(in_ch, 0, )], [(in_ch, 0, )], size=5, padding=2)
        self.pw_conv_1 = SE3Convolution(Rs_in=[(in_ch, 0, )], Rs_out=[(dimprod, 0, )], size=1)
        self.pw_conv_2 = nn.Sequential(
            SE3Convolution(Rs_in=[(dimprod, 0, )], Rs_out=[(in_ch, 0, )], size=1),
            nn.Sigmoid()
        )
        self.pw_conv_3 = nn.Sequential(
            SE3Convolution(Rs_in=[(in_ch, 0, )], Rs_out=[(3*in_ch, 0, )], size=1),
            SE3Convolution(Rs_in=[(3*in_ch, 0, )], Rs_out=[(in_ch, 0, )], size=1)
        )

    def forward(self, x):
        q, k, v = self.dw_conv(x), self.dw_conv(x), self.dw_conv(x)
        q, k = self.pw_conv_1(q), self.pw_conv_1(k)
        k = k.flatten(start_dim=1).T.reshape(q.shape)
        qk = self.pw_conv_2((q * k))
        qkv = v * qk
        out_feat = self.pw_conv_3(qkv)
        return x + out_feat

def ConvFeedForward(dim, mult, dropout):
    return nn.Sequential(
        nn.Conv3d(dim, dim*mult, 1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv3d(dim*mult, dim, 1))

def FeedForward(dim, mult, dropout):
    return nn.Sequential(
        nn.Linear(dim, dim*mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim*mult, dim))
