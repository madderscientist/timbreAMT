from typing import Literal
import torch
import torch.nn as nn
import numpy as np
from torch.nn.common_types import _size_2_t

class HarmonicaStacking(nn.Module):
    """
    Harmonica Stacking
    输入是CQT的log频率，频率上平移以对齐谐波
    shifts: list of int Harmonic shifts. 0 should be included to keep the original CQT.
    output_bins: int Number of output.
    """
    def __init__(self, shifts, output_bins):
        super().__init__()
        self.shifts = sorted(shifts)
        self.output_bins = output_bins
    
    def forward(self, x):
        """
        x: (batch, channel, n_bins, len) Time-Frequency representation
        output: (batch, channel*len(shifts), output_bins, len) Harmonica Stacking
        没有进行合法性检查
        """
        batch_size, channels, n_bins, length = x.shape
        shifted = torch.zeros((batch_size, channels * len(self.shifts), self.output_bins, length), device=x.device, dtype=x.dtype)

        for i, shift in enumerate(self.shifts):
            if shift < 0:
                shifted[:, i*channels:(i+1)*channels, -shift:self.output_bins, :] = x[:, :, :self.output_bins+shift, :]
            else:
                shifted[:, i*channels:(i+1)*channels, :self.output_bins-shift, :] = x[:, :, shift:self.output_bins, :]

        return shifted

    @staticmethod
    def harmonic_shifts(up_harmonics: int, down_harmonics: int = 1, bins_per_octave: int = 12) -> np.ndarray:
        """
        up_harmonics: int Number of harmonics to stack, including the original CQT
        down_harmonics: int Number of harmonics to stack below the original CQT
        bins_per_octave: int Number of bins per octave
        return: list of int Harmonic shifts
        """
        downs = np.round(-bins_per_octave * np.log2(np.arange(2, down_harmonics+2))).astype(int)
        ups = np.round(bins_per_octave * np.log2(np.arange(1, up_harmonics+1))).astype(int)
        return np.concatenate([downs, ups])

# print(HarmonicaStacking.harmonic_shifts(9, 36))
# [0, 36, 57.05865003, 72, 83.58941142, 93.05865003, 101.06477719, 108, 114.11730005]

class HarmonicaStacking_inv(HarmonicaStacking):
    def __init__(self, shifts, output_bins):
        super().__init__(shifts, output_bins)
    
    def forward(self, x):
        """
        x: (batch, channel*len(shifts), n_bins, len) Time-Frequency representation
        output: (batch, channel, output_bins, len) inverse Harmonica Stacking
        没有进行合法性检查
        """
        batch_size, channelxshifts, n_bins, length = x.shape
        channel = channelxshifts // len(self.shifts)
        original = torch.zeros((batch_size, channel, self.output_bins, length), device=x.device, dtype=x.dtype)

        for i, shift in enumerate(self.shifts):
            hmax = min(self.output_bins, n_bins+shift)
            original[:, :, shift:hmax, :] += x[:, i*channel:(i+1)*channel, :hmax-shift, :]

        return original

"""
补偿Harmonic Stacking中高频补零的部分。用1*1卷积核提取每个时频单元的能量，据此分配补偿的大小。
"""
class CompensateHS(nn.Module):
    def __init__(self, input_channels = 8, offset = [21, 15, 12, 9, 8], layer = [5, 4, 3, 2, 1]):
        super().__init__()
        self.offset = offset.copy()
        self.layer_c = layer.copy()
        self.w = nn.ModuleList([
           nn.Sequential(
                nn.Conv2d(input_channels - c, 1, 1, bias=True),
                nn.ReLU(inplace=True)
           ) for c in layer
        ])
        self.adds = nn.ParameterList([
            nn.Parameter(torch.rand(c,1,1)) for c in layer
        ])

    def forward(self, x):
        batch, dim, note, time = x.shape
        sum = note
        outs = []
        for (c, offset, w, add) in zip(self.layer_c, self.offset, self.w, self.adds):
            channel_start = sum - offset
            x_ = x[:, :dim - c, channel_start:sum, :]
            weight = w(x_)
            outs.append(torch.cat([
                x_, add * weight    # (batch, dim, offset, time)
            ], dim=1))
            sum -= offset
        outs.append(x[:, :, :sum, :])
        outs.reverse()
        return torch.cat(outs, dim=2)


class LayerNorm2d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1.01e-8, affine: bool = True) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_features, eps, affine)

    def forward(self, x):
        # x: (batch, channel, n_bins, len)
        x = x.permute(0, 2, 3, 1)  # (batch, n_bins, len, channel)
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)  # (batch, channel, n_bins, len)
        return x

# instance norm without affine
class CenterNorm2d(nn.Module):
    def __init__(self, dim, eps = 1e-9):
        super().__init__()
        self.k = nn.Parameter(torch.ones(1, dim, 1, 1), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        return self.center_norm(x, self.eps) * self.k

    @staticmethod
    def center_norm(x, eps=1e-9):
        # x: (batch, dim, freq, time)
        mean = x.mean(dim=(-1, -2), keepdim=True)
        std = x.std(dim=(-1, -2), keepdim=True)
        return (x - mean) / (std + eps)

class CenterNorm1d(nn.Module):
    def __init__(self, dim, eps = 1e-9):
        super().__init__()
        self.k = nn.Parameter(torch.ones(1, 1, dim), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        return self.center_norm(x, self.eps) * self.k

    @staticmethod
    def center_norm(x, eps=1e-9):
        # x: (batch, len, dim)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        return (x - mean) / (std + eps)


class CBS(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t = 3,
        stride: _size_2_t = 1,
        padding: _size_2_t | str = "same",
        dilation: _size_2_t = 1,
        padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] = "zeros"
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, dilation = dilation, padding_mode = padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class CBR(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t = 3,
        stride: _size_2_t = 1,
        padding: _size_2_t | str = "same",
        dilation: _size_2_t = 1,
        padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] = "zeros"
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, dilation = dilation, padding_mode = padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class CBLR(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t = 3,
        stride: _size_2_t = 1,
        padding: _size_2_t | str = "same",
        dilation: _size_2_t = 1,
        padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] = "zeros"
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, dilation = dilation, padding_mode = padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class EnergyNorm(nn.Module):
    """
    0: 原数据用能量归一化
    1: 能量的归一化
    2: 都输出
    """
    def __init__(self, output_type = 1, log_scale = True):
        super().__init__()
        self.output_type = output_type
        self.log_scale = log_scale
    
    def forward(self, x):
        """
        把每一帧的能量视为卡方分布，对其进行标准化
        x: (batch, channel, n_bins, len) Time-Frequency representation
        output: (batch, channel, n_bins, len) Energy Normalization
        """
        eng = torch.sum(x.pow(2), dim=1, keepdim=True)
        # eng: (batch, 1, n_bins, len)
        eng_per_frame = torch.sum(eng, dim=2, keepdim=True)
        # eng_per_frame: (batch, 1, 1, len)
        # 计算方差(样本方差)
        std = torch.std(eng_per_frame, dim=3, keepdim=True, unbiased=True)
        # std: (batch, 1, 1, 1)
        # 归一化
        if self.output_type == 0:
            return x / torch.sqrt(std)
        if self.output_type == 1:
            if self.log_scale:
                # less than 1e-8 will be dropped by onnx, but 1.01e-8 will be kept
                return torch.log(eng + 1.01e-8) - torch.log(std)
            return eng / std
        if self.output_type == 2:
            if self.log_scale:
                return x / torch.sqrt(std), torch.log(eng + 1.01e-8) - torch.log(std)
            return x / torch.sqrt(std), eng / std


class DiffPhase(nn.Module):
    """
    计算相位的差分
    """
    def forward(self, x):
        """
        x: (batch, 2, n_bins, len) Time-Frequency representation (real and imag parts interleaved along channel dimension)
        output: (batch, 1, n_bins, len) Phase difference
        """
        phase = torch.atan2(x[:, 1, :, :], x[:, 0, :, :])
        # phase: (batch, n_bins, len)
        d = torch.diff(phase, dim=-1)
        # d: (batch, n_bins, len-1)

        if self.training:
            # 训练时用可微分wrap
            d = torch.atan2(torch.sin(d), torch.cos(d))
        else:
            # 推理时用加减法wrap, 加速计算
            d = torch.where(d > np.pi, d - 2 * np.pi, d)
            d = torch.where(d < -np.pi, d + 2 * np.pi, d)

        d = torch.nn.functional.pad(d, (1, 0))
        return d.unsqueeze(1)