import torch
import torch.nn as nn
import numpy as np

class CAT(nn.Module):
    def __init__(self, *layers, dim=1):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.dim = dim
    
    def forward(self, x):
        outs = [layer(x) for layer in self.layers]
        return torch.cat(outs, dim=self.dim)


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

    def harmonic_shifts(up_harmonics, down_harmonics = 1, bins_per_octave = 12):
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


class CBS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = "same", dilation = 1, padding_mode = "zeros"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, dilation = dilation, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = "same", dilation = 1, padding_mode = "zeros"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, dilation = dilation, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class CBLR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = "same", dilation = 1, padding_mode = "zeros"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, dilation = dilation, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ChordConv_residual(nn.Module):
    """
    对谐波上有重叠的音符进行处理，最后残差连接并batchnorm+SiLU
    in_channels: int Number of input channels
    out_channels: int Number of output channels
    harmonics: list of int Harmonics to consider 按照十二平均律 具体数据参看[small数据集的构建](../data/septimbre/readme.md)
    bins_per_note: int Number of bins per note 因为harmonics是按照十二平均律的音符来的，默认是1
    """
    def __init__(self, in_channels, out_channels, harmonics = [0, 4, 5, 7, 12], bins_per_note = 1):
        super().__init__()
        self.cbrs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    padding = "same",
                    dilation = (max(1, h * bins_per_note), 1),  # 频率方向dilation，时间方向不变
                    kernel_size = (3, 3),
                    bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        for h in harmonics])
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        x: (batch, in_channels, n_bins, len) Time-Frequency representation
        output: (batch, out_channels, n_bins, len) Chord Convolution
        """
        x = sum(conv(x) for conv in self.cbrs)
        x = self.bn(x)
        return self.act(x)


class ChordConv_concat(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, harmonics = [0, 4, 5, 7, 12], bins_per_note = 1):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels, hidden_channels,
                padding = "same",
                dilation = (max(1, h * bins_per_note), 1),  # 频率方向dilation，时间方向不变
                kernel_size = (3, 3),
                bias=False
            )
        for h in harmonics])
        hidden = hidden_channels * len(harmonics)
        self.norm = nn.BatchNorm2d(hidden)
        self.act = nn.ReLU(inplace=True)
        self.pixel = nn.Conv2d(
            hidden, out_channels,
            padding = "same",
            kernel_size = (3, 3),
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act_pixel = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        x: (batch, in_channels, n_bins, len) Time-Frequency representation
        output: (batch, out_channels, n_bins, len) Chord Convolution
        """
        CONVS = [conv(x) for conv in self.convs]
        x = torch.cat(CONVS, dim=1)
        x = self.norm(x)
        x = self.act(x)
        x = self.pixel(x)
        x = self.bn(x)
        return self.act_pixel(x)


class ChordConv_concat_half(nn.Module):
    def __init__(self, in_channels, hidden_channels = [14, 8, 10, 12, 14], harmonics = [0, 4, 5, 7, 12], bins_per_note = 1):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels, hidden_channels[i],
                padding = (max(1, harmonics[i] * bins_per_note), 1),    # onnx不支持有dilation的padding="same"
                dilation = (max(1, harmonics[i] * bins_per_note), 1),   # 频率方向dilation，时间方向不变
                kernel_size = (3, 3),
                bias=False
            )
        for i in range(len(hidden_channels))])
        self.hidden = sum(hidden_channels)
        self.norm = nn.BatchNorm2d(self.hidden)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        x: (batch, in_channels, n_bins, len) Time-Frequency representation
        output: (batch, hidden_channels * len(harmonics), n_bins, len) Chord Convolution
        """
        x = torch.cat([conv(x) for conv in self.convs], dim=1)
        x = self.norm(x)
        x = self.act(x)
        return x


class EnergyNorm(nn.Module):
    """
    0: 原数据用能量归一化
    1: 能量的归一化
    2: 都输出
    """
    def __init__(self, output_type = 1):
        super().__init__()
        self.output_type = output_type
    
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
            return eng / std
        if self.output_type == 2:
            return x / torch.sqrt(std), eng / std
