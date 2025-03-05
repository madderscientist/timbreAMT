"""
    const Q transform for pytorch
    principle: https://zhuanlan.zhihu.com/p/716574483
    referrence: nnAudio
"""
import torch
import torch.nn as nn
from torch.nn.functional import conv1d
from torchaudio.functional import filtfilt
import numpy as np
import scipy.signal as signal

def createCQTkernel(Q, fs, fmin, n_bins, bins_per_octave = 12):
    n_bins = int(n_bins)
    bins_per_octave = int(bins_per_octave)
    # frequency list of all bins
    freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.double(bins_per_octave))
    lengths = np.ceil(Q * fs / freqs)
    max_len = int(lengths[0])    # 频率越低窗越长
    # 如果窗长是偶数则加一，因为奇数方便padding
    if max_len % 2 == 0:
        max_len += 1
    tempKernel = np.zeros((n_bins, max_len), dtype=np.complex64)
    for k in range(0, n_bins):
        l = int(lengths[k])
        start = (max_len - l) // 2
        window = signal.get_window('blackmanharris', l, fftbins=False)
        # 对window进行归一化后作用于旋转因子 保证相位0在中心
        tempKernel[k, start : start + l] = window / np.sum(window) * np.exp(1j * 2 * np.pi * np.r_[-l//2+1 : l//2+1] * freqs[k] / fs)
    return tempKernel, max_len

def createCQTkernel_fit(Q, fs, fmin, n_bins, bins_per_octave = 12):
    n_bins = int(n_bins)
    bins_per_octave = int(bins_per_octave)
    freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.double(bins_per_octave))
    lengths = np.ceil(Q * fs / freqs)
    tempKernel = [None] * n_bins
    lens = np.zeros(n_bins, dtype=int)
    for k in range(0, n_bins):
        l = int(lengths[k])
        # 如果窗长是偶数则加一，因为奇数方便padding
        if l % 2 == 0:
            l += 1
        lens[k] = l
        window = signal.get_window('blackmanharris', l, fftbins=False)
        # 对window进行归一化后作用于旋转因子 保证相位0在中心
        tempKernel[k] = window / np.sum(window) * np.exp(1j * 2 * np.pi * np.r_[-l//2+1 : l//2+1] * freqs[k] / fs)
    return tempKernel, lens

def audio_input(x):
    """
    Auto broadcast input so that it can fits into a Conv1d
    """
    if x.dim() == 2:    # (batch, len)
        x = x.unsqueeze(1)
    elif x.dim() == 1:  # (len)
        # If nn.DataParallel is used, this broadcast doesn't work
        x = x[None, None, :]
    elif x.dim() == 3:  # (batch, channel, len)
        pass
    else:
        raise ValueError(
            "Only support input with shape = (batch, len) or shape = (len)"
        )
    if x.size(1) > 1:   # only accept mono audio
        x = x.mean(dim=1, keepdim=True)
    return x

"""CQT的定义实现 每个kernel等长 因此参数量很多"""
class CQT(nn.Module):
    def __init__(self, fs, fmin, n_bins, bins_per_octave, hop, filter_scale=1, requires_grad = True):
        super().__init__()
        self.fs = fs
        self.fmin = fmin
        self.n_bins = int(n_bins)
        self.bins_per_octave = int(bins_per_octave)
        self.hop = int(hop)
        Q = float(filter_scale) / (2 ** (1 / bins_per_octave) - 1)
        cqt_kernels, self.width = createCQTkernel(Q, fs, fmin, n_bins, bins_per_octave)
        self.cqt_kernels = nn.Parameter(
            torch.cat((
                torch.tensor(np.real(cqt_kernels), dtype=torch.float32),
                torch.tensor(np.imag(cqt_kernels), dtype=torch.float32)
            ), dim=0).unsqueeze(1),
            requires_grad=requires_grad
        )
        self.padding = nn.ConstantPad1d(self.width//2, 0)

    def forward(self, x):
        """
        x: (batch, len) mono audio
        output: (batch, 2, n_bins, len) CQT transform real and imag
        """
        x = audio_input(x)
        x = self.padding(x)
        CQT_result = conv1d(x, self.cqt_kernels, stride=self.hop)
        return CQT_result.view(CQT_result.size(0), 2, self.n_bins, -1)

"""上面的CQT在高频时kernel有很多0 CQT_fit删去了这些冗余 但是慢"""
class CQT_fit(nn.Module):
    def __init__(self, fs, fmin, n_bins, bins_per_octave, hop, filter_scale=1, requires_grad = True):
        super().__init__()
        self.fs = fs
        self.fmin = fmin
        self.n_bins = int(n_bins)
        self.bins_per_octave = int(bins_per_octave)
        self.hop = int(hop)
        Q = float(filter_scale) / (2 ** (1 / bins_per_octave) - 1)
        cqt_kernels, self.width = createCQTkernel_fit(Q, fs, fmin, n_bins, bins_per_octave)
        self.cqt_kernels = nn.ParameterList()
        for i in range(n_bins):
            self.cqt_kernels.append(nn.Parameter(
                torch.stack((
                    torch.tensor(np.real(cqt_kernels[i]), dtype=torch.float32),
                    torch.tensor(np.imag(cqt_kernels[i]), dtype=torch.float32)
                ), dim=0).unsqueeze(1),
                requires_grad=requires_grad
            ))
    
    def forward(self, x):
        x = audio_input(x)
        CQT_result = [conv1d(x, self.cqt_kernels[i], stride=self.hop, padding=self.width[i]//2) for i in range(self.n_bins)]
        CQT_result = torch.cat(CQT_result, dim=-1)
        return CQT_result.view(CQT_result.size(0), 2, self.n_bins, -1)


"""基于降采样的CQT 参数量小 推荐使用"""
class CQTsmall(nn.Module):
    def __init__(self, fs, fmin, octaves, bins_per_octave, hop, filter_scale=1, requires_grad = True):
        super().__init__()
        self.fs = fs
        self.fmin = fmin
        self.bins_per_octave = int(bins_per_octave)
        self.octaves = int(octaves)
        self.n_bins = int(octaves * bins_per_octave)
        freqMul = 2 ** (octaves - 1)
        self.hop = int(max(1, round(hop / freqMul)) * freqMul)
        fmin_max = fmin * freqMul    # 最高八度的最低频率
        Q = float(filter_scale) / (2 ** (1 / bins_per_octave) - 1)
        cqt_kernels, self.width = createCQTkernel(Q, fs, fmin_max, bins_per_octave, bins_per_octave)
        self.cqt_kernels = nn.Parameter(
            torch.cat((
                torch.tensor(np.real(cqt_kernels), dtype=torch.float32),
                torch.tensor(np.imag(cqt_kernels), dtype=torch.float32)
            ), dim=0).unsqueeze(1),
            requires_grad=requires_grad
        )
        # 之后考虑换成FIR滤波器
        iir = signal.iirfilter(10, 0.48, btype='low', ftype='butter', output='ba')
        self.iir_num = nn.Parameter(torch.tensor(iir[0], dtype=torch.float32), requires_grad=False)
        self.iir_den = nn.Parameter(torch.tensor(iir[1], dtype=torch.float32), requires_grad=False) # 分母不方便变，不稳定
    
    def forward(self, x):
        x = audio_input(x)
        pad = self.width // 2
        hop = self.hop
        firstOctave = conv1d(x, self.cqt_kernels, stride=hop, padding=pad)
        CQT_results = [firstOctave.view(firstOctave.size(0), 2, self.bins_per_octave, -1)]
        for i in range(1, self.octaves):
            hop = hop // 2
            x = filtfilt(x, self.iir_den, self.iir_num)[:, :, ::2]
            CQT_result = conv1d(x, self.cqt_kernels, stride=hop, padding=pad)
            CQT_results.insert(0, CQT_result.view(CQT_result.size(0), 2, self.bins_per_octave, -1))
        return torch.cat(CQT_results, dim=2)