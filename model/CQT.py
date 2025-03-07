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
    return x    # (batch, 1, len)

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


"""基于降采样的CQT 参数量小 推荐使用 更推荐用后面的CQTsmall_fir"""
class CQTsmall(nn.Module):
    def __init__(self, fs, fmin, octaves, bins_per_octave, hop, filter_scale=1, requires_grad = True):
        super().__init__()
        CQTsmall.init(self, fs, fmin, octaves, bins_per_octave, hop, filter_scale, requires_grad)
        # 之后考虑换成FIR滤波器【已更换，在CQTsmall_fir类中】
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
            x = self.down2sample(x)
            CQT_result = conv1d(x, self.cqt_kernels, stride=hop, padding=pad)
            CQT_results.insert(0, CQT_result.view(CQT_result.size(0), 2, self.bins_per_octave, -1))
        return torch.cat(CQT_results, dim=2)

    def anti_aliasing_filter(self, x):
        return filtfilt(x, self.iir_den, self.iir_num)

    def down2sample(self, x):
        return self.anti_aliasing_filter(x)[:, :, ::2]

    @staticmethod
    def init(obj, fs, fmin, octaves, bins_per_octave, hop, filter_scale=1, requires_grad = True):
        obj.fs = fs
        obj.fmin = fmin
        obj.bins_per_octave = int(bins_per_octave)
        obj.octaves = int(octaves)
        obj.n_bins = int(octaves * bins_per_octave)
        freqMul = 2 ** (octaves - 1)
        obj.hop = int(max(1, round(hop / freqMul)) * freqMul)
        fmin_max = fmin * freqMul    # 最高八度的最低频率
        Q = float(filter_scale) / (2 ** (1 / bins_per_octave) - 1)
        cqt_kernels, obj.width = createCQTkernel(Q, fs, fmin_max, bins_per_octave, bins_per_octave)
        obj.cqt_kernels = nn.Parameter(
            torch.cat((
                torch.tensor(np.real(cqt_kernels), dtype=torch.float32),
                torch.tensor(np.imag(cqt_kernels), dtype=torch.float32)
            ), dim=0).unsqueeze(1),
            requires_grad=requires_grad
        )


"""
    基于FIR滤波器的filtfilt实现
    用于替换torchaudio的filtfilt
    写了篇文章：https://zhuanlan.zhihu.com/p/28510270001
"""
class FIRfiltfilt(nn.Module):
    def __init__(self, fir_coeff, compensate = False, requires_grad = False):
        super().__init__()
        fir = np.flip(fir_coeff).copy() # 由于torch中conv其实是corr，所以需要反转滤波器 不过不反转也没事，因为是对称的
        self.fir = nn.Parameter(
            torch.tensor(fir, dtype=torch.float32).view(1, 1, -1),
            requires_grad=requires_grad
        )
        fir_len= len(fir)
        if compensate:
            self.n_fact = (fir_len - 1) * 3
            self.padding = nn.ZeroPad1d((fir_len - 1, 0)) # 只在左边padding
            # filtfilt边缘效应的补偿
            zi = signal.lfilter_zi(fir_coeff, 1)
            self.zi = nn.Parameter(torch.tensor(zi, dtype=torch.float32), requires_grad=False)
            self.filtfilt = self.filtfilt_v1
        else:
            left_padding = fir_len//2
            right_padding = fir_len - 1 - left_padding
            self.padding = nn.ZeroPad1d((left_padding, right_padding))
            self.filtfilt = self.filtfilt_v2
        
    def forward(self, x):
        return self.filtfilt(x)
    
    @staticmethod
    def signal_extend(sig, nfact):
        """
        filtfilt补偿的信号扩展
        :param sig: (batch, 1, len)
        :param nfact: 补偿的长度，一般是滤波器长度减一的三倍
        """
        head_ext = 2 * sig[:,:,0] - sig[:,:,1:nfact+1].flip(-1)
        tail_ext = 2 * sig[:,:,-1] - sig[:,:,-nfact-1:-1].flip(-1)
        return torch.cat((head_ext, sig, tail_ext), dim=-1)

    def filtfilt_v1(self, x):
        """
        补偿的filtfilt。补偿后起点和终点，和输入一样
        :param x: (batch, 1, len)
        :return: (batch, 1, len)
        """
        # x: (batch, 1, len)
        # 两边补充
        x = FIRfiltfilt.signal_extend(x, self.n_fact)  # (batch, 1, len + 2 * n_fact)
        # 正向滤波
        compansate = x[:,:,:1] * self.zi
        forward = conv1d(self.padding(x), self.fir)
        forward[:,:,:compansate.shape[-1]] += compansate
        # 反向滤波
        forward = forward.flip(-1)
        compansate_b = forward[:,:,:1] * self.zi
        backward = conv1d(self.padding(forward), self.fir)
        backward[:,:,:compansate_b.shape[-1]] += compansate_b
        # 去掉两边padding
        backward = backward.flip(-1)
        return backward[:,:,self.n_fact:-self.n_fact]

    def filtfilt_v2(self, x):
        """
        简单的filtfiilt（两边补零），无补偿
        :param x: (batch, 1, len)
        :return: (batch, 1, len)
        """
        # 正向滤波
        forward = conv1d(self.padding(x), self.fir).flip(-1)
        # 反向滤波
        backward = conv1d(self.padding(forward), self.fir).flip(-1)
        return backward


"""
    用FIR滤波器代替IIR滤波器
    用了自己的filtfilt实现，因为torchaudio的filtfilt有问题
"""
class CQTsmall_fir(nn.Module):
    def __init__(self, compensate = False, **kwargs):
        """
        :param compensate: 是否使用补偿的filtfilt
        第一种构造方法：从CQTsmall构造：
            :param cqtsmall_iir: CQTsmall的实例
            :param requires_grad: 是否需要梯度
        第二种构造方法：直接构造：
            :param fs: 采样率
            :param fmin: 最低频率
            :param octaves: 八度数
            :param bins_per_octave: 每个八度的频率数
            :param hop: 跳跃长度
            :param filter_scale: 滤波器的scale
            :param requires_grad: 是否需要梯度
        """
        super().__init__()
        if 'cqtsmall_iir' in kwargs:
            self.from_CQTsmall_iir(**kwargs)
        else:
            CQTsmall.init(self, **kwargs)

        # 创建fir滤波器 不参与训练
        self.filtfilt = FIRfiltfilt(
            signal.firwin(33, 0.5, window='hamming'),
            compensate = compensate,
            requires_grad = False
        )

    def from_CQTsmall_iir(self, cqtsmall_iir: CQTsmall, requires_grad = True):
        self.fs = cqtsmall_iir.fs
        self.fmin = cqtsmall_iir.fmin
        self.bins_per_octave = cqtsmall_iir.bins_per_octave
        self.octaves = cqtsmall_iir.octaves
        self.n_bins = cqtsmall_iir.n_bins
        self.hop = cqtsmall_iir.hop
        self.width = cqtsmall_iir.width
        self.cqt_kernels = nn.Parameter(cqtsmall_iir.cqt_kernels.clone().detach(), requires_grad=requires_grad)
        return self

    def forward(self, x):
        x = audio_input(x)  # (batch, 1, len)
        pad = self.width // 2
        hop = self.hop
        firstOctave = conv1d(x, self.cqt_kernels, stride=hop, padding=pad)
        CQT_results = [firstOctave.view(firstOctave.size(0), 2, self.bins_per_octave, -1)]
        for i in range(1, self.octaves):
            hop = hop // 2
            x = self.down2sample(x)
            CQT_result = conv1d(x, self.cqt_kernels, stride=hop, padding=pad)
            CQT_results.append(CQT_result.view(CQT_result.size(0), 2, self.bins_per_octave, -1))
        CQT_results.reverse()
        return torch.cat(CQT_results, dim=2)

    def down2sample(self, x):
        return self.filtfilt(x)[:, :, ::2]