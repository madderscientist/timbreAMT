import torch
import torch.nn as nn
from itertools import permutations
from layers import HarmonicaStacking, HarmonicaStacking_inv, ChordConv, CBS, EnergyNorm
from attention import TimbreAttention
from denseconv import DenseConv2d_full, DenseCorr2d_full
from loss import AMT_loss, CQT_loss


class ReSepNet(nn.Module):
    def __init__(self):
        super().__init__()
        pulse_dim = 32
        corr_dim = 16
        self.input_norm = EnergyNorm(output_type=2)
        # 输入是CQT的实部和虚部，8个八度，每个音符3个频点。第八个八度用来提供谐波的，最后输出7个八度
        # 在进行HCQT前先引入幅度和相位，增加非线性
        harmonics = 8
        hshift = HarmonicaStacking.harmonic_shifts(harmonics, 36)
        self.HCQT = HarmonicaStacking(hshift, 7 * 36)
        self.early_conv = nn.Sequential(
            CBS(9 * 4, 36, 3, 1, "same"),
            ChordConv(36, 32, [0, 4, 5, 7, 12], 36//12),
            CBS(32, pulse_dim, (3*13, 3), padding="same")
        )
        self.pulse = TimbreAttention(pulse_dim, time_step = 3, topk = 64)
        self.corr = DenseCorr2d_full(pulse_dim, pulse_dim, corr_dim, (3, 3), 1)
        self.corr_aft = nn.Sequential(
            nn.SiLU(inplace=True),
            CBS(corr_dim, corr_dim, 3, padding="same"),
            CBS(corr_dim, corr_dim, 3, padding="same")
        )
        self.note = nn.Sequential(
            nn.Conv2d(corr_dim, 16, kernel_size=(5, 5), padding=(1, 2), stride = (3, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding="same"),
            nn.Sigmoid()
        )
        self.onset_pre = CBS(1, 16, kernel_size=(5, 5), padding=(1, 2), stride=(3, 1))   # 输入是幅度谱
        # 然后和note的输出concat
        self.onset = nn.Sequential(
            nn.Conv2d(17, 1, kernel_size=3, stride=1, padding="same"),
            nn.Sigmoid()
        )
        # 频谱重构 输入是corr的输出和pulse的输出
        self.reconstruction = nn.Sequential(
            DenseConv2d_full(pulse_dim, corr_dim, 36, (3, 3), 1),
            nn.Conv2d(36, harmonics * 2, kernel_size=3, stride=1, padding="same"),
            HarmonicaStacking_inv(hshift, 8 * 36),  # len(hshift)=harmonics，输出要为channel=2的话，输入得有2*harmonics个channel
            EnergyNorm(output_type=0)
        )


    def forward(self, x):
        # x: (batch, 2, n_bins, len) CQT transform real and imag
        mag = torch.sqrt(x[:, 0]**2 + x[:, 1]**2 + 1e-8)    # (batch, n_bins, len)
        phase = torch.atan2(x[:, 1], x[:, 0])
        r = torch.stack((mag, phase), dim=1)
        # r: (batch, 2, n_bins, len)
        tf = torch.cat((x, r), dim=1)
        # tf: (batch, 4, n_bins, len)
        tf = self.HCQT(tf)
        # tf: (batch, 36, n_bins, len)
        tf = self.early_conv(tf)
        # tf: (batch, hidden, n_bins, len)
        pulse = self.pulse(tf)
        # pulse: (batch, hidden, 1, len)
        corr = self.corr(pulse, tf)
        corr = self.corr_aft(corr)
        note = self.note(corr)  # (batch, 1, n_bins, time)
        onset_pre = self.onset_pre(mag.unsqueeze(1)[:,:,:252,:])
        onset = self.onset(torch.cat((onset_pre, note), dim=1))
        reconstr = self.reconstruction(pulse, corr) # (batch, 2, n_bins, len)
        reconstr = self.reconstruction_aft(reconstr)
        return onset.squeeze(1), note.squeeze(1), reconstr

    ##########################损失相关############################
    def loss(onset, note, CQT, target_midi, target_CQT):
        """
        计算两个误差：AMT音符误差和CQT重构误差
        dataloader给出的是混合前:
        input: [batch, mix, 2, freq, time]
        target: [batch, mix, notes, time]
        实际输入是torch.sum(input, dim=1, keepdim=False): [batch, 2, freq, time]
        每一次的输出是: [batch, 2, freq, time]和[batch, notes, time]*2（onset和note）
        堆叠后得到：[batch, mix, 2, freq, time]和[batch, mix, notes, time]
        每个batch的排列都不一样，所以loss要分batch用PIT
        """
        batch_size = target_midi.size(0)
        mix = target_midi.size(1)
        # 生成所有排列
        perms = [list(perm) for perm in permutations(range(mix))]
        losses = []
        for batch in range(batch_size):
            _onset = onset[batch]
            _note = note[batch]
            _CQT = CQT[batch]
            _target_midi = target_midi[batch]
            _target_CQT = target_CQT[batch]
            loss_perm = torch.stack([(
                AMT_loss(_onset[perm], _note[perm], _target_midi) +
                CQT_loss(_CQT[perm], _target_CQT)
            ) for perm in perms])
            losses.append(loss_perm.min())
        return torch.stack(losses).sum()    # 保持数据在GPU上

if __name__ == "__main__":
    model = ReSepNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")