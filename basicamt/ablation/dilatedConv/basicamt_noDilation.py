import torch
import torch.nn as nn

import sys
sys.path.append("..")
from model.layers import HarmonicaStacking, CBR, EnergyNorm, CBLR
from model.loss import AMT_loss

class BasicAMT_noDilation(nn.Module):
    def __init__(self):
        super().__init__()
        harmonics = 8
        self.eng = EnergyNorm(output_type=1, log_scale=True)
        self.k = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.HCQT = HarmonicaStacking(HarmonicaStacking.harmonic_shifts(harmonics-1, 1, 36), 7 * 36)
        self.early_conv = nn.Sequential(
            CBLR(harmonics, 16, (5, 5), 1, "same"),
            nn.Conv2d(16, 8, kernel_size=(39, 3), dilation=1, padding="same", stride=1),
            nn.BatchNorm2d(8),
        )
        self.res = nn.ReLU(inplace=True)
        self.neck = CBR(8, 16, (5, 5), 1, "same", 1)
        self.conv_yn = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=(5, 5), stride=(3, 1), padding=(1, 2)),
            nn.Sigmoid()
        )
        self.conv_yo1 = CBR(16, 7, kernel_size=(7, 3), padding=(2, 1), stride=(3, 1))
        self.conv_yo2 = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=(3, 5), stride=1, padding="same"),
            nn.Sigmoid()
        )


    def forward(self, x):
        # x: (batch, 2, 8*36, len)
        eng = self.eng(x) * self.k + self.b
        # eng: (batch, 1, 8*36, len)
        stacked = self.HCQT(eng)
        # stacked: (batch, 8, 7*36, len)
        early_conv = self.early_conv(stacked)
        # early_conv: (batch, 8, 7*36, len)

        res = self.res(early_conv + stacked)
        neck = self.neck(res)
        # neck: (batch, 8, 7*36, len)·

        yn = self.conv_yn(neck)
        # yn: (batch, 1, 7*12, len)

        _yo = self.conv_yo1(neck)
        # _yo: (batch, 7, 7*12, len)
        yo = torch.concat((yn, _yo), dim=1)
        # yo: (batch, 8, 7*12, len)
        yo = self.conv_yo2(yo)
        # yo: (batch, 1, 7*12, len)
        return yo.squeeze(1), yn.squeeze(1)
        # (batch, 7*12, len)
    
    def clampK(self, min=0.01, max=3.0):
        with torch.no_grad():
            self.k.clamp_(min, max)

    ##########################损失相关############################
    @staticmethod
    def loss(onset, note, midiarray):
        return AMT_loss(onset, note, midiarray, mse=False)


class BasicAMT_all_noDilation(BasicAMT_noDilation):
    def __init__(self, CQTconfig, sepParams = None, CQTlearnable = True):
        super().__init__()
        from model.CQT import CQTsmall_fir
        if sepParams is not None:
            super().load_state_dict(sepParams)
        self.cqt = CQTsmall_fir(
            False,
            fs = CQTconfig['fs'],
            fmin = CQTconfig['fmin'],
            octaves = CQTconfig['octaves'],
            bins_per_octave = CQTconfig['bins_per_octave'],
            hop = CQTconfig['hop'],
            filter_scale = CQTconfig['filter_scale'],
            requires_grad = CQTlearnable
        )

    def sep_params(self):
        return super().state_dict()

    def cqt_params(self):
        return self.cqt.state_dict()
    
    def forward(self, x):
        x = self.cqt(x)
        return super().forward(x)


class BasicAMT_44100_noDilation(torch.nn.Module):
    """
    相比BasicAMT_all，输入为44100Hz采样率的音频，会先进行降采样到22050Hz
    """
    def __init__(self, basciamt_all: BasicAMT_all_noDilation):
        super().__init__()
        self.basicamt_all = basciamt_all

    def forward(self, x):
        # (1, 1, time)
        # 降采样到22050Hz（假定输出为44100Hz）
        x = self.basicamt_all.cqt.down2sample(x)
        onset, note = self.basicamt_all(x)
        # 减小js代码量：在模型内部归一化
        onset /= onset.max()
        note /= note.max()
        return onset, note
        # (batch, 84, frame)   
