import torch
import torch.nn as nn

import sys
sys.path.append("..")
from model.layers import HarmonicaStacking, CBR, EnergyNorm, CBLR
from model.loss import AMT_loss

class BasicAMT(nn.Module):
    def __init__(self):
        super().__init__()
        harmonics = 8
        self.eng = EnergyNorm(output_type=1)
        self.HCQT = HarmonicaStacking(HarmonicaStacking.harmonic_shifts(harmonics-1, 1, 36), 7 * 36)
        self.early_conv = nn.Sequential(
            CBLR(harmonics, 16, (5, 5), 1, "same"),
            CBR(16, 10, kernel_size=(25, 3), dilation=(3, 1), padding=((25//2)*3, 1), stride=1)
        )
        self.neck = CBR(10 + harmonics, 18, (5, 5), 1, "same", 1)
        self.conv_yn = nn.Sequential(
            nn.Conv2d(18, 1, (5, 5), (3, 1), (1, 2)),
            nn.Sigmoid()
        )
        self.conv_yo1 = CBR(18, 7, kernel_size=(7, 3), padding=(2, 1), stride=(3, 1))
        self.conv_yo2 = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=(3, 5), stride=1, padding="same"),
            nn.Sigmoid()
        )


    def forward(self, x):
        # x: (batch, 2, 8*36, len)
        eng = self.eng(x)
        # eng: (batch, 1, 8*36, len)
        stacked = self.HCQT(eng)
        # stacked: (batch, 9, 7*36, len)
        early_conv = self.early_conv(stacked)
        # early_conv: (batch, 8, 7*36, len)

        early_conv = torch.concat((stacked, early_conv), dim=1)
        # early_conv: (batch, 17, 7*36, len)
        neck = self.neck(early_conv)
        # neck: (batch, 16, 7*36, len)

        yn = self.conv_yn(neck)
        # yn: (batch, 1, 7*12, len)

        _yo = self.conv_yo1(neck)
        # _yo: (batch, 17, 7*12, len)
        yo = torch.concat((yn, _yo), dim=1)
        # yo: (batch, 18, 7*12, len)
        yo = self.conv_yo2(yo)
        # yo: (batch, 1, 7*12, len)
        return yo.squeeze(1), yn.squeeze(1)
        # (batch, 7*12, len)

    ##########################损失相关############################
    @staticmethod
    def loss(onset, note, midiarray):
        return AMT_loss(onset, note, midiarray, mse=False)


class BasicAMT_all(BasicAMT):
    def __init__(self, CQTconfig, sepParams = None):
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
            requires_grad = True
        )

    def sep_params(self):
        return super().state_dict()

    def cqt_params(self):
        return self.cqt.state_dict()
    
    def forward(self, x):
        x = self.cqt(x)
        return super().forward(x)


class BasicAMT_44100(torch.nn.Module):
    """
    相比BasicAMT_all，输入为44100Hz采样率的音频，会先进行降采样到22050Hz
    """
    def __init__(self, basciamt_all: BasicAMT_all):
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
