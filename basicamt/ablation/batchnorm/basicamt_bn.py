import torch
import torch.nn as nn

import sys
sys.path.append("..")
from model.layers import HarmonicaStacking, CBR, CBLR
from model.loss import AMT_loss

class EnergyNorm_BN(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(1)
    
    def forward(self, x):
        # x: CQT
        # BasicPitch的CQT后处理
        power = torch.sum(x.pow(2), dim=1, keepdim=True)
        log_power = 10 * torch.log10(power + 1e-10) # (batch, 1, note, time)
        log_power_min = torch.amin(log_power, dim=(2, 3), keepdim=True)
        log_power_offset = log_power - log_power_min
        log_power_offset_max = log_power_offset_max = torch.amax(log_power_offset, dim=(2, 3), keepdim=True)
        return self.bn(log_power_offset / (log_power_offset_max + 1e-10))

class BasicAMT_BN(nn.Module):
    def __init__(self):
        super().__init__()
        harmonics = 8
        self.eng = EnergyNorm_BN()
        self.HCQT = HarmonicaStacking(HarmonicaStacking.harmonic_shifts(harmonics-1, 1, 36), 7 * 36)
        self.early_conv = nn.Sequential(
            CBLR(harmonics, 16, (5, 5), 1, "same"),
            nn.Conv2d(16, 8, kernel_size=(25, 3), dilation=(3, 1), padding=((25//2)*3, 1), stride=1),
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
        eng = self.eng(x)
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

    ##########################损失相关############################
    @staticmethod
    def loss(onset, note, midiarray):
        return AMT_loss(onset, note, midiarray, mse=False)


class BasicAMT_all_BN(BasicAMT_BN):
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