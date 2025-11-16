from loss import *

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..")
from model.layers import HarmonicaStacking

class BasicPitch(nn.Module):
    loss = basicpitch_loss(weighted=True, positive_weight=0.95)
    def __init__(self, harmonics=8):
        super(BasicPitch, self).__init__()
        self.CQT_BN = nn.BatchNorm2d(1)
        self.HCQT = HarmonicaStacking(HarmonicaStacking.harmonic_shifts(harmonics-1, 1, 36), 7 * 36)
        self.contours = nn.Sequential(
            nn.Conv2d(8, 32, (5, 5), padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 8, (3*13, 3), padding="same"),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.note = nn.Sequential(
            nn.Conv2d(8, 32, (7, 7), padding=(2, 3), stride=(3, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, (3, 7), padding="same"),
            nn.Sigmoid()
        )
        self.onset1 = nn.Sequential(
            nn.Conv2d(8, 32, (5, 5), stride=(3, 1), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.onset2 = nn.Sequential(
            nn.Conv2d(33, 1, (3, 3), padding="same"),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x是CQT
        # 下面这段是BasicPitch的CQT后处理
        power = torch.sum(x.pow(2), dim=1, keepdim=True)
        log_power = 10 * torch.log10(power + 1e-10) # (batch, 1, note, time)
        log_power_min = torch.amin(log_power, dim=(2, 3), keepdim=True)
        log_power_offset = log_power - log_power_min
        log_power_offset_max = log_power_offset_max = torch.amax(log_power_offset, dim=(2, 3), keepdim=True)
        log_power_normalized = self.CQT_BN(log_power_offset / (log_power_offset_max + 1e-10))
        # 源代码harmonicstacking部分
        x = self.HCQT(log_power_normalized) # (batch, 8, 7*36, len)
        contours = self.contours(x) # (batch, 8, 7*36, len)
        # 源代码可选是否计算contours，代价是变成1channel，为了公平起见不计算
        note = self.note(contours)  # (batch, 1, 7*12, len)

        onset = self.onset1(x)  # (batch, 32, 7*12, len)
        onset = torch.cat([onset, note], dim=1) # (batch, 33, 7*12, len)
        onset = self.onset2(onset)  # (batch, 1, 7*12, len)

        return onset.squeeze(1), note.squeeze(1)    # (batch, 7*12, len)


class BasicPitch_all(BasicPitch):
    def __init__(self, CQTconfig, sepParams = None, CQTlearnable=False):
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


if __name__ == "__main__":
    # 输出参数量
    model = BasicPitch()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")    # 56517