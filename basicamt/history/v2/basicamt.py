import torch
import torch.nn as nn
from layers import HarmonicaStacking, ChordConv2, CBR, EnergyNorm
from loss import AMT_loss

class BasicAMT(nn.Module):
    def __init__(self):
        super().__init__()
        harmonics = 9
        self.eng = EnergyNorm(output_type=1)
        self.HCQT = HarmonicaStacking(HarmonicaStacking.harmonic_shifts(harmonics-1, 1, 36), 7 * 36)
        self.early_conv = nn.Sequential(
            CBR(harmonics, 16, (5, 3), 1, "same"),
            CBR(16, 8, kernel_size=(25, 3), dilation=(3, 1), padding="same", stride=1)
        )
        self.conv_yn = nn.Sequential(
            ChordConv2(8 + harmonics, 14, 32, [0, 4, 5, 7, 12], 3),
            nn.Conv2d(32, 1, kernel_size=(7, 3), stride=(3, 1), padding=(2, 1)),
            nn.Sigmoid()
        )
        self.conv_yo1 = CBR(8 + harmonics, 17, kernel_size=(7, 3), padding=(2, 1), stride=(3, 1))
        self.conv_yo2 = nn.Sequential(
            nn.Conv2d(18, 1, kernel_size=(3, 5), stride=1, padding="same"),
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

        early_conv = torch.concat((stacked, early_conv), dim=1)
        # early_conv: (batch, 16, 7*36, len)
        yn = self.conv_yn(early_conv)
        # yn: (batch, 1, 7*12, len)

        _yo = self.conv_yo1(early_conv)
        # _yo: (batch, 15, 7*12, len)
        yo = torch.concat((yn, _yo), dim=1)
        # yo: (batch, 16, 7*12, len)
        yo = self.conv_yo2(yo)
        # yo: (batch, 1, 7*12, len)
        return yo.squeeze(1), yn.squeeze(1)
        # (batch, 7*12, len)

    ##########################损失相关############################
    @staticmethod
    def loss(onset, note, midiarray):
        return AMT_loss(onset, note, midiarray, mse=False)


from CQT import CQTsmall

class BasicAMT_all(BasicAMT):
    def __init__(self, CQTconfig, sepParams = None):
        super().__init__()
        if sepParams is not None:
            super().load_state_dict(sepParams)
        self.cqt = CQTsmall(
            CQTconfig['fs'],
            fmin=CQTconfig['fmin'],
            octaves=CQTconfig['octaves'],
            bins_per_octave=CQTconfig['bins_per_octave'],
            hop=CQTconfig['hop'],
            filter_scale=CQTconfig['filter_scale'],
            requires_grad=True
        )

    def sep_params(self):
        return super().state_dict()

    def cqt_params(self):
        return self.cqt.state_dict()
    
    def forward(self, x):
        x = self.cqt(x)
        return super().forward(x)