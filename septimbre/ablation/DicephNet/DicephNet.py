import torch
import torch.nn as nn
import sys
sys.path.append("../../../")
from model.layers import LayerNorm2d, CBLR, CBR, CenterNorm2d
from model.loss import AMT_loss, InfoNCE_loss
from model.config import CONFIG
from model.CQT import CQTsmall_fir
from septimbre.septimbre import HCQTpreprocess

"""
用于编码CQT输入
"""
class DicephNet(nn.Module):
    def __init__(self, emb_dims = 12, CQTconfig=CONFIG.CQT):
        super().__init__()
        self.cqt = CQTsmall_fir(
            False,
            fs = CQTconfig['fs'],
            fmin = CQTconfig['fmin'],
            octaves = CQTconfig['octaves'],
            bins_per_octave = CQTconfig['bins_per_octave'],
            hop = CQTconfig['hop'],
            filter_scale = CQTconfig['filter_scale'],
            requires_grad = False
        )

        self.hcqt = HCQTpreprocess()
        self.k = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.early_conv = nn.Sequential(
            CBLR(CONFIG.Harmonics, 16, (5, 5), 1, "same"),
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

        self.neck_emb = nn.Sequential(
            CenterNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=(5, 5), padding="same"),
            nn.Tanh()
        )
        self.res1 = nn.Sequential(
            LayerNorm2d(16),
            nn.Conv2d(16, 24, kernel_size=(3, 5), padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 16, kernel_size=(3, 5), padding="same"),
        )
        self.emb = nn.Sequential(
            CenterNorm2d(16),
            nn.Conv2d(16, emb_dims, kernel_size=(3, 5), stride=(3, 1), padding=(1, 2))
        )


    def forward(self, x):
        x = self.cqt(x)
        x = self.hcqt(x)
        x = x * self.k + self.b
        # x: (batch, 8, 8*36, len) HCQT
        x = self.early_conv(x) + x

        note_branch = self.neck(self.res(x))  # (B, 16, 84, T)
        yn = self.conv_yn(note_branch)  # (B, 1, 7*12, T)
        yo = self.conv_yo1(note_branch)  # (B, 7, 7*12, T)
        yo = self.conv_yo2(torch.cat((yo, yn), dim=1))

        emb = self.neck_emb(x)      # (B, 16, 84, T)
        emb = self.res1(emb) + emb      # (B, 16, 84, T)
        emb = self.emb(emb)  # (B, emb_dims, 84, T)
        return yo.squeeze(1), yn.squeeze(1), emb  # (batch, 7*12, len), (batch, emb_dims, 7*12, len)

    def clampK(self, min=0.005, max=3.0):
        with torch.no_grad():
            self.k.clamp_(min, max)

    @staticmethod
    def loss(emb, mask, onset, targets, cluster_type = 3):
        """
        计算聚类损失和AMT损失
        emb: (batch, emb_dims, 7*12, time)
        mask: (batch, 7*12, time)
        onset: (batch, 7*12, time)
        targets: (batch, mix, freqs, time)
        """
        # L_cluster = cluster_loss(emb, targets, cluster_type)
        L_cluster = InfoNCE_loss(emb, targets, 0.15)
        if isinstance(targets, list):
            targets = [t.max(dim=0, keepdim=False).values for t in targets]
            targets = torch.stack(targets, dim=0)
        else:
            targets = targets.max(dim=-3, keepdim=False).values
        L_amt = AMT_loss(onset, mask, targets, False)
        return L_cluster, L_amt    # 聚类损失太大了，权重放在训练脚本中当超参数调节


if __name__ == "__main__":
    # 输出参数量
    model = DicephNet(CQTconfig=CONFIG.CQT)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")    # 56517