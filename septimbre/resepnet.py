import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")
from model.layers import HarmonicaStacking, CBR, EnergyNorm, CBS, CBLR
from model.loss import AMT_loss, cluster_loss
from model.attention import FlowAttention, FreqPosEncoding


"""
用于编码CQT输入
"""
class Cluster(nn.Module):
    def __init__(self, emb_dims = 18):
        super().__init__()
        self.emb_dims = emb_dims
        harmonics = 8
        self.eng = EnergyNorm(output_type=1)
        self.HCQT = HarmonicaStacking(HarmonicaStacking.harmonic_shifts(harmonics-1, 1, 36), 7 * 36)
        self.early_conv = nn.Sequential(
            CBLR(harmonics * 2, 24, (5, 5), 1, "same"),
            CBR(24, 20, kernel_size=(25, 3), dilation=(3, 1), padding=((25//2)*3, 1), stride=1)
        )
        self.neck = CBS(20 + 12, 24, (5, 5), 1, "same", 1)
        self.compensate = FreqPosEncoding(24, 84 * 3)
        self.mask = nn.Sequential(
            nn.Conv2d(25, 1, (5, 5), (3, 1), (1, 2)),
            nn.Sigmoid()
        )
        self.onset_pre = CBR(25, 6, kernel_size=(7, 3), padding=(2, 1), stride=(3, 1))
        self.onset = nn.Sequential(
            nn.Conv2d(7, 1, kernel_size=(3, 5), stride=1, padding="same"),
            nn.Sigmoid()
        )
        self.FA = FlowAttention(emb_dims, emb_dims, 1, drop_out=0.05, eps=1e-6)
        self.emb_Q = nn.Conv2d(24, emb_dims, (3, 3), (3, 1), (0, 1))
        self.emb_K = nn.Conv2d(24, emb_dims, (3, 3), (3, 1), (0, 1))
        self.emb_V = nn.Conv2d(24, emb_dims, (3, 3), (3, 1), (0, 1))
        self.emb = nn.Conv2d(24, emb_dims, (5, 5), (3, 1), (1, 2))

    def forward(self, x):
        # x: (batch, 2, 8*36, len) CQT的输出
        # 实验表明，音色和相位的关系更大。因此保留相位信息，组成旋转矢量
        eng = self.eng(x)   # (batch, 1, 8*36, len)
        phase = torch.atan2(x[:, 1], x[:, 0])   # (batch, 8*36, len)
        rotvec = torch.concat((eng, phase.unsqueeze(1)), dim=1)
        # rotvec: (batch, 2, 8*36, len)
        stacked = self.HCQT(rotvec)
        # stacked: (batch, 16, 7*36, len)

        early_conv = self.early_conv(stacked)
        # early_conv: (batch, 20, 7*36, len)
        early_conv = torch.concat((stacked[:, [0,2,3,4,5,6,7,8,9,10,12,14], :, :], early_conv), dim=1)
        # early_conv: (batch, 32, 7*36, len)

        neck = self.neck(early_conv)
        # neck: (batch, 24, 7*36, len)

        neck_with_eng = torch.cat((neck, neck.pow(2).sum(dim=1, keepdim=True)), dim=1)
        mask = self.mask(neck_with_eng)
        onset = self.onset_pre(neck_with_eng)
        onset = torch.cat((onset, mask), dim=1)
        onset = self.onset(onset)

        # 不进行归一化不好。
        neck = self.compensate(neck)
        emb = self.FA(
            self.emb_Q(neck),
            masknorm(self.emb_K(neck), mask, True),
            masknorm(self.emb_V(neck), mask, True)
        ) + self.emb(neck)
        emb = masknorm(emb)
        return emb, mask.squeeze(1), onset.squeeze(1)

    @staticmethod
    def loss(emb, mask, onset, targets):
        """
        计算聚类损失和AMT损失
        emb: (batch, emb_dims, 7*12, time)
        mask: (batch, 7*12, time)
        onset: (batch, 7*12, time)
        targets: (batch, mix, freqs, time)
        """
        L_cluster = cluster_loss(emb, targets, 2)
        L_amt = AMT_loss(onset, mask, targets.max(dim=-3, keepdim=False).values, False)
        return L_cluster, L_amt    # 聚类损失太大了，权重放在训练脚本中当超参数调节


class Cluster_all(Cluster):
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
    
    def forward(self, x):
        x = self.cqt(x)
        return super().forward(x)


def masknorm(x, mask = None, mask_norm = False):
    amp = torch.sqrt(x.pow(2).sum(dim=1, keepdim=True) + 1e-8)
    if mask is None:
        return x / amp
    else:
        if mask_norm:
            mask_sum = mask.sum(dim=(2, 3), keepdim=True)
            return x * mask / (amp * mask_sum)
        else:
            return x * mask / amp