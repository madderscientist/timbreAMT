import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")
from model.layers import HarmonicaStacking, CBR, EnergyNorm, CBS, CBLR, DiffPhase
from model.loss import AMT_loss, cluster_loss
from model.attention import FlowAttention, FreqPosEncoding
from model.config import CONFIG
from model.CQT import CQTsmall_fir

_harmonics = CONFIG.Harmonics

class HCQTpreprocess(nn.Module):
    def __init__(self):
        super().__init__()
        self.eng = EnergyNorm(output_type=1, log_scale=True)
        self.k = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.HCQT = HarmonicaStacking(HarmonicaStacking.harmonic_shifts(_harmonics-1, 1, 36), 7 * 36)

    def forward(self, x):
        # x: (batch, 2, 8*36, len)
        eng = self.eng(x) * self.k + self.b
        # eng: (batch, 1, 8*36, len)
        stacked = self.HCQT(eng)
        # stacked: (batch, 8, 7*36, len)
        return stacked


class NotePredictBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.early_conv = nn.Sequential(
            CBLR(_harmonics, 16, (5, 5), 1, "same"),
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
        # x: (batch, 8, 7*36, len)
        early_conv = self.early_conv(x)
        # early_conv: (batch, 8, 7*36, len)

        res = self.res(early_conv + x)
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
        return yo, yn

"""
用于编码CQT输入
"""
class Cluster(nn.Module):
    def __init__(self, emb_dims = 12):
        super().__init__()
        self.emb_dims = emb_dims
        # self.compensate = FreqPosEncoding(24, 84 * 3)
        self.pre_encoding = CBLR(8, 16, (5, 5), padding="same")
        self.T = nn.Parameter(torch.tensor(2.0), requires_grad=True)
        self.FA = FlowAttention(emb_dims, emb_dims, 1, drop_out=0.05, eps=1.01e-8)
        self.emb_Q = nn.Conv2d(16, emb_dims, (3, 3), (3, 1), (0, 1))
        self.emb_KV = nn.Conv2d(16, emb_dims, (3, 3), (3, 1), (0, 1))
        # self.ln = nn.LayerNorm(emb_dims)

    def forward(self, x, frame):
        # x: (batch, 8, 8*36, len) HCQT
        # frame: (batch, 1, 8*12, len) 音符预测

        # 不进行归一化不好。
        # neck = self.compensate(x)
        x = self.pre_encoding(x)
        # x: (batch, 16, 8*36, len)
        KV = self.emb_KV(x) # (batch, emb_dims, 7*12, len)
        KV = masknorm(KV, frame, True) * self.T
        Q = self.emb_Q(x)  # (batch, emb_dims, 7*12, len)
        emb = self.FA(
            Q,
            KV,
            KV
        )   # 先试试没有残差
        # emb: (batch, emb_dims, 7*12, len)
        # emb = self.ln(emb.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return emb

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


class SepAMT():
    def __init__(self, CQTconfig, sepParams = None):
        super().__init__()
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
        self.hcqt = HCQTpreprocess()
        self.note_branch = NotePredictBranch()
        self.cluster = Cluster()
    
    def forward(self, x):
        x = self.cqt(x)
        x = self.hcqt(x)
        note_onset, note_pred = self.note_branch(x)
        emb = self.cluster(x, note_pred)
        return note_onset.squeeze(1), note_pred.squeeze(1), emb


def masknorm(x, mask = None, mask_norm = False):
    amp = torch.sqrt(x.pow(2).sum(dim=1, keepdim=True) + 1.01e-8)
    if mask is None:
        return x / amp
    else:
        if mask_norm:
            mask_sum = mask.sum(dim=(2, 3), keepdim=True)
            return x * mask / (amp * mask_sum)
        else:
            return x * mask / amp