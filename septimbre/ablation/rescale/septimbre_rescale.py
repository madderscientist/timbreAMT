import torch
import torch.nn as nn
import sys
sys.path.append("../../../")
from model.layers import LayerNorm2d
from model.loss import AMT_loss, InfoNCE_loss
from model.config import CONFIG
from model.CQT import CQTsmall_fir
from septimbre.septimbre import HCQTpreprocess, NotePredictBranch

class Encoder_joint(nn.Module):
    def __init__(self, emb_dims = 12):
        super().__init__()
        self.k = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.pre_encoding = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(5, 5), padding="same"),
            LayerNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=(25, 3), dilation=(3, 1), padding=((25//2)*3, 1), stride=1),
            LayerNorm2d(8),
        )
        # rescale
        self.neck = nn.Sequential(
            LayerNorm2d(8), ### changed!
            nn.Conv2d(8, 16, kernel_size=(5, 5), padding="same"),
            nn.PReLU()
        )
        self.res1 = nn.Sequential(
            LayerNorm2d(16),
            nn.Conv2d(16, 24, kernel_size=(3, 3), padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 16, kernel_size=(3, 3), padding="same"),
        )
        self.emb = nn.Sequential(
            LayerNorm2d(16),
            nn.Conv2d(16, emb_dims, kernel_size=(3, 5), stride=(3, 1), padding=(1, 2))
        )

    def forward(self, x, frame):
        x = x * self.k + self.b
        # x: (batch, 8, 8*36, len) HCQT
        x = self.pre_encoding(x) + x
        emb = self.neck(x)      # (B, 16, 84, T)
        emb = self.res1(emb) + emb      # (B, 16, 84, T)

        # frame: [B, F, T] -> frame_expand: [B, 1, 3F, T]
        frame_expand = frame.repeat_interleave(3, dim=1).unsqueeze(1)  # (B, 1, 3*F, T)
        emb = nn.functional.normalize(emb, p=2, dim=1) * frame_expand  # (B, 16, 84, T)

        emb = self.emb(emb)  # (B, emb_dims, 84, T)
        return emb  # (batch, emb_dims, 7*12, len)


class SepTimbreAMT_joint(nn.Module):
    def __init__(self, CQTconfig):
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
        self.note_branch = NotePredictBranch()
        dim = 12
        self.topk = dim
        self.encoder = Encoder_joint(dim)
    
    def forward(self, x):
        x = self.cqt(x)
        x = self.hcqt(x)
        note_onset, note_pred = self.note_branch(x)
        emb = self.encoder(x, note_pred.detach())
        emb = nn.functional.normalize(emb, p=2, dim=1)
        return note_onset, note_pred, emb

    def clampK(self, min=0.005, max=3.0):
        with torch.no_grad():
            self.note_branch.k.clamp_(min, max)
            self.encoder.k.clamp_(min, max)

    @staticmethod
    def loss(emb, mask, onset, targets):
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
    model = SepTimbreAMT_joint(CQTconfig=CONFIG.CQT)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")