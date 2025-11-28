import torch
import torch.nn as nn
import sys
sys.path.append("..")
from model.layers import HarmonicaStacking, CBR, EnergyNorm, CBLR, LayerNorm2d, CenterNorm2d
from model.loss import AMT_loss, cluster_loss, InfoNCE_loss
from model.config import CONFIG
from model.CQT import CQTsmall_fir

_harmonics = CONFIG.Harmonics

class HCQTpreprocess(nn.Module):
    def __init__(self):
        super().__init__()
        self.eng = EnergyNorm(output_type=1, log_scale=True)
        self.HCQT = HarmonicaStacking(HarmonicaStacking.harmonic_shifts(_harmonics-1, 1, 36), 7 * 36)

    def forward(self, x):
        # x: (batch, 2, 8*36, len)
        eng = self.eng(x)
        # eng: (batch, 1, 8*36, len)
        stacked = self.HCQT(eng)
        # stacked: (batch, 8, 7*36, len)
        return stacked

class NotePredictBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.k = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(0.0), requires_grad=True)
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
        x = x * self.k + self.b
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
        return yo.squeeze(1), yn.squeeze(1)  # (batch, 7*12, len), (batch, 7*12, len)

    def fromBasicAMT(self, basicamt_model):
        self.k.data = basicamt_model.k.data.clone()
        self.b.data = basicamt_model.b.data.clone()
        self.early_conv.load_state_dict(basicamt_model.early_conv.state_dict())
        self.neck.load_state_dict(basicamt_model.neck.state_dict())
        self.conv_yn.load_state_dict(basicamt_model.conv_yn.state_dict())
        self.conv_yo1.load_state_dict(basicamt_model.conv_yo1.state_dict())
        self.conv_yo2.load_state_dict(basicamt_model.conv_yo2.state_dict())

"""
用于编码CQT输入
"""
class Encoder(nn.Module):
    def __init__(self, emb_dims = 16):
        super().__init__()
        self.k = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.early_conv = nn.Sequential(
            nn.Conv2d(_harmonics, 16, kernel_size=(5, 5), padding="same"),
            nn.GroupNorm(4, 16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, _harmonics, kernel_size=(25, 3), dilation=(3, 1), padding=((25//2)*3, 1), stride=1),
        )
        self.neck = nn.Sequential(
            nn.GroupNorm(2, _harmonics),
            nn.Conv2d(_harmonics, 16, kernel_size=(5, 5), padding="same"),
            nn.PReLU(),
        )
        self.res1 = nn.Sequential(
            LayerNorm2d(16),
            nn.Conv2d(16, 24, kernel_size=(3, 3), padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 16, kernel_size=(3, 3), padding="same"),
        )
        self.emb = nn.Sequential(
            CenterNorm2d(16),
            nn.Conv2d(16, emb_dims, kernel_size=(3, 5), stride=(3, 1), padding=(1, 2))
        )

    def forward(self, x):
        x = x * self.k + self.b
        # x: (batch, 8, 8*36, len) HCQT
        x = self.early_conv(x) + x
        emb = self.neck(x)      # (B, 16, 84, T)
        emb = self.res1(emb) + emb      # (B, 16, 84, T)
        emb = self.emb(emb)  # (B, emb_dims, 84, T)
        return emb  # (batch, emb_dims, 7*12, len)


class SepTimbreAMT(nn.Module):
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
        self.encoder = Encoder(dim)
    
    def forward(self, x):
        x = self.cqt(x)
        x = self.hcqt(x)
        if torch.cuda.is_available():
            stream1 = torch.cuda.Stream()
            stream2 = torch.cuda.Stream()
            torch.cuda.current_stream().synchronize()
            with torch.cuda.stream(stream1):
                note_onset, note_pred = self.note_branch(x)
            with torch.cuda.stream(stream2):
                emb = self.encoder(x)
            torch.cuda.synchronize()
        else:
            note_onset, note_pred = self.note_branch(x)
            emb = self.encoder(x)
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
        return L_cluster, L_amt


### 以下为实际部署时使用的模型
class SepTimbreAMT_44100(nn.Module):
    """
    相比SepTimbreAMT，输入为44100Hz采样率的音频，会先进行降采样到22050Hz
    """
    def __init__(self, septimbre_amt: SepTimbreAMT):
        super().__init__()
        self.septimbre_amt = septimbre_amt

    def forward(self, x):
        # (1, 1, time)
        x = self.septimbre_amt.cqt.down2sample(x)
        onset, note, emb = self.septimbre_amt(x)
        onset /= onset.max()
        note /= note.max()
        # emb已经归一化了
        return onset, note, emb


class Encoder_44100(nn.Module):
    def __init__(self, septimbre_amt: SepTimbreAMT):
        super().__init__()
        self.cqt = septimbre_amt.cqt
        self.hcqt = septimbre_amt.hcqt
        self.encoder = septimbre_amt.encoder
    
    def forward(self, x):
        # (1, 1, time)
        x = self.cqt.down2sample(x)
        x = self.cqt(x)
        x = self.hcqt(x)
        emb = self.encoder(x)
        emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb


if __name__ == "__main__":
    # 输出参数量
    model = SepTimbreAMT(CQTconfig=CONFIG.CQT)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")