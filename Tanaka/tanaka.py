import torch
import torch.nn as nn
import sys
sys.path.append('..')
from model.loss import AMT_loss, InfoNCE_loss
from septimbre.septimbre import HCQTpreprocess, NotePredictBranch
from model.config import CONFIG
from model.CQT import CQTsmall_fir

class Encoder(nn.Module):
    def __init__(self, input_dim, H = 256, D = 12):
        super().__init__()
        self.bilstm = nn.LSTM(input_dim, H, num_layers=3, batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(
            nn.Linear(2*H, D*84),
            nn.Tanh()
        )
    
    def forward(self, x):
        # x: [batch, input_dim, time]
        x = x.permute(0, 2, 1)  # (batch, time, input_dim)
        output, _ = self.bilstm(x)  # (batch, time, 2*H)
        emb = self.proj(output)  # (batch, time, D*84)
        emb = emb.view(emb.size(0), emb.size(1), -1, 84)  # (batch, time, D, 84)
        emb = emb.permute(0, 2, 3, 1)  # (batch, D, 84, time)
        return emb


class Tanaka(nn.Module):
    n_fft_under44100 = 1024
    def __init__(self, CQTconfig = CONFIG.CQT, dim=12):
        super().__init__()
        sr = CONFIG.CQT.fs
        self.hop = CONFIG.CQT.hop
        self.n_fft = round(Tanaka.n_fft_under44100 * (sr / 44100))
        stft_bins = self.n_fft // 2 + 1
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
        window = torch.hann_window(self.n_fft, periodic=True)
        self.register_buffer('window', window)
        self.encoder = Encoder(input_dim=84+stft_bins, D=dim, H=256)
    

    def forward_note_branch(self, x):
        hcqt = self.hcqt(self.cqt(x))  # (batch, 1, 7*36, len)
        note_onset, note_pred = self.note_branch(hcqt)  # (batch, 7*12, len), (batch, 7*12, len)
        return note_onset, note_pred

    def prepare_stft(self, x):
        if x.dim() == 3:
            if x.size(1) > 1:
                x_ = x.mean(dim=1, keepdim=False)
            else:
                x_ = x.squeeze(dim=1)
        stft = torch.stft(
            input=x_,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.n_fft,
            window=self.window, # type: ignore
            center=True,           # 默认 True，等价于 librosa（对称 padding）
            pad_mode='reflect',    # 与 librosa 默认一致（当 center=True 时）
            normalized=False,
            onesided=True,         # 实信号只需正频率（默认）
            return_complex=True
        )
        mag = torch.abs(stft)   # (batch, stft_bins, time)
        return mag

    def forward_emb(self, mag, note_pred):
        if mag.shape[-1] > note_pred.shape[-1]:
            mag = mag[..., :note_pred.shape[-1]]
        elif mag.shape[-1] < note_pred.shape[-1]:
            pad = note_pred.shape[-1] - mag.shape[-1]
            mag = nn.functional.pad(mag, (0, pad))
        emb_input = torch.cat([note_pred, mag], dim=1)  # (batch, 84 + stft_bins, time)
        emb = self.encoder(emb_input)  # (batch, D, 84, time)
        emb = nn.functional.normalize(emb, p=2, dim=1)  # L2 normalize
        return emb

    def forward(self, x):
        if torch.cuda.is_available():
            stream1 = torch.cuda.Stream()
            stream2 = torch.cuda.Stream()
            # 预分配变量
            note_onset = note_pred = mag = None
            with torch.cuda.stream(stream1):
                note_onset, note_pred = self.forward_note_branch(x)
            with torch.cuda.stream(stream2):
                mag = self.prepare_stft(x)
            torch.cuda.synchronize()
        else:
            note_onset, note_pred = self.forward_note_branch(x)
            mag = self.prepare_stft(x)
        emb = self.forward_emb(mag, note_pred)
        return note_onset, note_pred, emb

    def clampK(self, min=0.005, max=3.0):
        with torch.no_grad():
            self.note_branch.k.clamp_(min, max)

    @staticmethod
    def loss(emb, mask, onset, targets):
        """
        计算聚类损失和AMT损失
        emb: (batch, emb_dims, 7*12, time)
        mask: (batch, 7*12, time)
        onset: (batch, 7*12, time)
        targets: (batch, mix, freqs, time)
        """
        L_cluster = InfoNCE_loss(emb, targets, 0.15)
        if isinstance(targets, list):
            targets = [t.max(dim=0, keepdim=False).values for t in targets]
            targets = torch.stack(targets, dim=0)
        else:
            targets = targets.max(dim=-3, keepdim=False).values
        L_amt = AMT_loss(onset, mask, targets, False)
        return L_cluster, L_amt


if __name__ == "__main__":
    # 输出参数量
    model = Tanaka()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")
    Encoder_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    print(f"Encoder parameters: {Encoder_params}")