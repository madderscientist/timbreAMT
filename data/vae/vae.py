import torch
import torch.nn as nn
import numpy as np
import os

class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = "same"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

"""时间序列生成器"""
class VAEt(nn.Module):
    def __init__(self, feature_num = 84):
        super().__init__()
        self.feature_num = feature_num
        self.latent_dim = feature_num // 4

        self.conv_enc = nn.Sequential(
            CBR(1, 8, (5,3), 1),
            CBR(8, 2, 3, 1)
        )   # [batch_size, 2, seq_len, feature_num]
        # 再concat一下，变成3维 为了让LSTM直接关注原始数据
        self.lstm_enc = nn.GRU(3 * feature_num, self.latent_dim, 1, batch_first=True)
        self.lstm_var = nn.GRU(3 * feature_num, self.latent_dim, 1, batch_first=True)

        # 来个反向抵消上面的正向
        self.gru_dec = nn.GRU(self.latent_dim, 2 * feature_num, 2, batch_first=True, bidirectional=True)
        # 记为gru_dec
        # 降噪
        self.conv_dec = nn.Sequential(
            CBR(4, 16, (5,3), 1),
            CBR(16, 16, 3, 1),
            CBR(16, 4, 3, 1)
        )
        # 记为conv_dec
        # 和gru_dec cat一下
        self.conv_note = nn.Sequential(
            CBR(8, 16, 3, 1),
            nn.Conv2d(16, 1, kernel_size=3, padding="same"),
            nn.Sigmoid()
        )
        # 记为conv_note
        # 和之前的都cat一下
        self.conv_onset = nn.Sequential(
            CBR(9, 8, 3, 1, 1),
            nn.Conv2d(8, 1, kernel_size=3, padding="same"),
            nn.Sigmoid()
        )

    def flat_img_to_seq(self, x):
        # x: [batch_size, C, seq_len, feature_num]
        return x.permute(0, 2, 1, 3).contiguous().view(x.size(0), x.size(2), -1)
        # [batch_size, seq_len, C * feature_num]
    
    def flat_seq_to_img(self, x):
        # x: [batch_size, seq_len, C * feature_num]
        return x.view(x.size(0), x.size(1), -1, self.feature_num).permute(0, 2, 1, 3).contiguous()
        # [batch_size, C, seq_len, feature_num]

    def encode(self, x):
        # x: [batch_size, seq_len, feature_num]
        x_2d = x.unsqueeze(1)  # 不能用unsqueeze_ 会改变数据集形状
        # x_2d: [batch_size, 1, seq_len, feature_num]
        conved = self.conv_enc(x_2d)
        # conved: [batch_size, 2, seq_len, feature_num]
        conved = self.flat_img_to_seq(conved)
        # conved: [batch_size, seq_len, 2 * feature_num]
        x = torch.cat((x, conved), -1)
        # x: [batch_size, seq_len, 3 * feature_num]
        mu, _ = self.lstm_enc(x)
        log_var, _ = self.lstm_var(x)
        # mu & log_var: [batch_size, seq_len, latent_dim]
        return mu, log_var

    def decode(self, x):
        # x: [batch_size, seq_len, latent_dim]
        x, _ = self.gru_dec(x)
        # x: [batch_size, seq_len, 4 * feature_num]
        gru_edc = self.flat_seq_to_img(x)
        # gru_edc: [batch_size, 4, seq_len, feature_num]
        conv_dec = self.conv_dec(gru_edc)
        # conv_dec: [batch_size, 4, seq_len, feature_num]
        note_input = torch.cat((conv_dec, gru_edc), 1)
        # note_input: [batch_size, 8, seq_len, feature_num]
        note = self.conv_note(note_input)
        # note: [batch_size, 1, seq_len, feature_num]
        onset_input = torch.cat((note_input, note), 1)
        # onset_input: [batch_size, 9, seq_len, feature_num]
        onset = self.conv_onset(onset_input)
        # onset: [batch_size, 1, seq_len, feature_num]
        return note.squeeze(1), onset.squeeze(1)

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.sample(mu, log_var)
        note, onset = self.decode(z)
        return note, onset, mu, log_var
    
    def generate(self, len, device, num = 1):
        # LSTM输出范围是(-1,1)
        z = torch.randn(num, len, self.latent_dim).to(device)
        return self.decode(z)


def focal_loss(ref, pred, gamma = 1.5):
    # ref: [batch_size, seq_len, feature_num]
    # pred: [batch_size, seq_len, feature_num]
    loss = - ref * (1 - pred).pow(gamma) * torch.log(pred) - (1 - ref) * pred.pow(gamma) * torch.log(1 - pred)
    return loss.sum()

def midi_weight_loss(x, reconst):
    mse = (x - reconst).pow(2)
    # mse = torch.nn.functional.mse_loss(reconst_x, x, reduction='sum')
    # 我想修正一下。因为label的0的个数很多 1的个数很少 2的个数更少
    feature_num = x.size(-1)
    # ↓ [batch, seq_len, feature_num]
    num1 = (x == 1).float()
    num2 = (x == 2).float()
    num0 = (x == 0).float()
    # ↓ [batch, seq_len] 再加一维一自动广播 不然乘法做不了
    weight1 = ((feature_num + 2 - num1.sum(-1)) / feature_num).unsqueeze(-1)
    weight0 = ((feature_num + 2 - num0.sum(-1)) / feature_num).unsqueeze(-1)
    weight2 = ((feature_num + 2 - num2.sum(-1)) / feature_num).unsqueeze(-1)
    mse = mse * num1 * weight1 + mse * num0 * weight0 + mse * num2 * weight2
    return mse.sum()


def loss_function(ref, note, onset, mu, log_var):
    """
    :param x: [batch_size, seq_len, feature_num] 实际的
    :param reconst_x: [batch_size, seq_len, feature_num] 重构的
    :param mu: [batch_size, seq_len, latent_dim]
    :param log_var: [batch_size, seq_len, latent_dim]
    """
    note_ref = (ref > 0).float()
    note_loss = focal_loss(note_ref, note)
    onset_ref = (ref == 2).float()
    onset_loss = focal_loss(onset_ref, onset)
    mse = midi_weight_loss(ref, note + onset)
    kld = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return mse + note_loss + onset_loss + kld

class myDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, time_first = False):
        self.data_folder = data_folder
        self.time_first = time_first
        self.data_files = os.listdir(data_folder)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.data_folder, self.data_files[idx]))
        if not self.time_first:
            data = data.transpose(1, 0)
        return torch.from_numpy(data).float()
