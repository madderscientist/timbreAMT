import torch
import torch.nn as nn

class LearnablePositionalEncoding(nn.Module):
    """
    可学习的位置编码 但是要求长度已知 适合在频率轴上位置编码
    """
    def __init__(self, emb_dim, seq_len):
        super().__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, seq_len, emb_dim))
        nn.init.xavier_uniform_(self.position_embeddings)  # 初始化参数
   
    def forward(self, x):
        return x + self.position_embeddings


class Attention(nn.Module):
    """
    一次注意力机制，包括多头注意力、残差连接和LayerNorm
    """
    def __init__(self, emb_dim, n_heads, dropout=0.1):
        super().__init__()
        self.freq_attention = nn.MultiheadAttention(emb_dim, n_heads, batch_first=True, dropout=dropout)
        self.layernorm = nn.LayerNorm(emb_dim)
   
    def forward(self, x):
        # x: (batch, seq, feature)
        y, _ = self.freq_attention(x, x, x)
        # y: (batch*len, freq_bins, channels)
        return self.layernorm(x + y)


class DualAttention(nn.Module):
    def __init__(self, channels, n_heads = 2, dropout=0.1, position_encoding = False, seq_len = 252):
        super().__init__()
        # 时序数据输入都是(batch, seq, feature)
        self.time_attention = nn.LSTM(channels, channels, 1, batch_first=True, bidirectional=True)
        self.position_encoding = LearnablePositionalEncoding(channels * 2, seq_len) if position_encoding else None
        self.freq_attention = nn.MultiheadAttention(channels * 2, n_heads, batch_first=True, dropout=dropout)
        self.layernorm = nn.LayerNorm(channels * 2)
        self.FFN = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.SiLU()
        )
        # 在此之前进行残差连接
        self.layernorm2 = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (batch, channels, freq_bins, len)
        batch_size, channels, freq_bins, length = x.shape

        # 首先在时间上进行注意力，用LSTM
        ## 整形
        time = x.permute(0, 2, 3, 1).contiguous()
        # time: (batch, freq_bins, len, channels)
        time = time.view(batch_size*freq_bins, length, channels)
        # time: (batch*freq_bins, len, channels)
        ## 每个频率同步处理
        time, _ = self.time_attention(time)
        # time: (batch*freq_bins, len, channels*2)
        time = time.view(batch_size, freq_bins, length, -1)

        # 其次在频率上进行self-attention
        ## 整形
        time = time.permute(0, 2, 1, 3).contiguous()
        # time: (batch, len, freq_bins, channels*2)
        time = time.view(batch_size*length, freq_bins, -1)
        # time: (batch*len, freq_bins, channels*2)
        if self.position_encoding:
            time = self.position_encoding(time)
        ## 每个时间同步处理
        freq, _ = self.freq_attention(time, time, time)
        # y: (batch*len, freq_bins, channels*2)
        time_freq = time + freq
        time_freq = self.layernorm(time_freq)
        # time_freq: (batch*len, freq_bins, channels*2)
        time_freq = self.FFN(time_freq)
        # time_freq: (batch*len, freq_bins, channels)

        # 残差处理 和原论文不太一样
        x = x.permute(0, 3, 2, 1).contiguous()
        # x: (batch, len, freq_bins, channels)
        x = x.view(batch_size*length, freq_bins, channels)
        # x: (batch*len, freq_bins, channels)
        x = x + time_freq
        x = self.layernorm2(x)
        # x: (batch*len, freq_bins, channels)
        x = x.view(batch_size, length, freq_bins, channels)
        x = x.permute(0, 3, 2, 1).contiguous()
        # x: (batch, channels, freq_bins, len)
        return x


class TimbreAttention(nn.Module):
    """
    用注意力机制提取频率响应
    试图模拟聚类算法，但是只找到明显的一类。无参数
    """
    def __init__(self, emb_dim, time_step = 3, topk = 84):
        super().__init__()
        self.time_step = time_step
        self.emb_dim = emb_dim * time_step
        self.topk = topk
        self.softmax = nn.Softmax(dim=-1)
        self.w = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        """
        x: (batch, channels, freq_bins, len) 频谱图
        output: (batch, channel, 1, self.time_step) 冲激响应
        """
        # x: (batch, channels, freq_bins, len)
        batch_size, channels, freq_bins, length = x.shape

        # 首先进行时间堆叠
        shifted = torch.zeros((batch_size, self.emb_dim, freq_bins, length), device=x.device, dtype=x.dtype)
        last_time = x[:, :, :, -1].unsqueeze(-1)
        for i in range(self.time_step):
            # 前移并堆叠
            shifted[:, i*channels:(i+1)*channels, :, :length-i] = x[:, :, :, i:length]
            # 后面补上最后一个时间步的复制
            shifted[:, i*channels:(i+1)*channels, :, length-i:] = last_time.expand(-1, -1, -1, i)
        # shifted: (batch, emb_dim, freq_bins, len)
        shifted = shifted.permute(0, 2, 3, 1).contiguous().view(batch_size, freq_bins*length, self.emb_dim)
        # shifted: (batch, freq_bins*len, emb_dim)

        # 之后进行topk注意力
        query = shifted.mean(dim=1, keepdim=True)
        # query: (batch, 1, emb_dim)
        attention_scores = torch.bmm(query, shifted.transpose(1, 2))
        # attention_scores: (batch, 1, freq_bins*len)
        topk_scores, topk_indices = torch.topk(attention_scores, self.topk, dim=-1)
        # topk_scores: (batch, 1, topk)
        # topk_indices: (batch, 1, topk)
        topk_indices = topk_indices.expand(-1, self.emb_dim, -1).permute(0, 2, 1)   # (batch, topk, emb_dim)
        topk_values = shifted.gather(1, topk_indices)
        # topk_values: (batch, topk, emb_dim)
        topk_scores = self.softmax(topk_scores)
        # topk_scores: (batch, 1, topk)
        topk_values = torch.bmm(topk_scores, topk_values) * self.w + query* (0.5 - self.w)
        # topk_values: (batch, 1, emb_dim)
        return topk_values.view(batch_size, channels, 1, self.time_step)
        # topk_values: (batch, channel, 1, self.time_step)


class TimbreGAP(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, weight_mask):
        """
        x: (batch, channels, freq_bins, len)
        weight_mask: (batch, 1, freq_bins, len)
        """
        weighted = x * weight_mask
        # weighted: (batch, channels, freq_bins, len)
        meanb = weighted.sum(dim=(2, 3)) / weight_mask.sum(dim=(2, 3))
