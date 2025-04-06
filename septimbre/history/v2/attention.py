import torch
import torch.nn as nn
import torch.nn.functional as F

"""
对频率进行位置编码，补偿音色随频率的变化和HS的平移
补偿量和当前幅度有关
"""
class FreqPosEncoding(nn.Module):
    def __init__(self, dims, freqs = 84):
        super().__init__()
        self.adds = nn.Parameter(torch.randn(1, dims, freqs, 1), requires_grad=True)
    
    def forward(self, x):
        # x: (batch, dims, freqs, len)
        eng = (x**2).sum(dim=1, keepdim=True)
        amp = torch.sqrt(eng + 1e-8)
        add = self.adds * amp
        return add + x

"""
加权进行的Hebbian学习 过于原始，不能用
"""
class HebbAttention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, mask = None):
        # x: (batch, dims, freqs, len)
        if mask is None:
            mask = torch.sqrt(x.pow(2).sum(dim=1, keepdim=True) + 1e-8)
            mask_sum = mask.sum(dim=(2, 3), keepdim=True)
            y = x / (mask_sum + 1e-6)
        else:
            # mask: (batch, 1, freqs, len)
            # 由于使用了mask因此每个时频特征都要归一化 先除以幅度归一化，再对mask归一化
            amp = torch.sqrt(x.pow(2).sum(dim=1, keepdim=True) + 1e-8)
            mask_sum = mask.sum(dim=(2, 3), keepdim=True)
            y = x * mask / (amp * mask_sum + 1e-6)
        # y: (batch, dims, freqs, len)

        y_t = y.permute(0, 2, 3, 1)
        connection = torch.matmul(y_t.unsqueeze(-1), y_t.unsqueeze(-2))
        # connection: (batch, freqs, len, dims, dims)
        connection = connection.sum(dim=(1,2), keepdim=False).unsqueeze(1)
        # connection: (batch, 1, dims, dims)
        # 去掉对角线后效果反而不好
        # diag_mask = 1 - torch.eye(connection.size(-1), device=connection.device).unsqueeze(0).unsqueeze(0)
        # connection = connection * diag_mask

        # 注意力修正
        x_ = x.permute(0, 2, 3, 1)
        return torch.matmul(x_, connection).permute(0, 3, 1, 2)


"""
用核函数的线性注意力
"""
class LinearAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.scaleK=nn.Parameter(torch.tensor(1.), requires_grad=True)


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        q: (batch_size, dim, freq, time)
        k: (batch_size, dim, freq, time)
        v: (batch_size, dim, freq, time)
        """
        B, Dim, FreqNum, TimeNum = q.shape
        Q = q.view(B, Dim, -1).contiguous().permute(0, 2, 1)
        K = k.view(B, Dim, -1).contiguous().permute(0, 2, 1)
        V = v.view(B, Dim, -1).contiguous().permute(0, 2, 1)
        # (batch_size, freq * time, dim)

        # 使用 ELU 激活函数对Q和K进行高维映射，并加1以确保非负
        phi_Q = F.elu(Q) + 1
        phi_K = F.elu(K*self.scaleK) + 1

        # 计算 K 和 V 的乘积
        KV = torch.matmul(phi_K.transpose(-2, -1), V)
        # (batch_size, dim, dim)

        # 计算K的累加和，用于归一化因子的计算
        K_sum = phi_K.sum(dim=-2, keepdim=True).transpose(-1, -2)   # (batch_size, dim, 1)

        # 计算归一化因子 Z，注意保留所有维度以确保广播正确
        Z = 1.0 / (torch.matmul(phi_Q, K_sum) + 1e-8) # (batch_size, freq * time, 1)

        # 计算新的 V 值
        V_new = torch.matmul(phi_Q, KV) * Z # (batch_size, freq * time, dim)

        return V_new.permute(0, 2, 1).view(B, Dim, FreqNum, TimeNum)

"""
清华的流注意力，同样线性，有竞争机制
"""
class FlowAttention(nn.Module):
    # Attention with head competing mechanism
    def __init__(self, d_model, d_output, n_heads, drop_out=0.05, eps=1e-6):
        super(FlowAttention, self).__init__()
        self.n_heads = n_heads
        self.out_projection = nn.Linear(d_model, d_output)
        self.dropout = nn.Dropout(drop_out)
        self.eps = eps

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def dot_product(self, q, k, v):
        kv = torch.einsum("nhld,nhlm->nhdm", k, v)
        qkv = torch.einsum("nhld,nhdm->nhlm", q, kv)
        return qkv

    def forward(self, queries, keys, values):
        """
        queries: (batch_size, dim, freq, time)
        keys: (batch_size, dim, freq, time)
        values: (batch_size, dim, freq, time)
        """
        B, Dim, FreqNum, TimeNum = queries.shape
        L = FreqNum * TimeNum
        ## input: B (L or S) D; output: B L D
        ## Note: queries, keys, values have been projected yet
        # 1. reshape
        queries = queries.view(B, Dim, L).contiguous().permute(0, 2, 1)
        keys = keys.view(B, Dim, L).contiguous().permute(0, 2, 1)
        values = values.view(B, Dim, L).contiguous().permute(0, 2, 1)
        # (B, freq * time, d_model)
        queries = queries.view(B, L, self.n_heads, -1).transpose(1, 2)
        keys = keys.view(B, L, self.n_heads, -1).transpose(1, 2)
        values = values.view(B, L, self.n_heads, -1).transpose(1, 2)
        # 2. Non-negative projection
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)
        ## 3. FlowAttention competing on the head dimension
        # (1) Calculate incoming and outgoing flow
        sink_incoming = 1.0 / torch.sum(
            (queries + self.eps) * (keys.sum(dim=1, keepdim=True).repeat(1, self.n_heads, 1, 1) + self.eps), dim=-1)
        source_outgoing = 1.0 / torch.sum(
            (keys + self.eps) * (queries.sum(dim=1, keepdim=True).repeat(1, self.n_heads, 1, 1) + self.eps), dim=-1)
        # (2) conservation refine for source and sink
        conserved_sink = torch.sum((queries + self.eps) * (
                (keys * source_outgoing[:, :, :, None]).sum(dim=1, keepdim=True).repeat(1, self.n_heads, 1,
                                                                                        1) + self.eps), dim=-1)
        conserved_source = torch.sum((keys + self.eps) * (
                (queries * sink_incoming[:, :, :, None]).sum(dim=1, keepdim=True).repeat(1, self.n_heads, 1,
                                                                                         1) + self.eps), dim=-1)
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)  # for stability
        # (3) Competition & Allocation
        sink_allocation = torch.sigmoid(conserved_sink * (float(queries.shape[2]) / float(keys.shape[2])))
        source_competition = torch.softmax(conserved_source, dim=-1) * float(keys.shape[2])
        # (4) dot product
        x = (self.dot_product(queries * sink_incoming[:, :, :, None],  # for value normalization
                              keys,
                              values * source_competition[:, :, :, None])  # competition
             * sink_allocation[:, :, :, None]).transpose(1, 2)  # allocation
        ## (5) Final projection
        x = x.reshape(B, L, -1)
        x = self.out_projection(x)
        x = self.dropout(x)
        return x.permute(0, 2, 1).view(B, Dim, FreqNum, TimeNum)