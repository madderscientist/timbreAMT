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
    def __init__(self, emb_in = 0, emb_out = 0, dropout = 0.05):
        super().__init__()
        if emb_in > 0:
            self.linear = nn.Linear(emb_in, emb_out)
            self.dropout = nn.Dropout(dropout)
        else:
            self.linear = None


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
        phi_K = F.elu(K) + 1


        # 计算 K 和 V 的乘积
        KV = torch.matmul(phi_K.transpose(-2, -1), V)
        # (batch_size, dim, dim)

        # 计算K的累加和，用于归一化因子的计算
        K_sum = phi_K.sum(dim=-2, keepdim=True).transpose(-1, -2)   # (batch_size, dim, 1)

        # 计算归一化因子 Z，注意保留所有维度以确保广播正确
        Z = 1.0 / (torch.matmul(phi_Q, K_sum) + 1e-8) # (batch_size, freq * time, 1)

        # 计算新的 V 值
        V_new = torch.matmul(phi_Q, KV) * Z # (batch_size, freq * time, dim)

        if self.linear:
            V_new = self.linear(V_new)
            V_new = self.dropout(V_new)

        return V_new.permute(0, 2, 1).view(B, -1, FreqNum, TimeNum)

"""
清华的流注意力，同样线性，有竞争机制
"""
class FlowAttention(nn.Module):
    # Attention with head competing mechanism
    def __init__(self, n_heads, eps=1.01e-8):
        super(FlowAttention, self).__init__()
        self.n_heads = n_heads
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
        # 3. FlowAttention competing on the head dimension
        ## (1) Calculate incoming and outgoing flow
        sink_incoming = 1.0 / torch.sum(
            (queries + self.eps) * (keys.sum(dim=1, keepdim=True).repeat(1, self.n_heads, 1, 1) + self.eps), dim=-1)
        source_outgoing = 1.0 / torch.sum(
            (keys + self.eps) * (queries.sum(dim=1, keepdim=True).repeat(1, self.n_heads, 1, 1) + self.eps), dim=-1)
        ## (2) conservation refine for source and sink
        conserved_sink = torch.sum((queries + self.eps) * (
                (keys * source_outgoing[:, :, :, None]).sum(dim=1, keepdim=True).repeat(1, self.n_heads, 1,
                                                                                        1) + self.eps), dim=-1)
        conserved_source = torch.sum((keys + self.eps) * (
                (queries * sink_incoming[:, :, :, None]).sum(dim=1, keepdim=True).repeat(1, self.n_heads, 1,
                                                                                         1) + self.eps), dim=-1)
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)  # for stability
        ## (3) Competition & Allocation
        sink_allocation = torch.sigmoid(conserved_sink * (float(queries.shape[2]) / float(keys.shape[2])))
        source_competition = torch.softmax(conserved_source, dim=-1) * float(keys.shape[2])
        ## (4) dot product
        x = (self.dot_product(queries * sink_incoming[:, :, :, None],  # for value normalization
                              keys,
                              values * source_competition[:, :, :, None])  # competition
             * sink_allocation[:, :, :, None]).transpose(1, 2)  # allocation
        ## (5) Final projection
        x = x.reshape(B, L, -1)
        return x.permute(0, 2, 1).view(B, Dim, FreqNum, TimeNum)
    

class Mobile_Attention(nn.Module):
    # Mobile Attention with head competing mechanism
    def __init__(self, d_model, d_output, n_heads, drop_out=0.05, eps=1.01e-8):
        super(Mobile_Attention, self).__init__()
        self.n_heads = n_heads
        self.out_projection = nn.Linear(d_model, d_output)
        self.dropout = nn.Dropout(drop_out)
        self.eps = eps

    def kernel_method(self, x):
        return torch.nn.functional.elu(x) + 1.0
        # return torch.sigmoid(x)

    def dot_product(self, q, k, v):
        kv = torch.einsum("nhld,nhlm->nhdm", k, v)
        qkv = torch.einsum("nhld,nhdm->nhlm", q, kv)
        return qkv

    def forward(self, queries, keys, values):
        ## input: B (L or S) D; output: B L D
        ## Note: queries, keys, values are not projected yet
        ## 1. Linear projection
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = queries.view(B, L, self.n_heads, -1)  # (B, L, n_heads, D)
        keys = keys.view(B, S, self.n_heads, -1)    # (B, S, n_heads, D)
        values = values.view(B, S, self.n_heads, -1)  # (B, S, n_heads, D)
        queries = queries.transpose(1, 2)   # (B, n_heads, L, D)
        keys = keys.transpose(1, 2)  # (B, n_heads, S, D)
        values = values.transpose(1, 2)  # (B, n_heads, S, D)
        # 2. Non-negative projection
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)
        ## 3. Flow-Attention
        # (1) Calculate incoming and outgoing flow
        sink_incoming = 1.0 / (torch.einsum("nhld,nhd->nhl", queries + self.eps, keys.sum(dim=2) + self.eps))
        source_outgoing = 1.0 / (torch.einsum("nhld,nhd->nhl", keys + self.eps, queries.sum(dim=2) + self.eps))
        # (2) conservation refine for source and sink
        conserved_sink = torch.einsum("nhld,nhd->nhl", queries + self.eps,
                                      (keys * source_outgoing[:, :, :, None]).sum(dim=2) + self.eps)
        conserved_source = torch.einsum("nhld,nhd->nhl", keys + self.eps,
                                        (queries * sink_incoming[:, :, :, None]).sum(dim=2) + self.eps)
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
        return x


class OcvtaveAtten(nn.Module):
    def __init__(self, R = 5, D = 12):
        super(OcvtaveAtten, self).__init__()
        self.R = R
        self.atten = Mobile_Attention(d_model=D, d_output=D, n_heads=1)

    def forward(self, x):
        B, D, F, T = x.size()
        # x: (batch, dims, 84, len)
        x_pad = torch.nn.functional.pad(x, (0, 0, self.R, self.R), mode='constant', value=0)
        neighbour = self.R * 2 + 1
        x_unfold = x_pad.unfold(dimension=2, size=neighbour, step=1)  # (batch, dims, 84, len, neighbour)
        x_unfold = x_unfold.reshape(x_unfold.size(0), x_unfold.size(1), x_unfold.size(2), -1)    # (batch, dims, 84, len*neighbour)
        # Q是x的每一行，K和V是领域
        Q = x.permute(0, 2, 3, 1).contiguous().view(-1, T, D)   # (batch*84, len, dims)
        K = x_unfold.permute(0, 2, 3, 1).contiguous().view(B*F, -1, D)   # (batch*84, len*neighbour, dims)
        # print(Q.shape, K.shape)
        attended = self.atten(Q, K, K)   # (batch*84, len, dims)
        attended = attended.view(B, F, T, D).permute(0, 3, 1, 2)   # (batch, dims, 84, len)
        return attended
    

class SlotAttention(nn.Module):
    def __init__(self, inputdim, hiddendim, iters = 1, topk = None, slots = None, norm = nn.LayerNorm, finalNorm = True):
        super().__init__()
        self.iters = iters
        self.topk = topk
        self.slots = slots
        self.hiddendim = hiddendim
        self.proj_k = nn.Linear(inputdim, hiddendim)
        self.proj_v = nn.Linear(inputdim, hiddendim)
        self.proj_Q = nn.Linear(inputdim, hiddendim)
        self.proj_back_V = nn.Linear(hiddendim, inputdim)
        self.norm_1 = norm(inputdim)
        self.finalNorm = finalNorm
        if finalNorm:
            self.norm_2 = norm(inputdim)
        self.ffn = nn.Sequential(
            nn.Linear(inputdim, inputdim*2),
            nn.GELU(),
            nn.Linear(inputdim*2, inputdim)
        )
        self.atten_scale = hiddendim ** -0.5
        self.atten_temperature1 = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.atten_temperature2 = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, inputs):
        b, n, d = inputs.shape

        if self.slots is None:
            # 用eye和-eye生成d*2个方向
            eye = torch.eye(self.hiddendim, device=inputs.device)
            slots = torch.cat([eye, -eye], dim=0)  # (hiddendim*2, hiddendim)
            slots = slots.unsqueeze(0).expand(b, -1, -1).contiguous()  # (b, hiddendim*2, hiddendim)
        else:
            slots = torch.randn(b, self.slots, self.hiddendim, device=inputs.device)    # (b, n_s, hiddendim)
            slots = torch.nn.functional.normalize(slots, p=2, dim=-1)


        Q = slots   # (b, n_s, hiddendim)
        # K = inputs  # (b, n, d)
        K = self.proj_k(inputs)  # (b, n, hiddendim)
        # V = inputs  # (b, n, d)
        V = self.proj_v(inputs)  # (b, n, hiddendim)

        for _ in range(self.iters):
            dots = torch.einsum('bid,bjd->bij', Q, K) * self.atten_scale  # (B, n_s, n)

            if self.topk is not None and self.topk < dots.size(-1):
                topk_vals, topk_idx = torch.topk(dots, k=self.topk, dim=-1)  # (B, n_s, k)
                attn_topk = (topk_vals / self.atten_temperature1).softmax(dim=-1)  # (B, n_s, k)
                attn = torch.zeros_like(dots)  # (B, n_s, n)
                attn.scatter_(dim=-1, index=topk_idx, src=attn_topk)
            else:
                # Full attention
                attn = dots.softmax(dim=-1)  # (B, n_s, n)

            # Aggregate: (B, n_s, n) @ (B, n, d) -> (B, n_s, d)
            slots = torch.einsum('bij,bjd->bid', attn, V)
            slots = torch.nn.functional.normalize(slots, p=2, dim=-1)

            # Update Q for next iteration
            Q = slots  # or Q = self.to_q(slots) if you add projection later

        emb_Q = self.proj_Q(inputs)  # (b, n, hiddendim)
        slots_K = slots # (b, n_s, hiddendim)
        slots_V = self.proj_back_V(slots)  # (b, n_s, inputdim)

        logits = torch.einsum('bid,bjd->bij', emb_Q, slots_K) * self.atten_scale  # (B, n, n_s)
        topk_vals, topk_idx = torch.topk(logits, k=slots.size(1)//3, dim=-1)
        atten = torch.zeros_like(logits)
        atten.scatter_(dim=-1, index=topk_idx, src=(topk_vals / self.atten_temperature2).softmax(dim=-1))
        emb_refined = torch.einsum('bij,bjd->bid', atten, slots_V)  # (B, n, inputdim)

        emb = self.norm_1(emb_refined + inputs)  # (B, n, inputdim)
        ffn_out = self.ffn(emb)
        emb = emb + ffn_out
        if self.finalNorm:
            emb = self.norm_2(emb)
        return emb  # (B, n, inputdim)