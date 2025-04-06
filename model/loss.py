import torch
from itertools import permutations
import copy
import random
import numpy as np

def focal_loss(ref, pred, gamma = 2, alpha = 0.5, eps = 1e-8):
    loss =  - ref * alpha * (1 - pred).pow(gamma) * torch.log(pred + eps) \
            - (1 - ref) * (1 - alpha) * pred.pow(gamma) * torch.log(1 - pred + eps)
    return loss.sum()


def AMT_loss(onset, note, target, mse = False):
    """
    计算AMT音符误差，包含onset和note的focal loss和MSE loss
    最后两个维度的含义:
    onset: [..., note, time]
    note: [..., note, time]
    target: [..., note, time]
    """
    note_ref = (target == 1).float()
    onset_ref = (target == 2).float()
    slience_ref = (target == 0).float()

    # 按理来说应该正例权值大一些以补偿，但是会导致模糊的边缘，特别是onset，alpha为0.95时几乎就是note了。所以和原论文一样，少样本的正例权值应该小
    note_loss = focal_loss(note_ref + onset_ref, note, gamma=1, alpha=0.2)
    onset_loss = focal_loss(onset_ref, onset, gamma=1, alpha=0.06)

    if mse:
        # 加权MSE
        feature_num = target.size(-2)
        mse = (onset + note - target).pow(2)
        ## 再加一维一自动广播 不然乘法做不了
        weight1 = ((feature_num + 2 - note_ref.sum(-2)) / feature_num).unsqueeze(-2)
        weight0 = ((feature_num + 2 - slience_ref.sum(-2)) / feature_num).unsqueeze(-2)
        weight2 = ((feature_num + 2 - onset_ref.sum(-2)) / feature_num).unsqueeze(-2)
        mse = mse * note_ref * weight1 + mse * slience_ref * weight0 + mse * onset_ref * weight2
        mse_loss = mse.sum()

        return (note_loss + onset_loss + mse_loss * 0.5) * 0.666   # 平衡AMTloss和CQTloss。note_loss有一份，onset_loss有一份，mse_loss有两份(三个weight加起来是2)
    else:
        return note_loss + onset_loss

def CQT_loss(pred, target):
    """
    计算CQT重构误差 直接用MSE
    """
    return torch.nn.functional.mse_loss(pred, target, reduction='sum')

def cluster_loss(embs, targets, mode = 0):
    """
    计算聚类损失：谱聚类相似度损失（余弦相似度） 要求输入已经幅度归一化
    Deep clustering: Discriminative embeddings for segmentation and separation
    embs: [batch, dim, freqs, time] 记录了每个时频单元的embedding
    targets: [batch, mix, freqs, time] 传入midiarray即可，大于零的为本类音符
    mode: 0: 对负数无要求，1: 对负数要求正交，2: 对负数要求低一些
    """
    B, D, F, T = embs.size()
    M = targets.size(1)
    targets = (targets > 0.1).float() # 把大于0的都变成1
    losses = []
    for batch in range(B):
        target = targets[batch] # [M, F, T]
        target_mask = target.sum(dim=0, keepdim=False) > 0  # [F, T]
        V = embs[batch].permute(1, 2, 0)[target_mask, :].view(-1, D) # [N, D]
        Y = target.permute(1, 2, 0)[target_mask, :].view(-1, M) # [N, M]

        if mode == 1: # 要求正交 简化
            VTV = torch.matmul(V.transpose(-1, -2), V) # [D, D]
            YTY = torch.matmul(Y.transpose(-1, -2), Y) # [M, M]
            VTY = torch.matmul(V.transpose(-1, -2), Y) # [D, M]
            losses.append(VTV.pow(2).sum() + YTY.pow(2).sum() - 2 * VTY.pow(2).sum())
        elif mode == 2: # 对负数要求低一些，但不能简化，开销大
            VVT = torch.matmul(V, V.transpose(-1, -2)) # [N, N]
            YYT = torch.matmul(Y, Y.transpose(-1, -2)) # [N, N]
            YYT[range(YYT.size(0)), range(YYT.size(0))] = 1
            losses.append((torch.nn.functional.leaky_relu(VVT, 0.3) - YYT).pow(2).sum())
        else: # 对负数无要求
            VVT = torch.matmul(V, V.transpose(-1, -2)) # [N, N]
            YYT = torch.matmul(Y, Y.transpose(-1, -2)) # [N, N]
            YYT[range(YYT.size(0)), range(YYT.size(0))] = 1
            mask = (YYT < 0.01) & (VVT < 0)
            losses.append((VVT * (~mask).float() - YYT).pow(2).sum())

    return torch.stack(losses).sum()


def PIT_loss(onset, note, CQT, target_midi, target_CQT):
    """
    计算两个误差：AMT音符误差和CQT重构误差
    dataloader给出的是混合前:
    input: [batch, mix, 2, freq, time]
    target: [batch, mix, notes, time]
    实际输入是torch.sum(input, dim=1, keepdim=False): [batch, 2, freq, time]
    每一次的输出是: [batch, 2, freq, time]和[batch, notes, time]*2（onset和note）
    堆叠后得到：[batch, mix, 2, freq, time]和[batch, mix, notes, time]
    每个batch的排列都不一样，所以loss要分batch用PIT
    """
    batch_size = target_midi.size(0)
    mix = target_midi.size(1)
    # 生成所有排列
    perms = [list(perm) for perm in permutations(range(mix))]
    losses = []
    for batch in range(batch_size):
        _onset = onset[batch]
        _note = note[batch]
        _CQT = CQT[batch]
        _target_midi = target_midi[batch]
        _target_CQT = target_CQT[batch]
        loss_perm = torch.stack([(
            AMT_loss(_onset[perm], _note[perm], _target_midi) +
            CQT_loss(_CQT[perm], _target_CQT)
        ) for perm in perms])
        losses.append(loss_perm.min())
    return torch.stack(losses).sum()    # 保持数据在GPU上

#=================MultiTask===================#
"""
针对损失量级差别大的情况，模仿电阻并联，得到归一化权值
"""
class LossNorm():
    def __init__(self, alpha = 0.9):
        self.last = None
        self.alpha = alpha
    
    def __call__(self, losses: torch.Tensor):
        now = LossNorm.w(losses)
        if self.last is None:
            self.last = now
        else:
            self.last = self.alpha * self.last + (1 - self.alpha) * now
        return self.last

    @staticmethod
    def w(losses: torch.Tensor):
        common = 1 / sum(1 / loss.item() for loss in losses)
        return torch.tensor([common / loss.item() for loss in losses]).to(losses.device)

    @staticmethod
    def norm_sum(losses: torch.Tensor):
        return len(losses) / (1 / losses).sum()


class DWA():
    def __init__(self, loss_num = 2, K = 1.0, T = 1.0):
        self.loss_history = [[None, None] for _ in range(loss_num)]
        self.K = K
        self.T = T
    
    def set_init(self, weights: list):
        if len(weights) != len(self.loss_history):
            raise ValueError("The length of weights must be equal to the number of losses")
        w = np.array(weights)
        # 归一
        w = w / w.sum()
        # 反求softmax
        w = np.log(w)
        w = w - w.min() + 1 # 防止分母为0
        for i in range(len(self.loss_history)):
            self.loss_history[i][1] = w[i]
            self.loss_history[i][0] = 1 / self.T

    def __call__(self, losses: torch.Tensor):
        speed = []
        for loss_history, loss in zip(self.loss_history, losses):
            l = loss.item()
            L2 = loss_history.pop(0)
            if L2 is None:
                L2 = l
            L1 = loss_history[0]
            if L1 is None:
                L1 = l
            loss_history.append(l)
            speed.append(L1/(L2*self.T))
        return self.K * torch.softmax(torch.tensor(speed), dim=0).to(losses.device)


"""
PCGrad: Projected Conflicting Gradient for Multi-Task Learning
https://github.com/WeiChengTseng/Pytorch-PCGrad
"""
class PCGrad():
    def __init__(self, optimizer, reduction='mean'):
        self._optim, self._reduction = optimizer, reduction
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def pc_backward(self, objectives):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''

        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).sum(dim=0)
        else: exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific 
        objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad