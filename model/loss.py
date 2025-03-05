import torch

def focal_loss(ref, pred, gamma = 2, alpha = 0.5, eps = 1e-8):
    loss =  - ref * alpha * (1 - pred).pow(gamma) * torch.log(pred + eps) \
            - (1 - ref) * (1 - alpha) * pred.pow(gamma) * torch.log(1 - pred + eps)
    return loss.sum()

def midi_weight_loss(x, reconst):
    """
    计算midi损失。支持batch（只关心最后两维）
    x: 目标，0为无音符，1为音符，2为onset
    reconst: 模型输出的midi
    """
    mse = (x - reconst).pow(2)
    # mse = torch.nn.functional.mse_loss(reconst_x, x, reduction='sum')
    # 我想修正一下。因为label的0的个数很多 1的个数很少 2的个数更少
    feature_num = x.size(-1)
    # ↓ [batch, seq_len, feature_num] 或者没有batch
    num1 = (x == 1).float()
    num2 = (x == 2).float()
    num0 = (x == 0).float()
    # ↓ [batch, seq_len] 
    weight1 = ((feature_num + 2 - num1.sum(-2)) / feature_num).unsqueeze(-2)
    weight0 = ((feature_num + 2 - num0.sum(-2)) / feature_num).unsqueeze(-2)
    weight2 = ((feature_num + 2 - num2.sum(-2)) / feature_num).unsqueeze(-2)
    mse = mse * num1 * weight1 + mse * num0 * weight0 + mse * num2 * weight2
    return mse.sum()

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