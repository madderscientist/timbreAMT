"""全连接卷积层"""
import torch
import torch.nn as nn


def denseConv2d_2(response, impulse):
    batch_size, channel_r, height_r, width_r = response.shape
    batches = []
    for b in range(batch_size):   # 对每个batch单独计算
        _output = []
        _impulse = impulse[b].unsqueeze(0)
        _response = response[b].unsqueeze(1) # [channel_r, 1, height, width]
        for ch in range(impulse.size(1)):
            _output.append(
                nn.functional.conv_transpose2d(
                    _impulse[:, ch:ch+1],
                    _response,
                    stride=1
                )[:, :, :height_r, :width_r]
            )   # [1, channel_r, height_i, width_i]
        batches.append(torch.cat(_output, dim=1))   # [1, channel_i*channel_r, height_i, width_i]
    return torch.cat(batches, dim=0)    # [batch, channel_i*channel_r, height_i, width_i]


def denseConv2d_1(response, impulse):
    """
    密集卷积，两个输入的层与层之间相互卷积
    input
        response: [batch, channel_r, height_r, width_r] 卷积核
        impulse:  [batch, channel_i, height_i, width_i] 冲激图
    output
        [batch, channel_i*channel_r, height_i, width_i]
    """
    batch_size, channel_r, height_r, width_r = response.shape
    _, channel_i, height_i, width_i = impulse.shape
    batches = [
        nn.functional.conv_transpose2d(           # [channel_i, channel_r, height_i, width_i]
            impulse[b].unsqueeze(1),    # (batch, in_channels = 1, kernel_size, kernel_size)
            response[b].unsqueeze(0),   # (in_channels = 1, out_channels, kernel_size, kernel_size)
            stride=1
        )[:, :, :height_i, :width_i].view(
            1, channel_r*channel_i, height_i, width_i
        ) for b in range(batch_size)
    ]   # 对每个batch单独计算
    return torch.cat(batches, dim=0)    # [batch, channel_i*channel_r, height_i, width_i]


def denseConv2d_0(response, impulse):
    batch_size, channel_r, height_r, width_r = response.shape
    _, channel_i, height_i, width_i = impulse.shape
    output = nn.functional.conv_transpose2d(
        impulse.transpose(0, 1),    # [channel_i, batch, height_i+height_r-1, width_i+width_r-1]
        response.view(1, batch_size * channel_r, height_r, width_r),    # [1, batch*channel_r, height_r, width_r]
        stride=1,
        groups=batch_size
    )[:, :, :height_i, width_i]  # [channel_i, batch*channel_r, height_i, width_i]
    return output.view(channel_i, batch_size, channel_r, height_i, width_i).transpose(0, 1).reshape(batch_size, channel_i*channel_r, height_i, width_i)


def sparseConv2d(response, impulse):
    """
    稀疏卷积，仅仅对应层相卷积
    input
        response: [batch, channel, height_r, width_r] 卷积核
        impulse:  [batch, channel, height_i, width_i] 冲激图
    outpur
        [batch, channel, height_i, width_i]
    """
    batch_size, channel_r, height_r, width_r = response.size()
    _, channel_i, height_i, width_i = impulse.size()
    impulse = nn.ZeroPad2d((0, width_r-1, 0, height_r-1))(impulse)
    output = nn.functional.conv_transpose2d(
        impulse.view(1, batch_size*channel_i, height_i+height_r-1, width_i+width_r-1),
        response.view(batch_size*channel_r, 1, height_r, width_r),
        stride=1,
        groups=batch_size*channel_r
    )[:, :, :height_i, :width_i]    # [1, batch*channel_i, height_i, width_i]
    return output.view(batch_size, channel_r, height_i, width_i)


class DenseConv2d_full(nn.Module):
    """
    层与层之间完全卷积，最后用一个卷积合并通道
    """
    def __init__(self, response_channels, impulse_channels, out_channels, kernel_size=(3,3), dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(impulse_channels*response_channels, out_channels, kernel_size, stride=1, padding="same", dilation=dilation, bias=bias)

    def forward(self, response, impulse):
        return self.conv(denseConv2d_1(response, impulse))


class DenseConv2d_light(nn.Module):
    """
    类似深度卷积，将DenseConv2d_full拆分成参数、计算更少的形式
    """
    def __init__(self, response_channels, impulse_channels, out_channels, impulse_kernel=(3,3), response_kernel=(3,3), dilation=1, bias=True):
        super().__init__()
        self.impulseConv = nn.Conv2d(impulse_channels, out_channels, impulse_kernel, stride=1, padding="same", dilation=dilation, bias=bias)
        self.responseConv = nn.Conv2d(response_channels, out_channels, response_kernel, stride=1, padding="same", dilation=dilation, bias=bias)

    def forward(self, response, impulse):
        im = self.impulseConv(impulse)
        rs = self.responseConv(response)
        return sparseConv2d(rs, im)
    
"""全连接相关层"""
def denseCorr2d_1(template, tomatch):
    """
    密集卷积，两个输入的层与层之间相互相关
    input
        template: [batch, channel_r, height_r, width_r] 匹配模板
        tomatch:  [batch, channel_i, height_i, width_i] 待匹配
    output
        [batch, channel_i*channel_r, height_i, width_i]
    """
    batch_size, channel_t, height_t, width_t = template.size()
    _, channel_m, height_m, width_m = tomatch.size()
    tomatch = nn.ReplicationPad2d((0, width_t-1, 0, height_t-1))(tomatch)
    batches = [
        nn.functional.conv2d(           # [channel_m, channel_t, height_m, width_m]
            tomatch[b].unsqueeze(1),    # [channel_m, 1, height_m+height_t-1, width_m+width_t-1]
            template[b].unsqueeze(1),   # [channel_t, 1, height_t, width_t]
            stride=1
        ).view(
            1, channel_t*channel_m, height_m, width_m
        ) for b in range(batch_size)
    ]   # 对每个batch单独计算
    return torch.cat(batches, dim=0)    # [batch, channel_i*channel_r, height_i, width_i]

def sparseCorr2d(template, tomatch):
    """
    稀疏卷积，仅仅对应层相卷积
    input
        template: [batch, channel, height_r, width_r] 卷积核
        tomatch:  [batch, channel, height_i, width_i] 冲激图
    outpur
        [batch, channel, height_i, width_i]
    """
    batch_size, channel_t, height_t, width_t = template.size()
    _, channel_m, height_m, width_m = tomatch.size()
    tomatch = nn.ReplicationPad2d((0, width_t-1, 0, height_t-1))(tomatch)
    output = nn.functional.conv2d(
        tomatch.view(1, batch_size*channel_m, height_m+height_t-1, width_m+width_t-1),
        template.view(batch_size*channel_t, 1, height_t, width_t),
        stride=1,
        groups=batch_size*channel_t
    )   # [1, batch*channel_t, height_m, width_m]
    return output.view(batch_size, channel_t, height_m, width_m)


class DenseCorr2d_full(nn.Module):
    """
    层与层之间完全相关，最后用一个卷积合并通道
    """
    def __init__(self, template_channels, tomatch_channels, out_channels, kernel_size=(3,3), dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(tomatch_channels*template_channels, out_channels, kernel_size, stride=1, padding="same", dilation=dilation, bias=bias)

    def forward(self, template, tomatch):
        return self.conv(denseCorr2d_1(template, tomatch))

class DenseCorr2d_light(nn.Module):
    """
    类似深度卷积，将DenseCorr2d_full拆分成参数、计算更少的形式
    """
    def __init__(self, template_channels, tomatch_channels, out_channels, template_kernel=(3,3), tomatch_kernel=(3,3), dilation=1, bias=True):
        super().__init__()
        self.templateConv = nn.Conv2d(template_channels, out_channels, template_kernel, stride=1, padding="same", dilation=dilation, bias=bias)
        self.tomatchConv = nn.Conv2d(tomatch_channels, out_channels, tomatch_kernel, stride=1, padding="same", dilation=dilation, bias=bias)

    def forward(self, template, tomatch):
        tm = self.templateConv(template)
        to = self.tomatchConv(tomatch)
        return sparseCorr2d(tm, to)