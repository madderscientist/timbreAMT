# 可能会用到的模块的介绍

# CQT
输入22050采样率，可以到E9，所以八度只能从C1到B8，B8是7900Hz，最高表示频率为11025，约为其0.72倍，降采样后需要保证频率fs/4*0.72以前幅度不被衰减
每个八度36bin，一共8个八度，每个参数是float32

算法    |  参数量   | 用时(seconds)
------- | -------- | --
CQT定义 | 20365632 | 4.344
CQT精简 | 3694192  | 9.4
CQT迭代 | 19955    | 0.365

要想让CQT可训练，上面一个都用不了。
CQT_small是迭代计算的，但是用同一个卷积核完成所有八度的运算容易梯度爆炸（变成nan）。即使把filtfilt降采样换成直接降采样，也只能在4个八度内梯度可计算（变成5则有4行梯度爆炸）。
CQT_small和前面两种基本没有误差。
草梯度爆炸是因为开了根号没加小量。是可以用的。

# denseConv2d
信号模型，两个输入层和层之间卷积。目前的想法是提取音色，假设音色仅仅和乐器有关（很不靠谱）
## 按原理的实现(巨慢，不要用)
```py
def forward(self, response, impulse):
    batches = []
    impulse = nn.ZeroPad2d((0, response.size(3)-1, 0, response.size(2)-1))(impulse)
    for b in range(response.size(0)):   # 对每个batch单独计算
        _output = []
        _impulse = impulse[b].unsqueeze(0)
        _response = response[b].unsqueeze(1) # [channel_r, 1, height, width]
        for ch in range(impulse.size(1)):
            _output.append(
                nn.functional.conv2d(
                    _impulse[:, ch:ch+1],
                    _response,
                    stride=1
                )
            )   # [1, channel_r, height_i, width_i]
        batches.append(torch.cat(_output, dim=1))   # [1, channel_i*channel_r, height_i, width_i]
    return torch.cat(batches, dim=0)    # [batch, channel_i*channel_r, height_i, width_i]
```
copilot说这样慢，告诉我把卷积展开：
```py
def forward(self, response, impulse):
    batch_size, channel_r, height_r, width_r = response.size()
    _, channel_i, height_i, width_i = impulse.size()

    # [batch, channel_i, height_i + height_r - 1, width_i + width_r]
    impulse_padded = nn.functional.pad(impulse, (0, width_r - 1, 0, height_r - 1))

    # 使用 unfold 将 impulse 展开为滑动窗口
    # [batch, channel_i * height_r * width_r, height_i * width_i]
    impulse_unfolded = nn.functional.unfold(impulse_padded, (height_r, width_r))
    # [batch, channel_i, height_r * width_r, height_i * width_i]
    impulse_unfolded = impulse_unfolded.view(batch_size, channel_i, height_r * width_r, height_i * width_i)

    # 将 response 展开为滑动窗口
    # [batch, channel_r, height_r * width_r]
    response_unfolded = response.view(batch_size, channel_r, height_r * width_r)

    # 计算卷积结果
    # [batch, channel_i, channel_r, height_i * width_i]
    impulse_unfolded = impulse_unfolded.transpose(2,3).reshape(batch_size, channel_i * height_i * width_i, height_r * width_r)
    response_unfolded = response_unfolded.transpose(1,2)
    output = torch.matmul(impulse_unfolded, response_unfolded)
    return output.view(batch_size, channel_i, height_i * width_i, channel_r).transpose(2,3).reshape(batch_size, channel_i * channel_r, height_i, width_i)
    # 第二种，稍微慢一些
    output = torch.einsum('brk,bikn->birn', response_unfolded, impulse_unfolded)
    return output.reshape(batch_size, channel_i * channel_r, height_i, width_i)
```
测试：
```py
if __name__ == '__main__':
    # 测试
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    dc = DenseConv2d().to(device)
    import time
    impulse = torch.randn(64, 4, 84, 300).to(device)
    response = torch.randn(64, 5, 37, 8).to(device)
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    with torch.no_grad():
        start_time = time.time()
        result_origin = dc.forward_origin(response, impulse)
        end_time = time.time()
        print(f"Original forward time: {end_time - start_time} seconds")

        start_time = time.time()
        result = dc.forward_opt1(response, impulse)
        end_time = time.time()
        print(f"matmul forward time: {end_time - start_time} seconds")
        print(result.shape)
        print((result - result_origin).abs().max())

        start_time = time.time()
        result = dc.forward_opt2(response, impulse)
        end_time = time.time()
        print(f"einsum forward time: {end_time - start_time} seconds")
        print(result.shape)
        print((result - result_origin).abs().max())
```
einsum确实比matmul慢一些，但不明显。可是卷积展开比循环慢很多很多（十倍）！无论是GPU还是CPU。而且卷积展开占用大量内存，batch_size到128就把12G的显卡吃爆了，到256连32G的内存都不够用。所以还是用循环。
## 改进
利用batch的并行，去掉了一层循环
```py
"""全连接卷积层"""
def denseConv2d(response, impulse):
    """
    input
        response: [batch, channel_r, height_r, width_r] 卷积核
        impulse:  [batch, channel_i, height_i, width_i] 冲激图
    output
        [batch, channel_i*channel_r, height_i, width_i]
    """
    batch_size, channel_r, height_r, width_r = response.size()
    _, channel_i, height_i, width_i = impulse.size()
    impulse = nn.ZeroPad2d((0, width_r-1, 0, height_r-1))(impulse)
    batches = []
    for b in range(batch_size):   # 对每个batch单独计算
        _impulse = impulse[b].unsqueeze(1)  # [channel_i, 1, height_i+height_r-1, width_i+width_r-1]
        _response = response[b].unsqueeze(1) # [channel_r, 1, height, width]
        batches.append(
            nn.functional.conv2d(    # [channel_i, channel_r, height_i, width_i]
                _impulse, _response, stride=1
            ).view(
                1, channel_r*channel_i, height_i, width_i
            )
        )
    return torch.cat(batches, dim=0)    # [batch, channel_i*channel_r, height_i, width_i]
```
再改进：有一个维度是1，没用上。试着用group将实际不同的batch一起算了：
```py
def denseConv2d(response, impulse):
    batch_size, channel_r, height_r, width_r = response.size()
    _, channel_i, height_i, width_i = impulse.size()
    impulse = nn.ZeroPad2d((0, width_r-1, 0, height_r-1))(impulse)
    output = nn.functional.conv2d(
        impulse.transpose(0, 1),    # [channel_i, batch, height_i+height_r-1, width_i+width_r-1]
        response.view(batch_size * channel_r, 1, height_r, width_r),    # [batch*channel_r, 1, height_r, width_r]
        stride=1,
        groups=batch_size
    )  # [channel_i, batch*channel_r, height_i, width_i]
    return output.view(channel_i, batch_size, channel_r, height_i, width_i).transpose(0, 1).reshape(batch_size, channel_i*channel_r, height_i, width_i)
```
后来发现torch的conv其实是信号的相关，真正的卷积是conv_transpose，卷积后的长度也是(长度和-1)，根据含义把后面的裁了就行。于是得到了下面的代码：
```py
def denseConv2d_1(response, impulse):
    batch_size, channel_r, height_r, width_r = response.size()
    _, channel_i, height_i, width_i = impulse.size()
    batches = [
        nn.functional.conv_transpose2d(           # [channel_i, channel_r, height_i, width_i]
            impulse[b].unsqueeze(1),    # [channel_i, 1, height_i+height_r-1, width_i+width_r-1]
            response[b].unsqueeze(0),   # [1, channel_r, height, width] 注意转置卷积的weight顺序和卷积不一样
            stride=1
        )[:, :, :height_i, :width_i].view(
            1, channel_r*channel_i, height_i, width_i
        ) for b in range(batch_size)
    ]   # 对每个batch单独计算
    return torch.cat(batches, dim=0)    # [batch, channel_i*channel_r, height_i, width_i]
```

### 性能测试
impulse = torch.randn(64, 4, 84, 600, requires_grad=True).to(device)
response = torch.randn(64, 5, 37, 16, requires_grad=True).to(device)

CPU E5 2666 V3：
model | forward | backward(second)
----- | ------- | --------
两重循环 | 0.6678380966186523 | 29.67567729949951 |
一重循环 | 0.5719051361083984 | 28.615421533584595 |
零重循环 | 31.564237117767334 | 26.854256868362427 |

GPU RTX 4070 SUPER
model | forward | backward(second)
----- | ------- | --------
两重循环 | 0.5074326992 | 0.3010115623474121 |
一重循环 | 0.2329559326 | 0.01100039482116992 |
零重循环 | 1.34751439 | 0.001001358 |

CPU R5 7500F
model | forward | backward(second)
----- | ------- | --------
两重循环 | 0.435 | 8.8629 |
一重循环 | 0.403 | 8.3934 |
零重循环 | 1.086 | 13.509 |

GPU下运行格外快，总用时最短的是一重循环
## 数学分析
假设有输入冲激 $ X = [x_i], i = 1,2,...,n $ 和响应 $ H = [h_i], i = 1,2,...,m $ 每个小写字母是一个2d tensor，有两种作用方式（conv2d用1*1卷积表示，即层加权求和）：
1. 用denseConv2d：
$$
\begin{aligned}
    Z &= denseConv2d(H, X) \\
    z_{ij} &= x_i \ast h_j \\ \\
    Y &= conv2d(A, Z) \\
    y_{i} &= \sum_{p=1}^{nm}a_{ijk}(x_j \ast h_k) \\
        &= \sum_{j=1}^{n}\sum_{k=1}^{m}a_{jk}(x_j \ast h_k), i=1,2,...,l
\end{aligned}
$$

2. 先对X和H分别做卷积，再合起来：
$$
\begin{aligned}
    U &= conv2d(A, X) \\
    u_i &= \sum_{j=0}^{n}a_{ij} x_j, i=1,2,...,l \\ \\
    V &= conv2d(B, H) \\
    v_i &= \sum_{j=0}^{m}b_{ij} h_j, i=1,2,...,l \\ \\
    Y &= conv2d(H, X) \\
    y_i &= u_i \ast v_i \\
        &= \sum_{j=1}^{n}a_{ij} x_j \ast \sum_{k=1}^{m}b_{ik} h_k \\
        &= \sum_{j=1}^{n}\sum_{k=1}^{m}a_{ij}b_{ik}(x_j \ast h_k) \\
        &= \sum_{j=1}^{n}\sum_{k=1}^{m}c_{ijk}(x_j \ast h_k), i=1,2,...l
\end{aligned}
$$

其实两者是等价的。前者的参数量是nml，后者参数量是(n+m)l。所以前者可以大力出奇迹，如果前者成功了就换后者。

对应层做卷积的实现：
```py
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
        impulse.view(1, batch_size*channel_i, height_i+width_r-1, width_i+height_r-1),
        response.view(batch_size*channel_r, 1, height_r, width_r),
        stride=1,
        groups=batch_size*channel_r
    )[:, :, :height_i, :width_i]    # [1, batch*channel_i, height_i, width_i]
    return output.view(batch_size, channel_r, height_i, width_i)
```