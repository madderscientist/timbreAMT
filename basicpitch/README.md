# "音色无关转录"Baseline——BasicPitch
为了评估模型，需要横向对比。自然是选择同一个赛道的BasicPitch了：https://github.com/spotify/basic-pitch

## 模型搭建（异同）
平台是windows，但最新的win-tensorflow无法使用gpu，2.10的可以，但项目要求使用python>=3.11，而tensorflow2.10只能在更低的python版本中用，因此无法跑源代码，只能在torch上复现。

结构上，我没有`pitch estimation`只有`note`和`onset`，而BasicPitch论文里说去掉`pitch estimation`效果基本不变，于是我就遵循了源码上无`pitch estimation`的结构（因为我没有pitch的监督数据）。如果有了`pitch estimation`，会先降维到1再拓展到32，损失了大量信息，我认为这会显著降低后续的性能，所以删去这个降维的步骤是非常合理的。不过去了一层导致模型参数变多了。

损失函数照搬源码。我用的是focalloss且负样本权值大；basicpitch的note用BCE，onset用正样本权值大的weighted-BCE。

为了比较公平我用了我的CQT。CQT后的处理照搬源码，即先取log再平移缩放到0~1，最后batchnorm。原论文CQT用的是hann窗，我的是blackmanharris；发现源码也是用卷积层实现抗混叠，但我的抗混叠是零相位的，而且滤了两次衰减肯定更好。综合来说应该都优于源码。

数据方面直接使用了basicamt第二阶段的训练数据，由于要用benchmark加速，所以batchsize沿用了我的18（论文中说用16，差不多）。为了加速训练给这个数据集补上了CQT，见[calc_cqt.ipynb](calc_cqt.ipynb)。

## 损失函数直观比较
BasicPitch使用的onset损失的加权策略为：稀疏方用大权值。我认为应该反过来。下面是不同权值的结果（每张图的上面为frame结果，下面为onset结果）：

1. 使用BasicPitch的加权方案：
![](./best_epoch_result.png)
![](./origin%20basicpitch's%20loss.png)

2. 使用我的加权方案：![](./basicpitch%20using%20focal%20loss.png)

可以看到前者的onset完全不能看。审稿人说：后处理也许会弥补onset的不足。实验表明，更精确的onset能取得更好的后处理效果。