# 使用VAE生成数据集
目的是生成midi数据集，达到“无限数据”的效果。借此机会尝试一下生成网络。一开始觉得手写生成方法很难模拟现实情况，不如让模型学习。GAN需要一个判别网络，不适合本任务，于是选择了VAE。基本原理是编码、加噪、解码，训练目标是复原输入，不需要标签。加噪时假设编码空间每个元素为高斯分布，加噪使得一定大小的编码空间都映射到同一个输入。

一个midi音符有三个要点：音高、起点和长度。

## 训练集
输入和输出都用nparray化后的midi数据，即钢琴卷帘。需要学习的midi见[sourceMIDI文件夹](../sourceMIDI/README.md)，我收集了不少。当时收集的时候一股脑全拿来用了，后面才发现有很多高难度的钢琴曲（钢琴曲训练集很多），导致整个谱面都是音符。

制作数据集时，首先用[midiarray.py](../utils/midiarray.py)的midi2array把每个midi转为nparray，每帧17.4ms；再将nparray分段， 如[上一层文件夹的README](../README.md)所言，每段660帧。对于被分段截断的音符，仅仅保留前半部分，丢弃后半部分。形象而言，就是把所有midi时间上排成一排，转为nparray，然后每660帧切一刀，切断音符的留头不留尾。具体实现参看[make.ipynb](make.ipynb)。由于钢琴复杂的曲子和长笛简单的曲子混在一起，很有可能难的段落太难，简单的太简单，所以进行如上操作前，我把所有midi打乱了。产生了22106段，约70.6小时。但是不知道是否有重复的曲子。

[观察数据集](viewData.ipynb)：可以看到音符非常多，风格各异。超复杂的曲子数目特别多，事实上对我的训练有不良影响。、

## 网络结构迭代历史
我写了有6版，由于是二位数据，因此加入了卷积；由于还是时序数据，因此加入了LSTM。第一版是精心设计的，用了dense防止梯度问题，层数较少，在4070Super上训练很快，一晚上900轮，开始出现反弹。能看出一条一条，但并不是很清晰。
我觉得是力气不够大。于是第二版用了比较深的全卷积网络，训练死慢，效果超差，不如v1.
我觉得是卷积网络的问题，于是第三版在v1的基础上加深了LSTM。但是是不行。
第四版对v1做减法，结果没几轮损失就收敛了，不如第一版。
第五版时我才发现别人的batchnorm都在激活函数前面，而且每个conv后面都有，所以写了个最深的conv+LSTM网络。训练超级超级慢，效果也一般般。
第六版我改了损失函数、把onset和duration分开，终于效果还行。后面细说。

## 反思与改进
本节针对第六版的相对胜利而言。

之前生成的卷帘图中，能看出是一条一条的，但是每一条边界模糊，且大片连在一起。我认为这是那些复杂钢琴曲数据导致的。但是我没有去动数据集，毕竟这都是真实数据。

此外，在我的定义下，我希望每个音符的第一帧是2，其余是1。这其实要求网络进行三分类，太为难了。v1~v5我都要求输出满足这个要求，为此我一般用tanh作为最终的激活函数（因为包含-1、0、1），但是得到的结果根本看不出onset（全糊在一起），完全不能用来生成midi。写完v5后我去翻译了一篇论文（basicPitch那篇，全卷积转录网络），看到他们先得到active卷帘，然后结合原数据得到onset卷帘，两个分开来算损失。于是在v6中我在解码部分进行了类似的处理，得到的两个输出只要做二分类，用sigmoid即可。分开两者对后续的处理（生成的卷帘二值化、生成midi）带来极大便利，还提升了训练效果。之后的模型一定要参照这个做法。

最重要的是损失函数。在卷帘中，note相对silence十分稀疏，而onset相对note更为稀疏，类别不平衡。一开始我就考虑到这个问题，因此我改动了寻常VAE损失中的MSE部分：按照真实值的频率进行对应MSE的加权，频率越小权值越大。这个做法其实是要求网络三分类的无奈之举。损失在400~500间吧，下不去了。由于0的个数很多，因此其权值最小，压制最小，因此输出的音符边界模糊、断不开。
在v6中变成二分类后，可以用交叉熵；进一步考虑到类别不平衡，我使用了focal loss（很多AMT论文也是用的这个），基本思想是对假类别加以严重的处罚，本任务中就是对0格外严格。但是对假类严格表示对真类放纵，因此输出的最大值基本到不了1，二值化时不能用0.5了（后面再说），这是前30轮发现的问题。因此我还是给v6加上了之前的加权MSE：把onset和note加起来，和ground truth做加权MSE，其实在要求最大值能顶到1。

还有不起眼的改进是用AdamW代替了Adam。由于被v5的速度实在太慢，我把LSTM改成了GRU（我还觉得似乎没有什么太长期记忆的必要，少一个隐状态也不会怎么样），把SiLU改成了ReLU，还减少了卷积层的数目，比v1还轻量。

以上改动使得v6的结果前所未有的好（还很快）。每个音符看起来都和流星一样：开头明显、结尾淡去（这可能是因为我没有强调结尾，因为只对开头和中间进行了强调。所以如果要进一步改进，可以把offset也分离出来），音符之间断开了。此时终于有后处理的可行性了。但是由于每个输出的最大值到不了1，用0.5作为二值化阈值得到的基本全是0，所以我用了大津法得到阈值。由于最后是淡出的，为了让音符长一点，我使用了类似施密特触发器的动态阈值方法：从silence变为note的阈值是OTSU的阈值，从note变为silence的阈值低一些。为了让midi更实际，我约束了音符的最小长度。

虽然是迄今最好的，但是仍然有让我不满的。首先onset和note并不是完全对应。不过鉴于onset最大的用途是断开连续相同的音，其实也能用。其次输出不稳定，这是数据集导致的。大概还是成为不了我想要的数据生成器。

## 文件结构
```
│   make.ipynb      将../sourceMIDI中的midi转为nparray，并在时间上分段，结果存于gen_example文件夹中
├─gen_example       切片midi得到的numpy数组，作为训练集
│   viewData.ipynb  观察gen_example中的数据，有一部分结果保存到了example_view文件夹
├─example_view      1:1可视化gen_example中的数据
│   vae.py          最后一版、最好的一版的模型结构与损失
│   train.ipynb     训练过程
│   model_best.pth  损失最小的训练结果
│   epoch120.pth    120轮时的训练结果
│   README.md       本文件
└─VAE-history       历时版本，由于V2过大，所有历史都发布到了release上
    │   vae_v1      第一版VAE
    │   vae_v2      第二版VAE
    │   vae_v3      第三版VAE
    │   vae_v4      第四版VAE
    │   vae_v5      第五版VAE
```