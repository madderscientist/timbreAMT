# 数据集生成
本项目的一大特点是完全使用随机生成的数据集。“生成”可行，是因为Slakh2100数据集也是合成的，好处是对齐精度高；而之所以要“随机”，一是这样数据成本更低（Slakh数据集实在太大了）而且无穷无尽；二是数据分布稳定：“音乐”是非常人工的产物，音符的分布极其不稳定（比如缺乏极高音和极低音），需要大量的数据才能有统计性和全覆盖。

为了让数据分布稳定，比如每个乐器中每个音符至少出现一次。在这个要求下，基于概率的数据生成方法就不好用了。因此，我决定先生成所有需要的音符，然后打乱，再排布到卷帘上。相关代码在[class Notes](gen.py)中。

一个音符有三个要素：音高、起点、时长。除了上文的音高选取基本方法，还考虑了不同音高的出现频率。我给每个八度加了个权值，生成“所有需要的音符”时，每个音设置“其权重”个，然后打乱。具体参看Notes.new函数。

为了尽量体现“主旋律”，加入了“nearest”机制：选取未来一段长度内和上一个音符最近的音符作为下一个（和本来的下一个互换位置）。

为了尽量有“polyphonic”，即同一时刻有多个音符，我设计了两个方法：
1. 下一个音符的开头和本音符的结尾之间的时间间隔可以是负数（下文提到的起点设置方法）
2. 排布每个音符时，有概率触发“多音”，即同一时刻有多个音符按下

起点的设置采用了类似马尔可夫的方法：以上一个音符的结尾为参照，随机一个时间偏移，作为本音符的起点。这个偏移可正可负，但为了保证时间推进，限制下一个音符最左（负）的偏移只能是上一个音符的长度（即本音符和上一个音符的起点对齐）。采用了截断的正态分布，偏移均值为0，下界为(-上映音符.时长)，上界理论上是正无穷，实际用了音符的最长时长。后来发现这样利用率太低了，于是改成均值为(-最短音长)、最大为一半的平均时长，提高了音符密度。

时长的设置没什么机制，以4分音符为均值，以32分音符为最小值，以略长于一分音符的时长为上界，用截断的正态分布采样。

关于每个音符的音高、起点、时长的设置，参看Notes.fetch；关于排布多音符，参看Notes.generate，需要解决时间重叠、实现“多音”。

## 和声与频谱重叠
如果同时演奏的音在频域没有重叠，训练出的模型可能会欠缺分离的能力。本节的目的是设计一些频谱重叠的音符排列。
泛音列在十二平均律下的位置（$log_{2^{\frac{1}{12}}}(n), n=1,2,3...$）:
- +0: 基频
- +12：二次谐波
- +19：三次谐波
- +24：四次谐波
- +28：五次谐波
- +31：六次谐波

所以频谱能形成重叠的主要有：
- +0的二次 & +12的一次（八度）
- +0的三次 & +7的二次（五度）
- +0的四次 & +5的三次（三度）
- +0的五次 & +4的四次（小三度）

原来和弦就是频谱重叠。
具体实现时，如果有多音，则有85%的概率是和弦，其余设置为随机音符。和弦往往是节奏或者背景，为了孤立长音和弦的出现，增加了“越长越容易触发多音”的机制。


## 数据增强
中途加入的功能。之前的随机仅仅是音符的音高和时长，唯二的增强是numpyarray->midi时帧内时间偏移和训练时的加性高斯噪声，乐曲和实际相比缺乏音量变化、演奏技巧。故在[midiarray.py](../../utils/midiarray.py)中实现了弯音、震音、音量起伏的随机化，见函数midi_randomize。


## 数据集组成
运行本文件下的“make{可能的后缀}.ipynb”即可创建数据集，按照乐器种归类到不同文件夹。每个文件夹下有相同数目的样本，每个样本有如下组成成分：
- `i.cqt.npy`: [2, bins, time]的CQT数据。可选是否生成。
- `i.npy`: [notes, time]的midi卷帘
- `i.wav`: 22050采样率的、长度和CQT一样的音频数据。用torchaudio读取后为[channel=1, sample]
- `i.mid`: midi文件。生成自i.npy，训练时用不到，制作数据集时作为i.npy到i.wav的桥梁

具体结构见本文档最后。

### 音色无关转录
“音色无关”要求模型“见多识广”，所以音色种类要多。人工挑选出以下常见或有特色且无歧义（歧义特指midi音色中的吉他泛音）的音色：
```
0, 1, 2, 6, 7,
8, 10, 14,
19, 21,
24, 25, 30,
42, 44, 46, 48,
53, 54,
56, 60, 61,
66, 68, 71,
72, 77,
81, 88, 98
```
钢琴和弦乐多一些，因为现实中更常见。

### 音色分离转录
对于“音色分离转录”任务，训练时各个音色应该区别较大。目标是9种乐器，每种乐器选了一个代表。测试的时候发现音高对乐器的音色影响非常大。因此我下调了最边缘两个音高数目权重。每种乐器有9个midi，若两两组合一共有C_9^2\*9\*9=2916种组合，约9.3小时。一个midi大概30个音符，因此加权和为22.5。由于发现音色在极高音和极低音时趋同，因此设置权重为[1, 3, 5, 6, 5, 3, 1]，和为24。

以C4音时音色的聚类结果如下：
- Cluster 0: 13 items: 15,19教堂风琴,26,27,29,30,56小号,61,65,104,106,109,111
- Cluster 1: 28 items: 0大钢琴,1,2,3,4,5,6,10,16,18,22,24,32,36,38,39,42,43,59,66,67,73,74,81,83,86,87,90
- Cluster 2: 11 items: 7,17,28,37,47,52,55,68双簧管,84,105,107
- Cluster 3: 18 items: 20,21,23,25,33,34,35,40小提琴,41,44,48,49,57,60,64,69,70,110
- Cluster 4: 7 items: 88,89,93合成音色6 （金属声）,97,98合成效果水晶,101,103
- Cluster 5: 5 items: 9钟琴,14,31,92,95
- Cluster 6: 10 items: 8,11,12,13,46竖琴,71单簧管,79,80,82,108
- Cluster 7: 11 items: 45,53,58,72短笛,75排箫,76,77,78,85,91,96
- Cluster 8: 9 items: 50,51,54合成人声,62,63,94,99,100,102

但是感觉分类结果不太行。于是我人工分类：
- 0大钢琴 24尼龙弦吉他
- 72短笛 73长笛
- 71单簧管 66中音萨克斯风
- 68双簧管 69英国管
- 40小提琴 42大提琴
- 19教堂风琴 21手风琴
- 52人声合唱“啊” 53人声“嘟”
- 56小号 57长号
- 98合成效果水晶 8 钢片琴

## 不同数据集的配置
在hop为256点时，长数据用900点（约10.45s），短数据用360点（约4.18s，命名为short）。每种乐器的样本数将用以下命名规则：

| 量级 | 10 | 30 | 300 | 1000 |
| --- | --- | --- | --- | --- |
| 命名 | small | medium | large | huge |

对于音色无关转录（本项目中特指模型basicamt）来说，使用的配置见[basicamt:README](../../basicamt/README.md);

而对于“音色分离转录”，拟使用以下配置：

```
instrument = [56, 0, 68, 40, 98, 9, 71, 72, 54]
midi_num = ?
octave_weight = [1, 3, 5, 6, 5, 3, 1]
frames = 900 or 360
```


## 文件结构
```
│  data.py          给torch用的DataSet
│  gen.py           随机生成midi数据的类
│  gen_test.ipynb   测试随机算法，兼可视化随机结果
│  make_basicamt.ipynb       制作basicamt的训练数据集，输出到data文件夹中
│  readme.md        本文件
│  view.png         以下三个都是用make.ipynb观察生成数据集时得到的可视化结果
│  view2.png
│  view3.png
│  
└─small             数据集
    ├─inst0         第一个类别是音色。仅仅显示第一个音色文件夹
    │      0.cqt.npy    每个音频有四个文件。仅仅显示第一个音频的数据。本文件是cqt计算结果
    │      0.mid        midi原数据
    │      0.npy        midi对应的nparray
    │      0.wav        由midi合成的音频
```