# 合成数据集
本文件夹用于存放数据集，自建，因为没有非常切合本项目任务的数据集。适合本任务的数据集有两个要点：音色和音符。

首先关注了音色。为了容易出效果，后面肯定要用音色差别大的乐器的音频的混合。首先尝试了用[CQT](../model/CQT.py)度量音色，得到了频谱含义下音色之间的距离和聚类。详见[音色度量](#timbre)。因为要和midi打交道，本文的最后附上了[midi协议下的音色、音高编码表](#midi)。

然后开始关注音符。为了以后的数据可以取之不尽，决定随机生成midi，再合成为wav。生成、合成部分见[array→midi](#midiarray)、[midi→wav](#synth)；随机算法见[随机MIDI](#rand)。

在上述基础上可以构建数据集。[数据尺寸](#size)明确了数据集元素的时间属性。在此规范下，形成了一个9(9种音色)\*9(每个音色有9个曲子)的小数据集[small](data/README.md)。至此，本文件夹的初期目标实现。

文件夹结构
```
├─fluidsynth 合成器相关
├─septimbre 【重点】随机合成数据集
├─inferMusic 一些简单的乐曲，用于主观评估
|
├─timbreMetric 【无用】CQT可视化
├─vae 【无用】尝试用VAE生成随机数据
├─sourceMIDI 【无用】VAE的训练集 收集了各种midi
|
| 以下是公共数据集，已ignore。详情见“~/evaluate/”文件夹下的预处理脚本
├─Batch10_v1.1-main
├─musicnet
├─PHENICX-Anechoic
└─URMP
```

## 音频合成工具: midi → wav{#synth}
见[fluidsynth](fluidsynth/README.md)。

输出的wav往往比midi要长，所以要进行剪裁。相关代码在[wavetool.py](../utils/wavtool.py)中。

## midi数据处理: midi ↔ nparray{#midiarray}
采用numpy数组作为可处理格式，含义是钢琴卷帘、时频形式。数据的ground truth会是nparray而不是基于事件的midi，midi作为和其他格式的中转，比如和mir_eval库交互、和人类交互，但都发生在模型训练之后。因此生成数据集的顺序是：生成nparray→生成midi→生成wav→预计算CQT。

为了实现midi和nparray的互转，需要明确每一帧的时长，具体取值见[下一章](#size)。由于时间精度不同（midi是tick，nparray是frame，前者一般短，后者一般长），因此存在转换误差。为了防止模型只学会某一分帧下的数据处理，在生成midi时（nparray->midi）加入了时间随机偏移，偏移距离小于半帧时长，保证midi转nparray时和之前结果一样。

相关函数在[midiarray.py](../utils/midiarray.py)中。支持多音轨。但是有如下局限：
- midi -> nparray: 只关注了节拍和音符事件
- nparray -> midi: bpm固定为120，tick固定为480，控制器只有乐器种类

## 数据尺寸{#size}
时长的考量：
由于CQT性质，8个八度导致时间间隔必须是128的整数倍，在22050点采样下，则最小时间单位为 $\frac{1000}{22050/128}=5.8ms$，太小了。考虑bpm120的音乐，32分音符是62.5ms，64分音符是31.25ms，要保证64分音符至少有两个采样点，则只能用256点或者384点的间隔了。

stride | ms | n/s
------ | -- | ---
128 | 5.8 | 172.3
256 | 11.6 | 86.13
384 | 17.4 | 57.42
512 | 23.2 | 43.1

我觉得384点挺好。一个样本12秒约689帧，不如就取660（是2、3、5的公倍数）？此时约为11.5s。那就取11477~11493ms作为训练集的大小。生成数据时，间隔保持为2056/147=17.414965

后记：然而后面训练basicamt模型时发现用256点的更准确，所以从384换成了256。时长上从660变为900，约10.45s。

VAE和最终的模型是按照变长设计的，但是要batch训练，要求数据一样长。torch提供了一些函数，用于不同长度的RNN训练，可参考[文章](https://blog.csdn.net/anshiquanshu/article/details/112868740)。懒得弄，索性都取660帧。

## 音色度量{#timbre}
midi有128种音色，但是只有112种是常用的。这112种音色有很多我都分辨不出来。为了更好的训练效果，组合时音色应该尽可能区分度大一些。实验时控制变量：音高为C4不变，仅仅改变乐器种类。过程见[timbreMetric](timbreMetric/TimbreMetric.ipynb)。

该notebook还实验了同一个乐器不同音符的频谱。结论是：哪怕是合成的，频谱结构依旧和音高相关。

如何度量音色？
http://www.360doc.com/content/24/0307/10/2616079_1116394138.shtml

## 随机MIDI{#rand}

### 偷懒
先去GITHUB上找找：
1. https://github.com/scraggo/Random-Music-Generators 生成的片段太短了，而且每个音符的长度固定
2. https://github.com/Jobsecond/random-midi 卧槽生成的不错啊？仔细一看节奏型是固定的
3. https://github.com/Ruslan-Pantaev/midi_music_generator C++写的，不会编译
4. https://github.com/adlaneKadri/Music-Generation-RNNs 用RNN生成，训练时输入是前100个音符的音高（对，没有时长！），输出是下一个音符（预测），不太行
5. https://github.com/RhythrosaLabs/MIDImaker 代码报错
6. https://github.com/xanderlewis/random-midi-generator 太傻卵了，生成结果太稀疏
7. https://github.com/smitec/random-midi 太简单了，根本不像音乐。源代码就是随机生成音高，时长是固定的

只能说都不太行。我的预期：
1. Polyphonic，可以有多个音符同时演奏
2. 跳跃不是很大，音符整体需要有一定的线条
3. 有一条主线，主线可以是单音的，时不时可以有多音一起
4. 多音的起始不必对齐
控制相邻音符的距离，我想用马尔可夫链实现，就是给跳转的距离的概率加权。

以下都是在[midi ↔ nparray](#midiarray)的基础上生成midi。

### VAE
也是偷懒，想用模型实现。简单了解了几个生成式模型后，选择了VAE。结果很差，最后一个稍微好点。参看文件夹[vae](vae/README.md)。

### 人工设计
且不管VAR生成结果如何，作为数据集还是不够合格，因为无法保证每一个音符都出现，不够稳定。在这个要求下，基于概率得到的数据集都不行，除非数据量很大。
所以我写了个生成方法，见[septimbre文件夹](septimbre/README.md)，基本实现了我的要求。

## midi编码{#midi}
### 音色
下面是midi定义中的所有音色，并且根据我的人耳区分度对音色进行了区分
```
0	Acoustic Grand Piano 大钢琴（声学钢琴）
1	Bright Acoustic Piano 明亮的钢琴
2	Electric Grand Piano 电钢琴
3	Honky-tonk Piano 酒吧钢琴

4	Rhodes Piano 柔和的电钢琴
5	Chorused Piano 加合唱效果的电钢琴

6	Harpsichord 羽管键琴（拨弦古钢琴）

7	Clavichord 科拉维科特琴（击弦古钢琴）


8	Celesta 钢片琴

9	Glockenspiel 钟琴

10	Music box 八音盒

11	Vibraphone 颤音琴
12	Marimba 马林巴

13	Xylophone 木琴

14	Tubular Bells 管钟

15	Dulcimer 大扬琴



16	Hammond Organ 击杆风琴

17	Percussive Organ 打击式风琴

18	Rock Organ 摇滚风琴

19	Church Organ 教堂风琴

20	Reed Organ 簧管风琴
22	Harmonica 口琴

21	Accordian 手风琴
23	Tango Accordian 探戈手风琴



24	Acoustic Guitar (nylon) 尼龙弦吉他

25	Acoustic Guitar (steel) 钢弦吉他

26	Electric Guitar (jazz) 爵士电吉他
27	Electric Guitar (clean) 清音电吉他

28	Electric Guitar (muted) 闷音电吉他

29	Overdriven Guitar 加驱动效果的电吉他
30	Distortion Guitar 加失真效果的电吉他

31	Guitar Harmonics 吉他和音
贝斯和吉他合并
32	Acoustic Bass 大贝司（声学贝司）
33	Electric Bass(finger) 电贝司（指弹）
34	Electric Bass (pick) 电贝司（拨片）
35	Fretless Bass 无品贝司
36	Slap Bass 1 掌击Bass 1
37	Slap Bass 2 掌击Bass 2
38	Synth Bass 1 电子合成Bass 1
39	Synth Bass 2 电子合成Bass 2

40	Violin 小提琴
41	Viola 中提琴
42	Cello 大提琴
43	Contrabass 低音大提琴

44	Tremolo Strings 弦乐群颤音音色

45	Pizzicato Strings 弦乐群拨弦音色

46	Orchestral Harp 竖琴

47	Timpani 定音鼓

48	String Ensemble 1 弦乐合奏音色1
49	String Ensemble 2 弦乐合奏音色2
50	Synth Strings 1 合成弦乐合奏音色1
51	Synth Strings 2 合成弦乐合奏音色2

52	Choir Aahs 人声合唱“啊”

53	Voice Oohs 人声“嘟”

54	Synth Voice 合成人声

55	Orchestra Hit 管弦乐敲击齐奏



56	Trumpet 小号

57	Trombone 长号

58	Tuba 大号

59	Muted Trumpet 加弱音器小号

60	French Horn 法国号（圆号）

61	Brass Section 铜管组（铜管乐器合奏音色）

62	Synth Brass 1 合成铜管音色1
63	Synth Brass 2 合成铜管音色2



64	Soprano Sax 高音萨克斯风

65	Alto Sax 次中音萨克斯风

66	Tenor Sax 中音萨克斯风
67	Baritone Sax 低音萨克斯风

68	Oboe 双簧管
69	English Horn 英国管

70	Bassoon 巴松（大管）

71	Clarinet 单簧管（黑管）



72	Piccolo 短笛
73	Flute 长笛
74	Recorder 竖笛
75	Pan Flute 排箫
76	Bottle Blow 吹瓶子
77	Shakuhachi 日本尺八
78	Whistle 口哨声
79	Ocarina 奥卡雷那


80	Lead 1 (square) 合成主音1（方波）
81	Lead 2 (sawtooth) 合成主音2（锯齿波）
82	Lead 3 (caliope lead) 合成主音3
83	Lead 4 (chiff lead) 合成主音4
84	Lead 5 (charang) 合成主音5
85	Lead 6 (voice) 合成主音6（人声）
86	Lead 7 (fifths) 合成主音7（平行五度）
87	Lead 8 (bass+lead)合成主音8（贝司加主音）

88	Pad 1 (new age) 合成音色1（新世纪）
89	Pad 2 (warm) 合成音色2 （温暖）
90	Pad 3 (polysynth) 合成音色3
91	Pad 4 (choir) 合成音色4 （合唱）
92	Pad 5 (bowed) 合成音色5
93	Pad 6 (metallic) 合成音色6 （金属声）
94	Pad 7 (halo) 合成音色7 （光环）
95	Pad 8 (sweep) 合成音色8

96	FX 1 (rain) 合成效果 1 雨声
97	FX 2 (soundtrack) 合成效果 2 音轨
98	FX 3 (crystal) 合成效果 3 水晶
99	FX 4 (atmosphere) 合成效果 4 大气
100	FX 5 (brightness) 合成效果 5 明亮
101	FX 6 (goblins) 合成效果 6 鬼怪
102	FX 7 (echoes) 合成效果 7 回声
103	FX 8 (sci-fi) 合成效果 8 科幻


104	Sitar 西塔尔（印度）
105	Banjo 班卓琴（美洲）
106	Shamisen 三昧线（日本）
107	Koto 十三弦筝（日本）
108	Kalimba 卡林巴
109	Bagpipe 风笛
110	Fiddle 民族提琴
111	Shanai 山奈
```

### 音高与编码
音名|频率|编号
---|-----|---
C1 | 32.70 | 24
C#1 | 34.65 | 25
D1 | 36.71 | 26
D#1 | 38.89 | 27
E1 | 41.20 | 28
F1 | 43.65 | 29
F#1 | 46.25 | 30
G1 | 49.00 | 31
G#1 | 51.91 | 32
A1 | 55.00 | 33
A#1 | 58.27 | 34
B1 | 61.74 | 35
C2 | 65.41 | 36
C#2 | 69.30 | 37
D2 | 73.42 | 38
D#2 | 77.78 | 39
E2 | 82.41 | 40
F2 | 87.31 | 41
F#2 | 92.50 | 42
G2 | 98.00 | 43
G#2 | 103.83 | 44
A2 | 110.00 | 45
A#2 | 116.54 | 46
B2 | 123.47 | 47
C3 | 130.81 | 48
C#3 | 138.59 | 49
D3 | 146.83 | 50
D#3 | 155.56 | 51
E3 | 164.81 | 52
F3 | 174.61 | 53
F#3 | 185.00 | 54
G3 | 196.00 | 55
G#3 | 207.65 | 56
A3 | 220.00 | 57
A#3 | 233.08 | 58
B3 | 246.94 | 59
C4 | 261.63 | 60
C#4 | 277.18 | 61
D4 | 293.66 | 62
D#4 | 311.13 | 63
E4 | 329.63 | 64
F4 | 349.23 | 65
F#4 | 369.99 | 66
G4 | 392.00 | 67
G#4 | 415.30 | 68
A4 | 440.00 | 69
A#4 | 466.16 | 70
B4 | 493.88 | 71
C5 | 523.25 | 72
C#5 | 554.37 | 73
D5 | 587.33 | 74
D#5 | 622.25 | 75
E5 | 659.26 | 76
F5 | 698.46 | 77
F#5 | 739.99 | 78
G5 | 783.99 | 79
G#5 | 830.61 | 80
A5 | 880.00 | 81
A#5 | 932.33 | 82
B5 | 987.77 | 83
C6 | 1046.50 | 84
C#6 | 1108.73 | 85
D6 | 1174.66 | 86
D#6 | 1244.51 | 87
E6 | 1318.51 | 88
F6 | 1396.91 | 89
F#6 | 1479.98 | 90
G6 | 1567.98 | 91
G#6 | 1661.22 | 92
A6 | 1760.00 | 93
A#6 | 1864.66 | 94
B6 | 1975.53 | 95
C7 | 2093.00 | 96
C#7 | 2217.46 | 97
D7 | 2349.32 | 98
D#7 | 2489.02 | 99
E7 | 2637.02 | 100
F7 | 2793.83 | 101
F#7 | 2959.96 | 102
G7 | 3135.96 | 103
G#7 | 3322.44 | 104
A7 | 3520.00 | 105
A#7 | 3729.31 | 106
B7 | 3951.07 | 107
