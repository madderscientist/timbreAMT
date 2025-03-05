# FluidSynth Python interface
合成器使用fluidsynth，这是一个C语言的项目。于是我根据项目pyfluidsynth写了一个python的音频文件合成使用接口（pyfluidsynth没有实现到文件的合成功能）（后来我提了个[pr](https://github.com/nwhitehead/pyfluidsynth/pull/85)，现在有了），即[fluidsynth.py](fluidsynth.py)

之前用的是[midi2audio](https://github.com/bzamecnik/midi2audio)，但是这个太慢了，因为每次合成都会重新创建合成器。所以我深入了fluidsynth的main函数，找到了复用合成器的方法，并利用[pyfluidsynth](https://github.com/nwhitehead/pyfluidsynth)提供的框架实现了调用。

## 文件结构：
```
│  fluidsynth.py    Synth类
│  MS Basic.sf3     来自Musescore4的soundfont文件 最大
│  gnusmas_gm_soundfont_2.01.sf3     来自https://github.com/Gnusmas49/gnusmas-soundfonts
│  MS Basic.sf3     来自Musescore1的soundfont文件 很小
│  readme.md
│  __init__.py      模块声明
│  (deprecated)synth.py 废弃的慢速版本
│
└─fluidsynth_win    windows下的fluidsynth，没有子文件夹，包含fluidsynth.exe和各种dll
```

若要在别的平台使用，应该更换fluidsynth_win为对应操作系统的release，并且修改代码开头的路径部分

## Usage
```py
from fluidsynth import Synth
s = Synth(22050)
s.midi2audio("test.mid", "test.wav")
s.midi2audio("test2.mid", "test2.wav")
del s       # or: s = None
```
一次对象创建可以多次合成。midi2audio也是用fluidsynth，但是只是利用其命令行，而命令行每次合成都需要从头加载，效率很低。我深入fluidsynth的main函数，实现了适用于短时间大批量的合成。

## 波表合成与音色
开始质疑音色音色是否一定。这里的音色考虑谐波幅度。

实测同一把口琴不同簧片的谐波分布也不一样。

但是训练数据拟采用合成器。fluidsynth是根据soundfont协议合成的，所以去了解了soundfont协议：https://www.synthfont.com/sfspec24.pdf。不过里面主要是soundfont2的文件结构，这方面可以参考更好看的：https://mrtenz.github.io/soundfont2/

soundfont描述了波表合成器的数据和参数。波表可以视为一个二维数组，第一维度是不同波形状，第二维度是每个波形的每个时刻的采样点。用代码表达如下：
```cpp
typedef float* wave;// wave是一个波形时域采样序列
typedef wave* wavetable;// 多个wave组成wavetable。每个wave长度一样
```

在wavetable前先介绍一下*sample synth*：乐音有周期，所以只要采样一个周期，反复播放就能还原音色了。然后用ADSR这样的幅度包络调制一下就得到了动态，但每个周期波形都一样（忽略动态带来的幅度变化，因为一个周期远小于动态变化时间），缺少“生命力”。而波表就是可以动态改变每个周期的波形的sample synth。三维画出来的波表，把波形堆叠的维度看成时间轴，波表就像波形的时间切片。通过插值可以平滑地在各个波形之间切换（渐变为下一个波形）。谈到“切换”这种动态的东西，花样就多了，可以用各种切换的曲线，控制切换的顺序和速度（视频里的LFO就可以调制切换的过程）。下面这个视频的介绍简短直观：https://www.youtube.com/watch?v=MWuU7BqbqBU。

视频里有制作波表。另一个[视频](https://www.bilibili.com/video/BV1SF411f7Dj)用中文展示了如何把心电图变为波表。据我所观察，先对导入的波形切分为长度相同且紧邻的波形分段，这些波形叠起来就是波表。如果从头到尾顺序播放波表中的波形就是切片前的音频，但是通过调整波形切换的速度可以改变音色。疑问：分段后要周期性播放，但段头和段尾幅度不一样，会导致幅度跳变。这个问题确实有，但是软件可以调整至首尾一样：[【用Serum制作令人惊叹的Vocal Synth (波表教程)】 【精准空降到 02:00】](https://www.bilibili.com/video/BV1ur421A7SD/?share_source=copy_web&vd_source=5ba343b2efa35366332287bb3d88dc7b&t=120)。神奇的是，即使每个波形重复不止一次再切换到下一个波形（比之前慢），竟然也能听出原本说话的内容，让我想到了ISTFT：每一帧单独IFFT，合起来虽然和原波形不一样，但是听起来还差不多。

因此wavetable的动态性非常强，但是给我的任务带来了挑战。如果是sample，波形不会变，那音色不会变；但wavetable波形在变啊，说明音色也会变。想象一下440Hz的正弦波用1秒平滑变为440Hz的方波，音色一直在变，就很难用“冲激响应”这一套来建模了。但我想乐器还是和电音不一样的，因为乐器合成的目的是模拟现实的乐器。我猜测大概流程是这样的：录制了一个笛子音（从吹响前到无声），切片成波表，波表前面的波形就是刚吹响时的响应，即ADSR的AD，中间是S，结尾是R，播放的时候先播放一遍AD，然后波形在R那里（如果要动态可以在R部分的波形中间来回切换）直到发送“note off”指令，这时播放R。

实测B1以下、C7以上不同乐器音色趋同，和中音部分的谐波完全不一样（虽然有的乐器根本达不到这样的音高）。