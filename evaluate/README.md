# 模型评估
使用了三个多音色数据集:
- Bach10
- PHENICX
- URMP

它们的共性是都比较小，方便下载，多音色，适合用于评估；且均为现实演奏的录制，结果更令人信服。

首先要对数据集进行处理：音频要转换为22050Hz、标注要转化为统一格式（选择了mid，之所以不用数据集中的midi是因为那是原曲的midi而不是演奏的midi。演奏的标注一般以表格的形式给出），每个数据集的处理分别在：
- [bach10.ipynb](bach10.ipynb)
- [phenicx.ipynb](phenicx.ipynb)
- [urmp.ipynb](urmp.ipynb)

然后记录模型的运行结果，后面调阈值的时候就不需要反复计算了。每个数据集的评估分为两个部分：帧级评估和音符级评估。帧级只使用frame的二值化，通过迭代细分找到F1值最高的阈值。音符级评估建立在最优帧级阈值上，通过细分迭代找到F值最优的onset阈值，使用basicpitch的方法得到音符，评估指标为：音高是否正确、音头是否在标准值的50ms内（不关心音尾）。

“音色无关转录”的评估十分简单，见[eval_basicamt.ipynb](eval_basicamt.ipynb)。对于“音色分离转录”，需要用PIT进行匹配，即选择得分最高的排列；而“得分最高”可以用最小误差代替，见[eval_septimbre.ipynb](eval_septimbre.ipynb)。

评估使用了库`mir_eval`。我安装了`mirdata`库，它的依赖包含了`mir_eval`。注意根据[mirdata:issue627](https://github.com/mir-dataset-loaders/mirdata/issues/627)所言，需要先去`jams`的github库下载源码安装最新`jams`，然后再安装`mirdata`。不过最新的`mirdata`已经不存在版本问题了。

仅仅进行了帧级评估，因为音符化引入了新参数，会影响对模型能力的判断。

## 结果
### 音色无关转录
有三个模型参与了比较：
- **basicamt**: 我的“音色无关转录”模型
- **basicpitch**: 使用我的数据集训练的basicpitch模型，删去了pitch输出
- **onsets&frames**: 一个参数量比较大的模型，只保留了onset和frame头

### 音色分离转录
- septimbre: 我的“音色分离转录”模型

暂时找不到可以比较的同类开源模型。帧级评估如下：
<table border="2">
  <caption>音色分离转录模型septimbre评估结果</caption>
  <thead>
    <tr>
      <th rowspan="1">类型</th>
      <th colspan="4">音色无关转录</th>
      <th colspan="4">音色分离转录</th>
    </tr>
    <tr>
      <th>混合数</th>
      <th>Acc</th>
      <th>P</th>
      <th>R</th>
      <th>F1</th>
      <th>Acc</th>
      <th>P</th>
      <th>R</th>
      <th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2</td>
      <td>0.680</td>
      <td>0.823</td>
      <td>0.789</td>
      <td>0.804</td>
      <td>0.419</td>
      <td>0.586</td>
      <td>0.558</td>
      <td>0.563</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.625</td>
      <td>0.786</td>
      <td>0.749</td>
      <td>0.766</td>
      <td>0.270</td>
      <td>0.436</td>
      <td>0.402</td>
      <td>0.407</td>
    </tr>
  </tbody>
</table>

可以发现混合数越多效果越差，且进行分离后准确率大打折扣，这非常正常；但即使不进行分离，也比不过basicamt，说明聚类损失影响了amt损失。

## 文件结构
```
│  eval_basicamt.ipynb      [eval my timbre-independent transcription model]
│  eval_basicpitch.ipynb    [eval basic-pitch model trained with my data]
│  eval_basicpitch_raw.ipynb[eval basic-pitch model trained by its author]
│  eval_septimbre.ipynb     [eval my timbre-separation transcription model]
|
│  bach10.ipynb     [pre-process of BACH10 dataset]
│  phenicx.ipynb    [pre-process of PHENICX dataset]
│  urmp.ipynb       [pre-process of URMP dataset]
|
│  README.md        [this file]
│          
├─basicamt          [from ./eval_basicamt.ipynb]
│  ├─BACH10_eval
│  │      01-AchGottundHerr@0.npy
│  │      ...
│  │      10-NunBitten@4.npy
│  │      
│  ├─PHENICX_eval
│  │      ...
│  │      mozart.npy
│  │      
│  └─URMP_eval
│          01_Jupiter_vn_vc@0.npy
│          ...
│          44_K515_vn_vn_va_va_vc@5.npy
│          
├─basicpitch        [from ./eval_basicpitch.ipynb]
|   ... (same with ./basicamt)
│          
├─basicpitch_raw    [from ./eval_basicpitch_raw]
|   ... (same with ./basicamt)
│          
├─septimbre         [from ./eval_septimbre.ipynb]
│  └─BACH10_eval
│      ├─01-AchGottundHerr_1&2
│      │      emb.npy
│      │      midi.npy
│      │      note.npy
|       ...
│      │      
│      └─10-NunBitten_3&4
│              ...
│  
├─BACH10_processed   [from ./bach10.ipynb]
│  ├─01-AchGottundHerr@0
│  │      01-AchGottundHerr.mid
│  │      01-AchGottundHerr.npy
│  │      01-AchGottundHerr.wav  
|   ...
│  │      
│  └─10-NunBitten@4
│          ...
│          
├─PHENICX_processed  [from ./phenicx.ipynb]
│  ├─beethoven
│  │      beethoven.mid
│  │      beethoven.npy
│  │      beethoven.wav
|   ...
|  |
│  └─mozart
│          ...
│              
└─URMP_processed     [from ./urmp.ipynb]
    ├─01_Jupiter_vn_vc@0
    │      01_Jupiter_vn_vc.mid
    │      01_Jupiter_vn_vc.npy
    │      01_Jupiter_vn_vc.wav
     ...
    │      
    └─44_K515_vn_vn_va_va_vc@5
            5_vc_44_K515.mid
            5_vc_44_K515.npy
            5_vc_44_K515.wav
```