# 模型评估
使用了三个多音色数据集:
- Bach10
- PHENICX
- URMP

它们的共性是都比较小，方便下载，多音色，适合用于评估；且均为显示演奏的录制，结果更令人信服。

首先要对数据集进行处理：音频要转换为22050Hz、标注要转化为统一格式（选择了mid，之所以不用数据集中的midi是因为那是原曲的midi而不是演奏的midi。演奏的标注一般以表格的形式给出），每个数据集的处理分别在：
- [bach10.ipynb](bach10.ipynb)
- [phenicx.ipynb](phenicx.ipynb)
- [urmp.ipynb](urmp.ipynb)

然后记录模型的运行结果，后面调阈值二值化的时候就不需要反复计算了。最后进行模型的帧级评估，并用细分的方式得到最佳阈值，为后续的音符创建提供参考。对于“音色无关转录”，帧级评估非常简单，见[eval_basicamt.ipynb](eval_basicamt.ipynb)。对于“音色分离转录”，需要用PIT进行匹配，即选择得分最高的排列；而“得分最高”可以用最小误差代替，见[eval_septimbre.ipynb](eval_septimbre.ipynb)。

评估使用了库`mir_eval`。我安装了`mirdata`库，它的依赖包含了`mir_eval`。注意根据[mirdata:issue627](https://github.com/mir-dataset-loaders/mirdata/issues/627)所言，需要先去`jams`的github库下载源码安装最新`jams`，然后再安装`mirdata`（不过好像`jams`已经被`mirdata`新的`pull request`删除了？也许从最新的源码上安装会好很多）。

仅仅进行了帧级评估，因为音符化引入了新参数，会影响对模型能力的判断。

## 结果
### 音色无关转录
- **basicamt**: 我的“音色无关转录”模型
- **basicpitch**: 使用我的数据集训练的basicpitch模型
- **basicpitch_raw**: 其论文中使用的模型，使用pip安装

<table border="2">
    <caption>音色无关转录模型评估结果</caption>
    <tr>
        <th>数据集</th> <th>指标</th> <th>basicpitch</th> <th>basicamt</th> <th>basicpitch_raw</th>
    </tr>
    <tr> <td rowspan="5">BACH10<br>合奏</td>
  		<td>阈值</td>
        <td>0.212</td> <td>0.15312</td> <td>0.356</td>
    </tr>
    <tr>
        <td>Acc</td>
        <td>0.64909</td> <td>0.67135</td> <td>0.68033</td>
    </tr>
    <tr>
        <td>P</td>
        <td>0.77487</td> <td>0.80869</td> <td>0.81744</td>
    </tr>
    <tr>
        <td>R</td>
        <td>0.79957</td> <td>0.79789</td> <td>0.80097</td>
    </tr>
    <tr>
        <td>F1</td>
        <td>0.78689</td> <td>0.80291</td> <td><strong>0.80916</strong></td>
    </tr>
    <tr> <td rowspan="5">BACH10<br>所有</td>
  		<td>阈值</td>
        <td>0.385</td> <td>0.30184</td> <td>0.47428</td>
    </tr>
    <tr>
        <td>Acc</td>
        <td>0.79255</td>  <td>0.76612</td> <td>0.71582</td>
    </tr>
    <tr>
        <td>P</td>
        <td>0.88961</td> <td>0.87571</td> <td>0.83161</td>
    </tr>
    <tr>
        <td>R</td>
        <td>0.87691</td> <td>0.85777</td> <td>0.83598</td>
    </tr>
    <tr>
        <td>F1</td>
        <td><strong>0.87913</strong></td> <td>0.86087</td> <td>0.83157</td>
    </tr>
    <tr> <td rowspan="5">PHENICX<br>合奏</td>
  		<td>阈值</td>
        <td>0.13912</td> <td>0.06464</td> <td>0.2032</td>
    </tr>
    <tr>
        <td>Acc</td>
        <td>0.33753</td> <td>0.42476</td> <td>0.26585</td>
    </tr>
    <tr>
        <td>P</td>
        <td>0.52686</td> <td>0.58628</td> <td>0.37882</td>
    </tr>
    <tr>
        <td>R</td>
        <td>0.48936</td> <td>0.60524</td> <td>0.46700</td>
    </tr>
    <tr>
        <td>F1</td>
        <td>0.50307</td> <td><strong>0.59512</strong></td> <td>0.41823</td>
    </tr>
    <tr> <td rowspan="5">URMP<br>合奏</td>
  		<td>阈值</td>
        <td>0.19984</td> <td>0.13840</td> <td>0.30640</td>
    </tr>
    <tr>
        <td>Acc</td>
        <td>0.52076</td> <td>0.57102</td> <td>0.36206</td>
    </tr>
    <tr>
        <td>P</td>
        <td>0.68602</td> <td>0.74857</td> <td>0.49911</td>
    </tr>
    <tr>
        <td>R</td>
        <td>0.68401</td> <td>0.70437</td> <td>0.53963</td>
    </tr>
    <tr>
        <td>F1</td>
        <td>0.68058</td> <td><strong>0.72323</strong></td> <td>0.51706</td>
    </tr>
    <tr> <td rowspan="5">URMP<br>独奏</td>
  		<td>阈值</td> <td>0.3848</td> <td>0.35183</td> <td>0.49</td>
    </tr>
    <tr>
        <td>Acc</td>
        <td>0.67248</td> <td>0.68630</td> <td>0.40660</td>
    </tr>
    <tr>
        <td>P</td>
        <td>0.82921</td> <td>0.85788</td> <td>0.54802</td>
    </tr>
    <tr>
        <td>R</td>
        <td>0.77676</td> <td>0.76998</td> <td>0.56436</td>
    </tr>
    <tr>
        <td>F1</td>
        <td>0.79619</td> <td><strong>0.80552</strong></td> <td>0.55520</td>
    </tr>
    <tr>
        <td>参数量</td>
        <td>CQT<br>19944</td><td>56517<br>不含CQT</td> <td>46564<br>含CQT</td>
        <td>27518<br>不含CQT</td>
    </tr>
</table>

> 注：`basicpitch_raw`的参数量按照论文给出的图，用pytorch搭建并计算。两个`basicpitch`的实际参数量应该加上CQT参数，但由于其未训练CQT参数，故表中没有加。

在`basicamt`和`basicpitch`的比较中，可以发现我的模型更小但更强；在`basicpitch`和`basicpitch_raw`的比较中，可以发现我的数数据集竟然更好？此外`basicpitch`的阈值普遍高于`basicamt`，这是损失函数选择不同带来的影响。

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