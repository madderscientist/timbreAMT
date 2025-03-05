# 模型评估
使用了三个多音色数据集:
- Bach10
- PHENICX
- URMP

它们的共性是都比较小，方便下载，适合用于评估。

首先要对数据集进行处理：音频要转换为22050Hz、标注要转化为统一格式（选择了mid，之所以不用数据集中的midi是因为那是原曲的midi而不是演奏的midi。演奏的标注一般以表格的形式给出），每个数据集的处理分别在：
- [bach10.ipynb](bach10.ipynb)
- [phenicx.ipynb](phenicx.ipynb)
- [urmp.ipynb](urmp.ipynb)

然后记录模型的运行结果，后面调阈值二值化的时候就不需要反复计算了。最后进行模型的帧级评估，并用细分的方式得到最佳阈值，为后续的音符创建提供参考。对于“音色无关转录”，帧级评估非常简单，见[eval_basicamt.ipynb](eval_basicamt.ipynb)。

评估使用了库mir_eval。我安装了mirdata库，它的依赖包含了mir_eval。注意根据[mirdata:issue627](https://github.com/mir-dataset-loaders/mirdata/issues/627)所言，需要先去jams的github库下载源码安装最新jams，然后再安装mirdata（不过好像JAMS已经被mirdata新的pull request删除了？也许从最新的源码上安装会好很多）。