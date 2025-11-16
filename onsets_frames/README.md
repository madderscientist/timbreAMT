# _Onsets and Frames_ reproduction
origin paper: _Onsets and Frames: Dual-Objective Piano Transcription_

code modified from: https://github.com/jongwook/onsets-and-frames

To maintain comparability, only the onset and frame branches were retained during replication, and the parameters were adjusted to make them compatible with the current training data. In comparison, this model has a large number of learnable parameters: 1,714,076 (9 times that of my model, which has 18,978). This model is the first typical "note-level transcription": it utilizes LSTM to process temporal information and predicts onsets, but this also results in extremely long CPU running times, making it not worth deploying.

为了保持可比性，复现时只保留了onset和frame分支，并调整了参数使其兼容当前的训练数据。相比之下，这个模型可学习参数量很大：1714076（9倍于我的模型18978）。这个模型是第一个典型的“音符级转录”：利用LSTM实现了时序信息的处理并输出了音头，然而这也导致cpu运行时间超久，不具备部署价值。

Initially, we trained using the loss function provided in the paper, but the results were unsatisfactory and prone to overfitting. After training for 5 epochs, the loss on the evaluation set skyrocketed.

首先使用论文的损失进行训练，但效果并不好，并且容易过拟合，训练5个epoch，在eval上的损失就会暴涨。

However, after training six times, I accidentally obtained a model that performed exceptionally well on `孤独な巡礼simple.wav`, which was saved in `best_of_model.pth` and can be loaded using the following code. However, its score on URMP is still very low. Below is its transcription result for [/data/inferMusic/孤独な巡礼simple.wav](../data/inferMusic/孤独な巡礼simple.wav), which has very few ghost images:

但是训练了6次，偶然得到了一种在`孤独な巡礼simple.wav`上效果特别好的模型，保存在了 `best_of_model.pth` 中，可以用如下的代码加载。但是在URMP上评分依然很低。下面是其对[/data/inferMusic/孤独な巡礼simple.wav](../data/inferMusic/孤独な巡礼simple.wav)的转录结果，虚影非常少:

```py
torch.load(f"best_of_model.pth", weights_only=False)
```

![result_origin_train_scheme](<onsets and frames output.png>)

Therefore, we switched to our loss function (focal loss), which significantly delayed overfitting and improved the results.
于是换为我的损失函数，过拟合被大大延迟，效果更好。

The results are still decent: using the Mel spectrum reduces octave errors, but it tends to lose octave information. Additionally, the onset results are interesting: they span the entire frequency domain directly, rather than being confined to a specific time-frequency unit. I believe this is due to the final fully connected layer, as each frame of features is expanded into a piano roll without frequency position information. Perhaps cross-layer connections can help address this issue.

结果还是不错的：用mel谱使得八度误差少，但容易丢失八度信息。此外，发现onset结果很有趣：直接横跨整个频域，而不是某个时频单元。我认为这是最后的全连接导致的，每一帧特征展开为piano-roll的时候没有频率位置信息，可能跨层连接可以帮助解决这个问题。

Transcription result for [/data/inferMusic/孤独な巡礼simple.wav](../data/inferMusic/孤独な巡礼simple.wav):

下图是对
[/data/inferMusic/孤独な巡礼simple.wav](../data/inferMusic/孤独な巡礼simple.wav)的转录结果:

![result](<onsets and frames output focal.png>)


