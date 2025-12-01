# "音色分离转录"Baseline——BasicPitch

MULTI-INSTRUMENT MUSIC TRANSCRIPTION BASED ON DEEP SPHERICAL CLUSTERING OF SPECTROGRAMS AND PITCHGRAMS

但是论文没有开源代码，只能凭经验设置参数：
- MultiPitch Estimation 使用了我的BasicAMT训练好的模型，固定参数
- 原文STFT对44100Hz采样的音频使用了2048点的FFT，由于本项目使用22050Hz，为了保持相同的频域分辨率，使用了1024点
- 保持了原文的Hop=11ms（刚好和本项目一样）
- 删去了音频分离分支，因为没有监督数据
- BiLSTM 的 Hidden Size 取了256，感觉比较合理，但是参数量还是爆炸
- Embedding Size 取了12，和我的保持一致。
- 损失函数用的我的，因为已经证明更好

这个模型的缺点显而易见——参数量超多，因此非常容易过拟合，所以评估效果不太行。可能是我砍掉了音频分离的监督分支，导致性能下降。

## 导出失败
疑似pytorch的bug。导出onnx失败，encoder的动态维度一直被固化