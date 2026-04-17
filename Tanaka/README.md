# "音色分离转录"Baseline —— Tanaka deep clustering

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

--------

# "Timbre-Separation Transcription" Baseline —— Tanaka deep clustering

Since the original paper did not open-source their code, the following parameters were configured based on empirical experience and adaptation to this project:

- MultiPitch Estimation: Utilizes my pre-trained BasicAMT model with fixed parameters.
- STFT Settings: The original paper used a 2048-point FFT for audio sampled at 44100Hz. Since this project uses a 22050Hz sampling rate, a 1024-point FFT is used to maintain the same frequency resolution.
- Hop Length: Maintained the original 11ms hop size (which aligns with this project's settings).
- Architecture Changes: The audio separation branch was removed due to the lack of supervised data.
- BiLSTM: The Hidden Size is set to 256. While this seems reasonable, the parameter count is still massive.
- Embedding Size: Set to 12, consistent with my other models.
- Loss Function: Adopted my custom loss function, as it has been proven to be more effective.

The disadvantage of this model is obvious - there are too many parameters, making it very easy to overfit, resulting in poor evaluation performance. Perhaps I cut off the supervised branch for audio separation, which resulted in a decrease in performance.

## Issue: Suspected PyTorch bug.
ONNX export failed; the dynamic dimensions of the encoder remain固化 (hardcoded/fixed) instead of remaining dynamic.