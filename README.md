![architecture](./readmeSRC/arch.jpg)

[中文README](./README.zh.md)

#  [A Lightweight Architecture for Multi-instrument Transcription with Practical Optimizations](https://arxiv.org/abs/2509.12712)👈paper
The goal of this project is to develop a lightweight AI model for audio source separation and music transcription that generalizes well beyond the training data — specifically, one that can handle unseen timbres without relying on pre-defined instrument categories in the training set.

- **Lightweight**: Designed for real-world applicability. The final model is half the size of the baseline and has been successfully deployed in [noteDigger](https://madderscientist.github.io/noteDigger/). While many excellent commercial transcription tools already exist, and large models using tokenization (e.g., MT3) are trending toward universal solutions, we deliberately chose a lightweight approach to avoid direct competition. However, this constraint significantly limits the range of applicable techniques.
- **Source Separation**: More precisely termed blind source separation, as it aims to separate instruments based purely on timbre without prior knowledge from a training set. Unlike conventional source separation (which reconstructs separated spectrograms), our method operates directly at the note level.
- **Music Transcription**: Combined with source separation, this refers to "transcribing multi-track recordings": given a polyphonic audio mixture containing multiple timbres, the system outputs multiple note tracks, each corresponding to a distinct timbre.

Current timbre-aware transcription approaches suffer from poor generalization, primarily due to two issues:

1. Heavy reliance on datasets: Models essentially "memorize" instrument classes from training data rather than learning to distinguish timbres in a generalizable way. They perform exceptionally well on specific instruments seen during training but fail completely on unseen ones. Only very large models like MT3 begin to overcome this limitation.
2. Architectural constraints on separation capacity: Similar to challenges in speech separation (e.g., unknown number of speakers), classification models require a fixed number of output sources. A model trained for two instruments cannot handle three, severely limiting practical utility.

The goal of this research is to address the two issues mentioned above, while also prioritizing usability—specifically, enabling full functionality directly within a web browser. Human perception of timbre does not work by classifying instruments into predefined categories; rather, we distinguish timbres without necessarily knowing the instrument names. In other words, the task is not "timbre classification" but "timbre discrimination." Here is my hypothesis: when we hear a note with a certain timbre, we compare it against timbres stored in our memory. If it closely matches an existing memory, we group it into that category; if it differs significantly, we treat it as a new timbre. The "memory" of timbres consists partly of previously learned examples (analogous to a training set), but more importantly, it is dynamically built from the earlier parts of the same audio being listened to—essentially performing dynamic timbre learning and online clustering. This "dynamic" nature is key to achieving our research objectives.

This study decomposes the task into two stages:

1. Timbre-agnostic transcription: First, transcribe all notes without relying on timbre-based classification. The model is inspired by BasicPitch but includes improvements. (magnitude encoding)
2. Timbre-separated transcription: Then, assign timbre labels to the notes obtained from stage 1 through clustering timbre embeddings obtained at this stage. (directional encoding)

Contributions of this research:


1. A lightweight timbre-agnostic transcription model, halving both parameter count and computational overhead compared to the baseline, while maintaining comparable performance. The model has only 18,978 trainable parameters yet demonstrates strong generalization and accuracy.
2. An extended timbre-encoding branch built upon the timbre-agnostic transcription model, capable of correctly separating 2–3 instruments with over 70% accuracy.
3. A novel deep clustering post-processing method specifically designed for music transcription, which performs clustering at the note level, enhancing robustness and reducing computational cost.
4. Optimized loss functions: (a) refined the weighting strategy of BasicPitch; (b) replaced conventional deep clustering loss that applies MSE on affinity matrix with contrastive learning losses InfoNCE, yielding superior embedding quality.

## Future Work
Although "memory" was mentioned above, this study did not implement such a mechanism. The original idea was to let the model first process the input audio once to build a "memory," then re-process the same input while querying this memory to produce the final output. In this memory, similar timbres would already be grouped together. I experimented with Hopfield Networks and extended them into attention mechanisms. However, when the timbre-encoding network lacked sufficient capacity, introducing memory led to category merging—a phenomenon I call "memory blurring." Theoretically, a Hopfield network’s maximum memory capacity is roughly 0.14 times the encoding dimension. To separate three timbres, the encoding dimension would need to be at least 22 — an impractical size for browser-deployable models. Thus, I suspect this memory mechanism may become viable only with larger models. The explored architectures are preserved in [./model/attention.py](./model/attention.py). Refer to [./model/memory.md](./model/memory.md) for more detailed descriptions.

I also designed a synthetic data generation method, but its performance fell far short of real-world datasets. The paper discusses its limitations, one major issue being the "lack of pitch-range constraints": while timbre can be approximated as invariant across adjacent pitches, it varies significantly over wider intervals. A potential solution is to generate notes within specific frequency bands. Of course, many modern algorithms simulate human-like composition; integrating such methods might yield much better results.

Although the ultimate goal is "training-set independence," a large and diverse training set remains essential for learning generalizable representations. Good generalization does not imply that a small training set suffices.

The two branches of this model encode magnitude and direction separately. I believe a more promising approach would be to directly learn a unified feature vector where direction represents timbre and magnitude represents intensity—akin to Hinton’s Capsule Networks. Unfortunately, small networks seem to lack the capacity to achieve such representation (I tried but failed). However, I remain optimistic that sufficiently large models could realize this idea.

I highly recommend the paper "Harmonic Frequency-Separable Transformer for Instrument-Agnostic Music Transcription." Although it does not perform timbre separation, it employs several techniques I considered highly promising (though they're not very useful to this study), such as harmonic convolution and attention mechanisms for timbre modeling.

Regarding phase: phase information plays a minimal role in musical signals (as evidenced by additive synthesizers). Since our task does not require audio reconstruction, phase can be safely discarded. Even if audio reconstruction were needed, I suspect that losing phase information would have negligible impact.

## Folder Structure
```
├─basicamt [our timbre-agnostic transcription model]
├─basicpitch [baseline1 of timbre-agnostic transcription]
├─onsets_frames [baseline2 of timbre-agnostic transcription]
|
├─septimbre [our timbre-separated transcription model]
├─Tanaka [baseline of timbre-separated transcription]
|
├─evaluate [notebook for model evaluation]
|
├─data 
├─model [some common torch.nn.Module]
└─utils [some common tool]
```

## Usage
### online usage
This project has been integrated into [noteDigger](https://madderscientist.github.io/noteDigger/), enabling convenient manual post-processing or assisting manual transcription. Usage instructions are provided below:

![in noteDigger](readmeSRC/how_to_use.png)

> Instrument-separated transcription has also been deployed, but the number of instrument classes is needed.

### for development
The main training results of this project have been exported to ONNX, and they can be used once the runtime environment is properly configured. For details on the model's input and output, please refer to the ONNX export section in `use_model.ipynb` in each folder.

### 训练 for training
This project uses `uv` to manage the environment. Please ensure it has been installed. Then, in the root directory of this project, execute:

```
uv sync
```

Then you can execute `. ipynb`. The first step is to prepare the data and follow the instructions in the [data](./data/README.md) folder.

In addition, the project relies on `ffmpeg`, which should be able to be directly called through the command line and requires additional installation.

> Note: We found that EPS has a significant impact on model performance (smaller EPS values lead to better results when taking the logarithm). However, if EPS is too small, it may be interpreted as zero when exported to ONNX and run in a browser, resulting in NaN values. Therefore, we ultimately selected 1.01e-8.