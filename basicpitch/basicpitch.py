import torch
import torch.nn as nn
import numpy as np
from typing import Callable
import sys
sys.path.append("..")
from model.layers import HarmonicaStacking

AUDIO_SAMPLE_RATE = 22050
ANNOTATIONS_BASE_FREQUENCY = 27.5  # lowest key on a piano
ANNOTATIONS_N_SEMITONES = 88  # number of piano keys
CONTOURS_BINS_PER_SEMITONE = 3
FFT_HOP = 256
N_FFT = 8 * FFT_HOP
AUDIO_WINDOW_LENGTH = 2
N_FREQ_BINS_CONTOURS = ANNOTATIONS_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE
AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP

MAX_N_SEMITONES = int(np.floor(12.0 * np.log2(0.5 * AUDIO_SAMPLE_RATE / ANNOTATIONS_BASE_FREQUENCY)))

def transcription_loss(y_true, y_pred, label_smoothing: float):
    if label_smoothing > 0:
        y_true = y_true * (1 - label_smoothing) + 0.5 * label_smoothing
    bce = nn.BCELoss()(y_pred, y_true)
    return bce


def weighted_transcription_loss(
    y_true: torch.Tensor, y_pred: torch.Tensor, label_smoothing: float, positive_weight: float = 0.5
) -> torch.Tensor:
    """The transcription loss where the positive and negative true labels are balanced by a weighting factor.

    Args:
        y_true: The true labels.
        y_pred: The predicted labels.
        label_smoothing: Smoothing factor. Squeezes labels towards 0.5.
        positive_weight: Weighting factor for the positive labels.

    Returns:
        The weighted transcription loss.
    """
    if label_smoothing > 0:
        y_true = y_true * (1 - label_smoothing) + 0.5 * label_smoothing

    negative_mask = y_true < 0.5
    nonnegative_mask = ~negative_mask

    bce_negative = nn.BCELoss()(
        y_pred[negative_mask],
        y_true[negative_mask],
    )
    bce_nonnegative = nn.BCELoss()(
        y_pred[nonnegative_mask],
        y_true[nonnegative_mask],
    )
    return ((1 - positive_weight) * bce_negative) + (positive_weight * bce_nonnegative)


def onset_loss(
    weighted: bool, label_smoothing: float, positive_weight: float
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Args:
        weighted: Whether or not to use a weighted cross entropy loss.
        label_smoothing: Smoothing factor. Squeezes labels towards 0.5.
        positive_weight: Weighting factor for the positive labels.

    Returns:
        A function that calculates the transcription loss. The function will
        return weighted_transcription_loss if weighted is true else it will return
        transcription_loss.
    """
    if weighted:
        return lambda x, y: weighted_transcription_loss(
            x, y, label_smoothing=label_smoothing, positive_weight=positive_weight
        )
    return lambda x, y: transcription_loss(x, y, label_smoothing=label_smoothing)


def basicpitch_loss(label_smoothing: float = 0.2, weighted: bool = False, positive_weight: float = 0.5):
    loss_fn = lambda x, y: transcription_loss(x, y, label_smoothing=label_smoothing)
    loss_onset = onset_loss(weighted, label_smoothing, positive_weight)
    return {
        "note": loss_fn,
        "onset": loss_onset,
    }


class BasicPitch(nn.Module):
    loss = basicpitch_loss(weighted=True, positive_weight=0.95)
    def __init__(self, harmonics=8):
        super(BasicPitch, self).__init__()
        self.CQT_BN = nn.BatchNorm2d(1)
        self.HCQT = HarmonicaStacking(HarmonicaStacking.harmonic_shifts(harmonics-1, 1, 36), 7 * 36)
        self.contours = nn.Sequential(
            nn.Conv2d(8, 32, (5, 5), padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 8, (3*13, 3), padding="same"),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.note = nn.Sequential(
            nn.Conv2d(8, 32, (7, 7), padding=(2, 3), stride=(3, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, (3, 7), padding="same"),
            nn.Sigmoid()
        )
        self.onset1 = nn.Sequential(
            nn.Conv2d(8, 32, (5, 5), stride=(3, 1), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.onset2 = nn.Sequential(
            nn.Conv2d(33, 1, (3, 3), padding="same"),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x是CQT
        # 下面这段是BasicPitch的CQT后处理
        power = torch.sum(x.pow(2), dim=1, keepdim=True)
        log_power = 10 * torch.log10(power + 1e-10) # (batch, 1, note, time)
        log_power_min = torch.amin(log_power, dim=(2, 3), keepdim=True)
        log_power_offset = log_power - log_power_min
        log_power_offset_max = log_power_offset_max = torch.amax(log_power_offset, dim=(2, 3), keepdim=True)
        log_power_normalized = self.CQT_BN(log_power_offset / (log_power_offset_max + 1e-10))
        # 源代码harmonicstacking部分
        x = self.HCQT(log_power_normalized) # (batch, 8, 7*36, len)
        contours = self.contours(x) # (batch, 8, 7*36, len)
        # 源代码可选是否计算contours，代价是变成1channel，为了公平起见不计算
        note = self.note(contours)  # (batch, 1, 7*12, len)

        onset = self.onset1(x)  # (batch, 32, 7*12, len)
        onset = torch.cat([onset, note], dim=1) # (batch, 33, 7*12, len)
        onset = self.onset2(onset)  # (batch, 1, 7*12, len)

        return onset.squeeze(1), note.squeeze(1)    # (batch, 7*12, len)

if __name__ == "__main__":
    # 输出参数量
    model = BasicPitch()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")    # 56517