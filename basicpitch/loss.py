from typing import Callable
import torch
import torch.nn as nn

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