"""Multi-class metrics and confidence calibration."""

from __future__ import annotations

import torch
from torch import Tensor


def confusion_matrix(targets: Tensor, predictions: Tensor, num_classes: int) -> Tensor:
    targets = targets.to(dtype=torch.int64, device="cpu").flatten()
    predictions = predictions.to(dtype=torch.int64, device="cpu").flatten()
    if targets.shape != predictions.shape:
        raise ValueError("targets and predictions must have the same shape")
    if targets.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.int64)
    if targets.min() < 0 or predictions.min() < 0:
        raise ValueError("class indices cannot be negative")
    if targets.max() >= num_classes or predictions.max() >= num_classes:
        raise ValueError("class index is outside num_classes")
    encoded = targets * num_classes + predictions
    return torch.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)


def classification_metrics(
    logits: Tensor,
    targets: Tensor,
    *,
    num_classes: int,
    calibration_bins: int = 10,
) -> dict[str, object]:
    """Calculate aggregate, per-class, top-k, and calibration metrics."""
    logits = logits.detach().to(dtype=torch.float64, device="cpu")
    targets = targets.detach().to(dtype=torch.int64, device="cpu").flatten()
    if logits.ndim != 2 or logits.shape[0] != targets.shape[0]:
        raise ValueError("logits must have shape [examples, classes] matching targets")
    if logits.shape[1] != num_classes:
        raise ValueError("logit class dimension does not match num_classes")

    probabilities = logits.softmax(dim=1)
    confidences, predictions = probabilities.max(dim=1)
    matrix = confusion_matrix(targets, predictions, num_classes)
    matrix_float = matrix.to(torch.float64)
    true_positives = matrix_float.diag()
    predicted_counts = matrix_float.sum(dim=0)
    actual_counts = matrix_float.sum(dim=1)
    precision = torch.where(predicted_counts > 0, true_positives / predicted_counts, 0.0)
    recall = torch.where(actual_counts > 0, true_positives / actual_counts, 0.0)
    denominator = precision + recall
    f1 = torch.where(denominator > 0, 2 * precision * recall / denominator, 0.0)
    accuracy = (predictions == targets).double().mean().item()

    top_k = min(3, num_classes)
    top_predictions = logits.topk(top_k, dim=1).indices
    top_k_accuracy = top_predictions.eq(targets.unsqueeze(1)).any(dim=1).double().mean().item()

    return {
        "accuracy": accuracy,
        "top_3_accuracy": top_k_accuracy,
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
        "expected_calibration_error": expected_calibration_error(
            confidences,
            predictions.eq(targets),
            num_bins=calibration_bins,
        ),
        "mean_confidence": confidences.mean().item(),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "support": actual_counts.to(dtype=torch.int64).tolist(),
        "confusion_matrix": matrix.tolist(),
    }


def expected_calibration_error(
    confidences: Tensor,
    correct: Tensor,
    *,
    num_bins: int = 10,
) -> float:
    """Measure the weighted gap between confidence and accuracy in confidence bins."""
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    confidences = confidences.to(dtype=torch.float64, device="cpu").flatten()
    correct = correct.to(dtype=torch.float64, device="cpu").flatten()
    if confidences.shape != correct.shape:
        raise ValueError("confidences and correct must have the same shape")
    if confidences.numel() == 0:
        return 0.0

    boundaries = torch.linspace(0, 1, num_bins + 1, dtype=torch.float64)
    calibration_error = torch.tensor(0.0, dtype=torch.float64)
    for index in range(num_bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        in_bin = (confidences > lower) & (confidences <= upper)
        if in_bin.any():
            proportion = in_bin.double().mean()
            accuracy = correct[in_bin].mean()
            confidence = confidences[in_bin].mean()
            calibration_error += proportion * (accuracy - confidence).abs()
    return calibration_error.item()
