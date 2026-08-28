"""Classification metrics implemented with PyTorch tensors."""

from __future__ import annotations

import torch
from torch import Tensor


def confusion_matrix(targets: Tensor, predictions: Tensor, num_classes: int) -> Tensor:
    """Return a matrix whose rows are true classes and columns are predictions."""
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


def metrics_from_confusion_matrix(matrix: Tensor) -> dict[str, object]:
    """Calculate accuracy and macro/per-class precision, recall, and F1."""
    matrix = matrix.to(dtype=torch.float64, device="cpu")
    true_positives = matrix.diag()
    predicted_counts = matrix.sum(dim=0)
    actual_counts = matrix.sum(dim=1)

    precision = torch.where(predicted_counts > 0, true_positives / predicted_counts, 0.0)
    recall = torch.where(actual_counts > 0, true_positives / actual_counts, 0.0)
    denominator = precision + recall
    f1 = torch.where(denominator > 0, 2 * precision * recall / denominator, 0.0)
    total = matrix.sum()
    accuracy = (true_positives.sum() / total).item() if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "support": actual_counts.to(dtype=torch.int64).tolist(),
    }
