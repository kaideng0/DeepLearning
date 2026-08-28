"""The explicit PyTorch training and evaluation loops."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from terraclass.metrics import confusion_matrix, metrics_from_confusion_matrix


def run_epoch(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, object]:
    """Run one epoch; passing an optimizer enables training and backpropagation."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_examples = 0
    all_targets: list[Tensor] = []
    all_predictions: list[Tensor] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, targets)
            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        all_targets.append(targets.detach().cpu())
        all_predictions.append(logits.argmax(dim=1).detach().cpu())

    if total_examples == 0:
        raise ValueError("The data loader produced no examples")

    targets = torch.cat(all_targets)
    predictions = torch.cat(all_predictions)
    num_classes = _infer_num_classes(model, targets, predictions)
    matrix = confusion_matrix(targets, predictions, num_classes)
    metrics = metrics_from_confusion_matrix(matrix)
    metrics["loss"] = total_loss / total_examples
    metrics["confusion_matrix"] = matrix.tolist()
    return metrics


def _infer_num_classes(model: nn.Module, targets: Tensor, predictions: Tensor) -> int:
    """Infer class count from the classifier, falling back to observed labels."""
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return model.fc.out_features
    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, nn.Sequential):
        for layer in reversed(classifier):
            if isinstance(layer, nn.Linear):
                return layer.out_features
    return int(torch.cat((targets, predictions)).max().item()) + 1
