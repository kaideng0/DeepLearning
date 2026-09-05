"""Explicit PyTorch training and evaluation loops for both model families."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from support_router.metrics import classification_metrics
from support_router.models import extract_logits


def run_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, Tensor]],
    criterion: nn.Module,
    device: torch.device,
    *,
    num_classes: int,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip: float = 1.0,
) -> dict[str, object]:
    """Run one epoch; an optimizer enables gradient calculation and updates."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_examples = 0
    all_logits: list[Tensor] = []
    all_targets: list[Tensor] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        targets = batch["labels"].to(device, non_blocking=True)
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = extract_logits(output)
            loss = criterion(logits, targets)
            if is_training:
                loss.backward()
                if gradient_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()

        batch_size = targets.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())

    if total_examples == 0:
        raise ValueError("The data loader produced no examples")
    metrics = classification_metrics(
        torch.cat(all_logits),
        torch.cat(all_targets),
        num_classes=num_classes,
    )
    metrics["loss"] = total_loss / total_examples
    return metrics
