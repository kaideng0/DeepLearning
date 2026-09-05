"""Plots for experiment learning curves and classification errors."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_history(history: list[dict[str, object]], path: str | Path) -> None:
    epochs = [record["epoch"] for record in history]
    train_loss = [record["train"]["loss"] for record in history]
    validation_loss = [record["validation"]["loss"] for record in history]
    train_f1 = [record["train"]["macro_f1"] for record in history]
    validation_f1 = [record["validation"]["macro_f1"] for record in history]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, validation_loss, label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    axes[1].plot(epochs, train_f1, label="train")
    axes[1].plot(epochs, validation_f1, label="validation")
    axes[1].set(title="Macro F1", xlabel="Epoch", ylabel="F1", ylim=(0, 1))
    axes[1].legend()
    figure.tight_layout()
    _save(figure, path)


def plot_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    class_names: Sequence[str],
    path: str | Path,
) -> None:
    values = np.asarray(matrix)
    size = max(10, min(24, len(class_names) * 0.25))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(values, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    axis.set(
        title="Test confusion matrix",
        xlabel="Predicted intent",
        ylabel="True intent",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    tick_size = 5 if len(class_names) > 30 else 8
    axis.tick_params(axis="both", labelsize=tick_size)
    plt.setp(axis.get_xticklabels(), rotation=90, ha="center")
    figure.tight_layout()
    _save(figure, path)


def _save(figure, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
