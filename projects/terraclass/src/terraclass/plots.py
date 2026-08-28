"""Plot experiment history, confusion matrices, and Grad-CAM overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def plot_history(history: list[dict[str, object]], path: str | Path) -> None:
    epochs = [record["epoch"] for record in history]
    train_loss = [record["train"]["loss"] for record in history]
    val_loss = [record["val"]["loss"] for record in history]
    train_f1 = [record["train"]["macro_f1"] for record in history]
    val_f1 = [record["val"]["macro_f1"] for record in history]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()

    axes[1].plot(epochs, train_f1, label="train")
    axes[1].plot(epochs, val_f1, label="validation")
    axes[1].set(title="Macro F1", xlabel="Epoch", ylabel="F1", ylim=(0, 1))
    axes[1].legend()
    figure.tight_layout()
    _save_figure(figure, path)


def plot_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    class_names: Sequence[str],
    path: str | Path,
) -> None:
    values = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(values, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        title="Test confusion matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = values.max() / 2 if values.size else 0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                str(values[row, column]),
                ha="center",
                va="center",
                color="white" if values[row, column] > threshold else "black",
                fontsize=8,
            )
    figure.tight_layout()
    _save_figure(figure, path)


def save_gradcam_overlay(
    image_path: str | Path,
    heatmap: np.ndarray,
    path: str | Path,
    *,
    title: str,
) -> None:
    image = Image.open(image_path).convert("RGB")
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(image)
    axis.imshow(
        heatmap,
        cmap="jet",
        alpha=0.4,
        interpolation="bilinear",
        extent=(0, image.width, image.height, 0),
    )
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    _save_figure(figure, path)


def _save_figure(figure, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
