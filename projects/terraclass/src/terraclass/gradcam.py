"""A small Grad-CAM implementation for understanding model attention."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def create_gradcam(
    model: nn.Module,
    target_layer: nn.Module,
    input_tensor: Tensor,
    class_index: int | None = None,
) -> tuple[Tensor, Tensor, int]:
    """Return a normalized heatmap, logits, and the explained class index."""
    activations: list[Tensor] = []
    gradients: list[Tensor] = []

    def save_activation(_module, _inputs, output):
        activations.append(output)
        output.register_hook(lambda gradient: gradients.append(gradient))

    forward_handle = target_layer.register_forward_hook(save_activation)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor)
        selected_class = int(logits.argmax(dim=1).item()) if class_index is None else class_index
        if not 0 <= selected_class < logits.shape[1]:
            raise ValueError(f"class_index must be between 0 and {logits.shape[1] - 1}")
        logits[0, selected_class].backward()

        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")
        activation = activations[-1].detach()
        gradient = gradients[-1].detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        heatmap = F.interpolate(
            heatmap,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        heatmap -= heatmap.min()
        maximum = heatmap.max()
        if maximum > 0:
            heatmap /= maximum
        return heatmap.cpu(), logits.detach().cpu(), selected_class
    finally:
        forward_handle.remove()
