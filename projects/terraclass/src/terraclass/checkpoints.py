"""Checkpoint serialization with basic format validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

REQUIRED_KEYS = {
    "model_name",
    "image_size",
    "class_names",
    "model_state_dict",
    "epoch",
}


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, destination)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    """Load a TerraClass checkpoint containing tensors and primitive metadata."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a dictionary")
    missing = REQUIRED_KEYS.difference(checkpoint)
    if missing:
        raise ValueError(f"Invalid TerraClass checkpoint; missing: {', '.join(sorted(missing))}")
    return checkpoint
