"""Experiment metadata and scratch-model checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from support_router.models import ScratchModelConfig, ScratchTransformerClassifier
from support_router.utils import load_json, save_json

METADATA_FILE = "experiment.json"


def save_metadata(metadata: dict[str, Any], experiment_dir: str | Path) -> None:
    save_json(metadata, Path(experiment_dir) / METADATA_FILE)


def load_metadata(experiment_dir: str | Path) -> dict[str, Any]:
    path = Path(experiment_dir) / METADATA_FILE
    if not path.exists():
        raise FileNotFoundError(f"Experiment metadata not found: {path}")
    value = load_json(path)
    if not isinstance(value, dict) or "model_type" not in value or "class_names" not in value:
        raise ValueError("Invalid SupportRouter experiment metadata")
    return value


def save_scratch_checkpoint(
    model: ScratchTransformerClassifier,
    path: str | Path,
    *,
    epoch: int,
    best_validation_f1: float,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config.to_dict(),
            "epoch": epoch,
            "best_validation_f1": best_validation_f1,
        },
        destination,
    )


def load_scratch_model(path: str | Path, device: torch.device) -> ScratchTransformerClassifier:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if "model_state_dict" not in checkpoint or "model_config" not in checkpoint:
        raise ValueError("Invalid scratch-transformer checkpoint")
    config = ScratchModelConfig(**checkpoint["model_config"])
    model = ScratchTransformerClassifier(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()
