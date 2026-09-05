"""Banking77 loading and deterministic validation splitting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset


class TextIntentDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        self.texts = list(texts)
        self.labels = [int(label) for label in labels]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {"text": self.texts[index], "label": self.labels[index]}


@dataclass(frozen=True)
class IntentSplits:
    train: TextIntentDataset
    validation: TextIntentDataset
    test: TextIntentDataset
    class_names: list[str]


def stratified_train_validation_indices(
    targets: Sequence[int],
    *,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    if len(targets) == 0:
        raise ValueError("targets cannot be empty")

    target_tensor = torch.as_tensor(targets, dtype=torch.int64)
    generator = torch.Generator().manual_seed(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for class_id in torch.unique(target_tensor, sorted=True).tolist():
        indices = torch.where(target_tensor == class_id)[0]
        indices = indices[torch.randperm(len(indices), generator=generator)]
        validation_size = max(1, round(len(indices) * validation_ratio))
        if validation_size >= len(indices):
            raise ValueError(f"Class {class_id} has too few examples for validation")
        validation_indices.extend(indices[:validation_size].tolist())
        train_indices.extend(indices[validation_size:].tolist())

    for indices in (train_indices, validation_indices):
        order = torch.randperm(len(indices), generator=generator).tolist()
        indices[:] = [indices[index] for index in order]
    return train_indices, validation_indices


def load_banking77(
    *,
    validation_ratio: float = 0.1,
    seed: int = 42,
    dataset_name: str = "PolyAI/banking77",
) -> IntentSplits:
    """Download Banking77 through Hugging Face Datasets and create a validation split."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face Datasets is required. Install the project with `pip install -e .`."
        ) from error

    raw: Any = load_dataset(dataset_name)
    official_train = raw["train"]
    official_test = raw["test"]
    class_names = list(official_train.features["label"].names)
    train_indices, validation_indices = stratified_train_validation_indices(
        official_train["label"],
        validation_ratio=validation_ratio,
        seed=seed,
    )

    def select(dataset, indices: Sequence[int]) -> TextIntentDataset:
        subset = dataset.select(indices)
        return TextIntentDataset(subset["text"], subset["label"])

    test_indices = list(range(len(official_test)))
    return IntentSplits(
        train=select(official_train, train_indices),
        validation=select(official_train, validation_indices),
        test=select(official_test, test_indices),
        class_names=class_names,
    )
