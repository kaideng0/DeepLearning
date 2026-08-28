"""EuroSAT loading, transforms, and deterministic stratified splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import EuroSAT

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TransformSubset(Dataset):
    """Apply a split-specific transform to selected items from one base dataset."""

    def __init__(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        transform: Callable | None = None,
    ) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, target = self.dataset[self.indices[index]]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


@dataclass(frozen=True)
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]


def stratified_split_indices(
    targets: Sequence[int],
    *,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Split every class proportionally so minority classes remain represented."""
    if not 0 <= val_ratio < 1 or not 0 <= test_ratio < 1:
        raise ValueError("val_ratio and test_ratio must each be in [0, 1)")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be less than 1")
    if len(targets) == 0:
        raise ValueError("targets cannot be empty")

    target_tensor = torch.as_tensor(targets, dtype=torch.int64)
    generator = torch.Generator().manual_seed(seed)
    splits: list[list[int]] = [[], [], []]

    for class_id in torch.unique(target_tensor, sorted=True).tolist():
        class_indices = torch.where(target_tensor == class_id)[0]
        class_indices = class_indices[torch.randperm(len(class_indices), generator=generator)]
        class_size = len(class_indices)
        val_size = round(class_size * val_ratio)
        test_size = round(class_size * test_ratio)
        if val_size + test_size >= class_size:
            raise ValueError(f"Class {class_id} has too few samples for the requested split")

        splits[1].extend(class_indices[:val_size].tolist())
        splits[2].extend(class_indices[val_size : val_size + test_size].tolist())
        splits[0].extend(class_indices[val_size + test_size :].tolist())

    # Shuffle each completed split so examples are not grouped by class.
    for split in splits:
        order = torch.randperm(len(split), generator=generator).tolist()
        split[:] = [split[index] for index in order]

    return splits[0], splits[1], splits[2]


def build_transforms(image_size: int) -> tuple[Callable, Callable]:
    """Return augmentation for training and deterministic preprocessing elsewhere."""
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def make_dataloaders(
    data_dir: str | Path,
    *,
    image_size: int,
    batch_size: int,
    num_workers: int = 0,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    download: bool = True,
) -> DataBundle:
    """Download EuroSAT if needed and build train/validation/test loaders."""
    base_dataset = EuroSAT(root=Path(data_dir), download=download)
    train_indices, val_indices, test_indices = stratified_split_indices(
        base_dataset.targets,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    train_transform, eval_transform = build_transforms(image_size)

    train_dataset = TransformSubset(base_dataset, train_indices, train_transform)
    val_dataset = TransformSubset(base_dataset, val_indices, eval_transform)
    test_dataset = TransformSubset(base_dataset, test_indices, eval_transform)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    generator = torch.Generator().manual_seed(seed)

    return DataBundle(
        train_loader=DataLoader(
            train_dataset,
            shuffle=True,
            generator=generator,
            **loader_options,
        ),
        val_loader=DataLoader(val_dataset, shuffle=False, **loader_options),
        test_loader=DataLoader(test_dataset, shuffle=False, **loader_options),
        class_names=list(base_dataset.classes),
    )
