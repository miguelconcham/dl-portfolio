"""Minimal in-memory dataset used by the template smoke tests.

Replace this with a real `torch.utils.data.Dataset` for MNIST, CIFAR-10, etc.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset


class TinyClassificationDataset(Dataset):
    """Synthetic (image, label) pairs for wiring up training without downloads.

    Each sample is a random float tensor shaped like a tiny grayscale image.
    Enough to prove Dataset → Model → Trainer plumbing on CPU.
    """

    def __init__(
        self,
        num_samples: int = 32,
        num_classes: int = 3,
        image_size: int = 8,
        seed: int = 0,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")

        generator = torch.Generator().manual_seed(seed)
        self.images = torch.randn(
            num_samples,
            1,
            image_size,
            image_size,
            generator=generator,
        )
        self.labels = torch.randint(
            0,
            num_classes,
            (num_samples,),
            generator=generator,
        )
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[index]
        return image, label


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader with Windows-friendly defaults (`num_workers=0`)."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
