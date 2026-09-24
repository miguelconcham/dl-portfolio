"""Minimal Trainer: one train step + a short fit loop for smoke tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TrainStepResult:
    """Metrics from a single optimization step."""

    loss: float
    accuracy: float


class Trainer:
    """Owns the train/eval step pattern every portfolio project should reuse.

    Real projects extend this (checkpointing, logging, schedulers) rather than
    rewriting loops inside notebooks.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if device is None:
            device = torch.device("cpu")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

    def train_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> TrainStepResult:
        """Run one forward/backward/optimizer step on a batch."""
        self.model.train()
        images = images.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(images)
        loss = self.criterion(logits, labels)
        loss.backward()
        self.optimizer.step()

        predictions = logits.argmax(dim=1)
        accuracy = (predictions == labels).float().mean().item()
        return TrainStepResult(loss=float(loss.item()), accuracy=float(accuracy))

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> TrainStepResult:
        """Average loss and accuracy over a loader (eval mode, no grad)."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_examples += batch_size

        if total_examples == 0:
            raise ValueError("evaluate() received an empty loader")

        return TrainStepResult(
            loss=total_loss / total_examples,
            accuracy=total_correct / total_examples,
        )

    def fit(self, loader: DataLoader, epochs: int = 1) -> list[TrainStepResult]:
        """Run a few epochs; returns the last train-step result per epoch."""
        if epochs < 1:
            raise ValueError("epochs must be >= 1")

        history: list[TrainStepResult] = []
        for _ in range(epochs):
            last: TrainStepResult | None = None
            for images, labels in loader:
                last = self.train_step(images, labels)
            if last is None:
                raise ValueError("fit() received an empty loader")
            history.append(last)
        return history
