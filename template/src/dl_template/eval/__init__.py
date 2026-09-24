"""Evaluation helpers (metrics / plots live here in real projects)."""

from __future__ import annotations

import torch


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy from logits and integer labels."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape (N, C)")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must have shape (N,) matching logits")
    predictions = logits.argmax(dim=1)
    return float((predictions == labels).float().mean().item())
