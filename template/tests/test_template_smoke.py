"""Smoke tests for the reusable OOP template (CPU-only)."""

from __future__ import annotations

import torch

from dl_template import __version__
from dl_template.data import TinyClassificationDataset, build_dataloader
from dl_template.eval import accuracy_from_logits
from dl_template.models import TinyCNN
from dl_template.training import Trainer


def test_package_imports() -> None:
    assert isinstance(__version__, str)
    assert __version__


def test_dataset_shapes() -> None:
    dataset = TinyClassificationDataset(num_samples=16, num_classes=4, image_size=8, seed=1)
    image, label = dataset[0]
    assert image.shape == (1, 8, 8)
    assert label.ndim == 0
    assert 0 <= int(label) < 4
    assert len(dataset) == 16


def test_model_forward_shape() -> None:
    model = TinyCNN(num_classes=5)
    batch = torch.randn(4, 1, 8, 8)
    logits = model(batch)
    assert logits.shape == (4, 5)


def test_one_training_step_on_cpu() -> None:
    dataset = TinyClassificationDataset(num_samples=24, num_classes=3, seed=2)
    loader = build_dataloader(dataset, batch_size=8, shuffle=False)
    model = TinyCNN(num_classes=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model=model, optimizer=optimizer, device=torch.device("cpu"))

    images, labels = next(iter(loader))
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    result = trainer.train_step(images, labels)

    assert result.loss >= 0.0
    assert 0.0 <= result.accuracy <= 1.0
    # At least one parameter should move after a real backward/step.
    moved = any(
        not torch.allclose(before[name], param.detach())
        for name, param in model.named_parameters()
    )
    assert moved


def test_fit_and_evaluate_smoke() -> None:
    dataset = TinyClassificationDataset(num_samples=32, num_classes=3, seed=3)
    loader = build_dataloader(dataset, batch_size=8, shuffle=True)
    model = TinyCNN(num_classes=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model=model, optimizer=optimizer, device="cpu")

    history = trainer.fit(loader, epochs=1)
    assert len(history) == 1
    eval_result = trainer.evaluate(loader)
    assert eval_result.loss >= 0.0
    assert 0.0 <= eval_result.accuracy <= 1.0


def test_accuracy_helper() -> None:
    logits = torch.tensor([[2.0, 0.1], [0.2, 3.0]])
    labels = torch.tensor([0, 1])
    assert accuracy_from_logits(logits, labels) == 1.0
