"""CLI stub for the template.

Real projects replace this with `train` / `evaluate` commands that load configs
and call Dataset / Model / Trainer classes under `src/`.
"""

from __future__ import annotations

import argparse

import torch

from dl_template.data import TinyClassificationDataset, build_dataloader
from dl_template.models import TinyCNN
from dl_template.training import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="dl_template smoke CLI — proves Dataset → Model → Trainer on CPU.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of tiny training epochs (default: 1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for the synthetic loader (default: 8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for reproducible smoke runs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)

    dataset = TinyClassificationDataset(num_samples=32, num_classes=3, seed=args.seed)
    loader = build_dataloader(dataset, batch_size=args.batch_size, shuffle=True)
    model = TinyCNN(num_classes=dataset.num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model=model, optimizer=optimizer, device="cpu")

    history = trainer.fit(loader, epochs=args.epochs)
    last = history[-1]
    print(
        f"smoke ok | epochs={args.epochs} "
        f"last_loss={last.loss:.4f} last_acc={last.accuracy:.3f} "
        f"cuda={torch.cuda.is_available()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
