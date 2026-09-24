# Deep Learning Portfolio — Miguel Miranda

A Windows-friendly **PyTorch** monorepo for learning CNNs with real software habits: packages, classes, tests, and reproducible Anaconda environments — not notebook dumps.

**Honest scope:** this is a learning portfolio. Training targets **CPU-only PyTorch** (no NVIDIA GPU on the author machine; Intel Arc is present but CUDA is not used for v1). Smaller models and shorter runs are intentional.

## What’s here (Phase 0)

| Piece | Purpose |
|-------|---------|
| [`template/`](template/) | Reusable OOP skeleton: `Dataset`, `Model`, `Trainer`, CLI stub, pytest smoke tests |
| [`projects/`](projects/) | Future numbered projects (MNIST → CIFAR → transfer learning → stretch) |
| [`environment.yml`](environment.yml) | Conda env: Python 3.11, **CPU-only** PyTorch + torchvision, numpy, pandas, matplotlib, pytest, pillow |
| [`docs/learning-log.md`](docs/learning-log.md) | Optional short learning notes |

Full roadmap (phases, daily cadence, CV bullets): see the planning docs in the Cursor project store / your overview plan.

## Quick start (Windows + Anaconda)

```bash
conda env create -f environment.yml
conda activate dl-portfolio
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pytest -q
```

Expect `torch.cuda.is_available()` → **`False`**. That is correct for the CPU build.

Optional CLI smoke (after setting `PYTHONPATH` or `pip install -e .`):

```bash
set PYTHONPATH=template\src
python -m dl_template.cli --epochs 1
```

## OOP pattern (every project)

- **Dataset** — loading, transforms, loaders  
- **Model** — `nn.Module` subclasses  
- **Trainer** — train step, eval, metrics hooks  
- **CLI** — `train` / `evaluate` entrypoints  
- **Notebooks** — demos only; real code under `src/`

## Project roadmap

| # | Project | Framework | Status |
|---|---------|-----------|--------|
| 01 | MNIST CNN (warm-up) | PyTorch | Planned |
| 02 | CIFAR-10 CNN (from-scratch) | PyTorch | Planned |
| 03 | Transfer learning classifier | PyTorch + torchvision | Planned |
| 04 | Detection/segmentation **or** PyTorch vs Keras | TBD | Planned |

## License / use

Personal portfolio code. Feel free to fork the template layout for your own learning projects.
