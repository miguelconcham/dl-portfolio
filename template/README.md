# OOP project template

Copy this folder when you start a new portfolio project:

```text
projects/0N-short-name/
```

Then rename the Python package (`dl_template` → something project-specific), fill in real Dataset / Model / Trainer logic, and keep notebooks as demos only.

## Layout

```text
template/
  README.md
  configs/          # hyperparams (YAML/JSON) — optional early
  src/dl_template/
    data/           # Dataset classes, transforms, loaders
    models/         # nn.Module subclasses
    training/       # Trainer / Engine
    eval/           # metrics, plots
    cli.py          # entrypoint stub
  tests/            # pytest: shapes + one train step
  notebooks/        # demos only — import from src/
  scripts/          # thin wrappers if needed
  artifacts/        # gitignored checkpoints / local plots
```

## Smoke run (from repo root, env activated)

```bash
pytest template/tests -q
python -m dl_template.cli --epochs 1
```

`PYTHONPATH` is set via `pyproject.toml` for pytest. For the CLI module path, either install editable (`pip install -e .`) or:

```bash
# Anaconda Prompt / PowerShell from repo root
set PYTHONPATH=template\src
python -m dl_template.cli --epochs 1
```

On bash / macOS / Linux:

```bash
PYTHONPATH=template/src python -m dl_template.cli --epochs 1
```

## Notebook policy

- Notebooks under `notebooks/` are for demos, plots, and exploration.
- If logic only exists in a notebook, it is **not done** — promote it into a class or function under `src/`.
