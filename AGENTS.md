# Repository Guidelines

## Project Overview

TB Detection Model is a Python project for preparing tuberculosis chest-X-ray data. The implemented path is raw ZIP archives to patient-grouped, preprocessed PNGs. Feature extraction, model training, prediction, and visualization modules are currently placeholders.

Treat `data/raw/` as immutable. Generated `data/interim/` and `data/processed/` content is not source code and should not be edited manually or committed.

## Architecture & Data Flow

Current production path:

```text
data/raw/*.zip
    -> src/data/make_dataset.py
    -> data/processed/{train,val,test}/{TB,Normal}/*.png
    -> data/processed/manifest.csv
```

`make_dataset.py` reads labelled images from ZIP archives, validates and deduplicates them, groups images by patient, assigns deterministic stratified splits, applies grayscale/ROI/resize-padding/median-blur/CLAHE preprocessing, and writes 224x224 uint8 PNGs. Model loading must apply the notebook's fixed mean/std normalization through `normalize_xray()`.

`extract.py` and `create_interim.py` are separate legacy utilities for extracting or sampling archives into `data/interim/`; they are not inputs to the current dataset builder. There is no implemented training or inference data flow.

## Key Directories

- `src/data/` — archive extraction, interim sampling, final dataset generation, and intended validation utilities.
- `src/features/` — feature-engineering placeholder.
- `src/models/` — training and prediction placeholders.
- `src/visualization/` — visualization placeholder.
- `data/raw/` — immutable source ZIP archives.
- `data/interim/` — optional extracted/sampled working data.
- `data/processed/` — generated train/validation/test PNGs and manifest.
- `notebooks/` — exploratory and preprocessing references; use numbered names for workflow order.
- `tests/` — pytest tests using temporary ZIP fixtures.
- `docs/` — Sphinx documentation scaffold.

## Development Commands

Use the repository root as the working directory.

```bash
# Install/synchronize the modern uv environment
uv sync

# Build the processed dataset; seed is deterministic
.venv/bin/python src/data/make_dataset.py data/raw data/processed --seed 42

# Run tests; there is no Makefile test target
.venv/bin/python -m pytest -q

# Lint source
make lint
# Equivalent targeted check when working on the dataset builder:
.venv/bin/python -m flake8 src/data/make_dataset.py tests/test_make_dataset.py

# Check the Python environment
make test_environment

# Build Sphinx docs
make -C docs html
```

Legacy workflow commands remain available:

```bash
make requirements
python src/data/extract.py
python src/data/create_interim.py
```

Do not use `extract.py` or `create_interim.py` when the goal is the current raw-ZIP-to-processed-dataset pipeline.

## Code Conventions & Common Patterns

- Python, synchronous functions, `pathlib.Path`, small typed helpers, and dataclasses for structured records.
- Follow PEP 8. `tox.ini` sets a 79-character line limit and McCabe complexity limit of 10; `make lint` runs flake8 on `src/`.
- Use deterministic ordering/seeds for dataset operations. Preserve patient grouping and label stratification when changing split logic.
- Preserve exact processed label directory names: `TB` and `Normal`.
- Prefer `logging` in new production code. Older extraction/sampling utilities still use `print` and may be modernized only when in scope.
- No async framework, dependency-injection container, or state-management framework is used.
- Keep generated data out of source changes. Add or update tests with behavior changes, and update README/docs when commands or data contracts change.
- Avoid adding model-training abstractions until a real model/data-loader contract exists.

## Important Files

- `src/data/make_dataset.py` — implemented dataset-builder CLI and reusable preprocessing/splitting functions.
- `tests/test_make_dataset.py` — current regression coverage for preprocessing, reproducibility, and patient isolation.
- `notebooks/02_data_preprocessing.ipynb` — preprocessing reference and fixed normalization statistics.
- `pyproject.toml` — canonical package metadata, Python requirement, runtime dependencies, and `uv_build` backend.
- `uv.lock` — checked-in uv resolution; update it when `pyproject.toml` dependencies change.
- `requirements.txt` — legacy pip/development dependency list used by `make requirements`; it is not identical to `pyproject.toml`.
- `Makefile` — legacy automation for requirements, data generation, linting, environment checks, cleanup, and S3 sync.
- `tox.ini` — flake8 settings only; it does not define tox environments or test commands.
- `test_environment.py` — checks only that the interpreter is Python 3.

Known incomplete paths:

- `src/data/validate_processed.py` is intended to validate processed data but currently references undefined configuration/helpers and is not wired into the Makefile.
- `src/tb_detection_model/__init__.py` is empty even though `pyproject.toml` declares `tb-detection-model = "tb_detection_model:main"`; do not assume that console entry point works.
- `architecture.md` and parts of `CONTRIBUTING.md` still describe the dataset builder as a stub; prefer current source and README behavior.

## Runtime/Tooling Preferences

- Local Python selection is pinned to 3.13 by `.python-version`; `pyproject.toml` requires Python 3.12 or newer.
- Prefer `uv` and the checked-in `uv.lock` for environment/package synchronization. `uv_build` is the configured build backend.
- Use the project environment (`.venv/bin/python`) for commands after `uv sync`.
- OpenCV and NumPy are required for preprocessing. PyTorch, TensorFlow, CUDA, and GPU training are not currently part of the project dependencies.
- The repository includes unrelated Node metadata; do not introduce Node tooling for Python/data work.

## Testing & QA

- Test framework: pytest. Run from the repository root with `.venv/bin/python -m pytest -q`.
- Current tests cover preprocessing output shape/dtype/normalization and deterministic patient-grouped dataset generation from temporary ZIP fixtures.
- Run `make lint` or targeted flake8 before delivery. Tests and lint are separate; there is no CI workflow or integrated Makefile test target.
- `coverage` is listed as a dependency, but no coverage command or minimum threshold is configured. Use coverage when measuring new behavior, not as a substitute for behavioral tests.
- For dataset changes, verify the generated manifest, PNG dimensions/dtype, per-split labels, and that no patient occurs in multiple splits.
- Before expensive processing, use a small representative ZIP fixture or subset, then run the full dataset command only after the smoke result is valid.
