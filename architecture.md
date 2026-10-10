# Project Architecture

## Project overview

The repository is a Cookiecutter Data Science scaffold for TB chest‑X‑ray exploration. It contains notebooks for data inspection and preprocessing, but **no functional classifier, training loop, or inference script** has been implemented yet.

**Current status:** Source modules `features/build_features.py`, `models/train_model.py`, `models/predict_model.py`, and `visualization/visualize.py` are empty stubs. The Makefile `data` target runs a placeholder script that only logs a message. Consequently, there is **no end‑to‑end pipeline** from raw data to a trained model.

## Repository structure

```text
.
├── data/                     # Data directories; raw contents intentionally not inspected
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/                     # Sphinx scaffold
├── models/                   # Intended model artifacts; no training implementation found
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_data_annotation.ipynb
├── reports/figures/
├── src/
│   ├── data/
│   │   ├── create_interim.py
│   │   ├── extract.py
│   │   ├── make_dataset.py
│   │   └── validate_processed.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── predict_model.py
│   │   └── train_model.py
│   ├── tb_detection_model/
│   │   └── __init__.py
│   └── visualization/
│       └── visualize.py
├── Makefile
├── requirements.txt
├── setup.py
└── test_environment.py
```

## Data pipeline

Two utilities exist:

* `src/data/extract.py` – extracts every `*.zip` in `data/raw/` to `data/interim/<archive‑stem>/`.
* `src/data/create_interim.py` – randomly samples up to 100 entries per archive into the same interim location. Randomness is uncontrolled (no fixed seed).
* `src/data/validate_processed.py` – placeholder script intended to verify files in `data/processed/` (currently does nothing).

Both scripts only populate `data/interim/`; no labeling, splitting, or processing to `data/processed/` occurs. The Makefile `make data` target runs `src/data/make_dataset.py`, which is a stub that logs a message.

## Preprocessing (notebook‑only)

`notebooks/02_data_preprocessing.ipynb` performs grayscale conversion, central cropping, resizing to 224×224, median blur, CLAHE, and normalization using hard‑coded mean/std values. No source‑code function is invoked by any pipeline.

## Dataset / DataLoader

**Not implemented.** No `torch.utils.data.Dataset`, `tf.data.Dataset`, or similar abstraction exists.

## Model architecture

**Not implemented.** `src/models/train_model.py` and `src/models/predict_model.py` are empty placeholders. No deep‑learning framework is listed in `requirements.txt`.

## Training pipeline

**Absent.** No training script, CLI, or Makefile target exists. Implementers must add a training module and integrate it with the data pipeline.

## Validation and evaluation

No evaluation code exists. Notebook validation only checks image quality, not model performance.

## Configuration

No dedicated configuration file for model or training parameters. The Makefile defines generic variables; notebooks hard‑code paths and constants.

## Dependencies

`requirements.txt` lists only generic tooling (click, Sphinx, coverage, awscli, flake8, python‑dotenv). Notebook‑specific packages (numpy, matplotlib, pillow, opencv‑python, torch/tensorflow) are missing.
## Package manager

The project builds its distribution with **uv** (`uv_build` backend declared in `pyproject.toml`). Runtime installation uses `pip install -e .` after dependencies are installed via `make requirements`.
* `python src/data/create_interim.py` – sample interim data.
* `make data` – stub script; does not create a dataset.
* `make requirements` – install `requirements.txt`.
* `make lint` – run flake8 on `src/`.

Notebook execution is manual; no end‑to‑end command runs extraction, preprocessing, training, evaluation, or inference.

## Hardware and runtime

Only Python 3 is required; no CPU/GPU or CUDA specifications are declared because no model code exists.

## Known issues / TODOs

* Implement real dataset building and splitting.
* Add deterministic randomness (fixed seed).
* Move preprocessing steps into reusable source code.
* Implement Dataset/DataLoader, model, training entry point, checkpointing, prediction, and evaluation metrics.
* Add deep‑learning dependencies with version constraints.
* Keep documentation (`architecture.md`, `AGENTS.md`) up‑to‑date.
* Notebook contents for `01_data_exploration.ipynb` and `03_data_annotation.ipynb` could not be parsed; verify separately.
