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

The default CLI reads every raw ZIP and creates deterministic patient-grouped,
stratified splits. For source-segmented evaluation, explicit archive options
keep training archives separate from evaluation archives:

```bash
.venv/bin/python src/data/make_dataset.py data/raw data/processed \
  --train-archives nih.zip mont_139.zip \
  --validation-archives shenzen.zip \
  --test-archives shenzen.zip
```

All three archive options are required together, and every named archive must
contain valid labelled images. Unlisted ZIP files are ignored. An archive
listed for both validation and test is split by patient group between those
two outputs. An archive listed for only one split remains entirely in that
split.

`make_dataset.py` reads labelled images from ZIP archives, validates and
deduplicates them, groups images by patient, assigns deterministic splits,
applies grayscale/ROI/resize-padding/median-blur/CLAHE preprocessing, and
writes 224x224 uint8 PNGs. Model loading must apply the notebook's fixed
mean/std normalization through `normalize_xray()`.

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

## CLI and configuration

Install and synchronize the environment:

```bash
uv sync
```

The installed command is the authoritative interface. Start with discovery:

```bash
.venv/bin/tb-detection-model --help
.venv/bin/tb-detection-model data --help
.venv/bin/tb-detection-model config --help
```

Recommended build:

```bash
.venv/bin/tb-detection-model data build
```

Source-segmented evaluation:

```bash
.venv/bin/tb-detection-model data build \
  --seed 42 \
  --train-archive nih.zip \
  --train-archive mont_139.zip \
  --validation-archive shenzen.zip \
  --test-archive shenzen.zip
```

The root-level `tb-detection.toml` is the checked-in experiment example.
Its documented keys cover paths, seed, archive selection, split ratios,
target size, preprocessing thresholds/options, manifest filename, and
logging. Relative TOML paths are resolved from the TOML file directory.
Configuration precedence is:

1. typed built-in defaults;
2. `tb-detection.toml`, when present;
3. an explicit `--config PATH`;
4. explicit command options.

For example, these command options override the corresponding TOML keys:

```bash
.venv/bin/tb-detection-model data build \
  --config tb-detection.toml \
  --input-dir /tmp/tb/raw \
  --output-dir /tmp/tb/processed \
  --seed 17 \
  --target-size 192 192 \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --log-level WARNING
```

Configuration commands:

```bash
.venv/bin/tb-detection-model config init
.venv/bin/tb-detection-model config show --json
.venv/bin/tb-detection-model config validate
```

`config init` refuses to overwrite an existing file unless `--force` is
supplied. `config validate` checks cross-field values before processing:
positive dimensions, positive ratios summing to 1, valid paths, complete
archive assignments, valid preprocessing ranges, and a recognized logging
level.

The complete command inventory is:

```text
tb-detection-model data build       recommended raw-ZIP-to-PNG workflow
tb-detection-model data validate    explicit failure: validator is incomplete
tb-detection-model data extract     legacy extraction compatibility command
tb-detection-model data interim     legacy sampling compatibility command
tb-detection-model config init
tb-detection-model config show
tb-detection-model config validate
```

Do not expose model training, prediction, or visualization placeholders as
working commands. `data validate` is intentionally discoverable but fails
with an actionable unsupported message because
`src/data/validate_processed.py` still references undefined helpers.

## Development and QA commands

```bash
# Run tests
.venv/bin/python -m pytest -q

# Lint source
make lint
.venv/bin/python -m flake8 src tests

# Check the Python environment
make test_environment

# Build Sphinx docs
make -C docs html
```

Legacy scripts remain available when compatibility work requires them:

```bash
python src/data/extract.py
python src/data/create_interim.py
```

Do not use legacy extraction or interim sampling for the current processed
dataset pipeline.

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

- `src/tb_detection_model/cli.py` — Typer command tree and thin workflow adapters.
- `src/tb_detection_model/config.py` — typed TOML loading, precedence, and
  boundary validation.
- `src/tb_detection_model/data/make_dataset.py` — dataset-builder source of
  truth; `src/data/make_dataset.py` preserves the legacy script path.
- `tests/test_make_dataset.py` and `tests/test_cli.py` — dataset and CLI
  regression coverage using temporary ZIP fixtures.
- `tb-detection.toml` — checked-in starter configuration.
- `notebooks/02_data_preprocessing.ipynb` — preprocessing reference and fixed normalization statistics.
- `pyproject.toml` — canonical package metadata, CLI entry point, dependencies,
  and `uv_build` backend.
- `uv.lock` — checked-in uv resolution; update it when `pyproject.toml` dependencies change.
- `requirements.txt` — legacy pip/development dependency list used by `make requirements`; it is not identical to `pyproject.toml`.
- `Makefile` — legacy automation for data generation, linting, environment checks, cleanup, and S3 sync.
- `tox.ini` — flake8 settings only; it does not define tox environments or test commands.
- `test_environment.py` — checks only that the interpreter is Python 3.

Known incomplete paths:

- `src/data/validate_processed.py` is intended to validate processed data but
  currently references undefined configuration/helpers. The CLI exposes this
  as an explicit unsupported command rather than claiming validation.
- `architecture.md` still contains historical scaffold descriptions; prefer
  current source, README, and this guide for dataset-builder behavior.

## Runtime/Tooling Preferences

- Local Python selection is pinned to 3.13 by `.python-version`; `pyproject.toml` requires Python 3.12 or newer.
- Prefer `uv` and the checked-in `uv.lock` for environment/package synchronization. `uv_build` is the configured build backend.
- Use the project environment (`.venv/bin/python`) for commands after `uv sync`.
- OpenCV and NumPy are required for preprocessing. PyTorch, TensorFlow, CUDA, and GPU training are not currently part of the project dependencies.
- The repository includes unrelated Node metadata; do not introduce Node tooling for Python/data work.

## Testing & QA

- Test framework: pytest. Run from the repository root with `.venv/bin/python
  -m pytest -q`.
- Current tests cover preprocessing output shape/dtype/normalization,
  deterministic patient-grouped dataset generation, source-specific archive
  assignment, and the CLI path.
- Run `make lint` or targeted flake8 before delivery. Tests and lint are
  separate; there is no CI workflow or integrated Makefile test target.
- `coverage` is listed as a dependency, but no coverage command or minimum
  threshold is configured. Use coverage when measuring new behavior, not as a
  substitute for behavioral tests.
- For dataset changes, verify the generated manifest, PNG dimensions/dtype,
  per-split labels, archive provenance, and that no patient occurs in multiple
  splits.
- Before expensive processing, use a small representative ZIP fixture or
  subset, then run the full dataset command only after the smoke result is
  valid.

