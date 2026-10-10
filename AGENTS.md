# Repository Guidelines

## Project Overview
The **TB Detection Model** repository is a Cookiecutter Data Science scaffold for exploring, preprocessing, and eventually training a tuberculosis chest‑X‑ray classifier. Core utilities extract raw ZIP archives and sample a limited number of images; model, feature‑engineering, and visualization modules are empty placeholders.

## Architecture & Data Flow
```
data/
  raw/          # Immutable source ZIP archives
  interim/       # Extracted archives + sampled images
  processed/    # Intended final, cleaned dataset (currently empty)

src/
  data/         # extract.py → unzip raw → data/interim/
                # create_interim.py → random sample (≤100 per archive) → data/interim/
                # make_dataset.py (stub CLI)
                # validate_processed.py (partial PNG/manifest validation)
  features/     # build_features.py (placeholder)
  models/       # train_model.py / predict_model.py (placeholders)
  visualization/ # visualize.py (placeholder)
```
Current flow: `extract.py` → `create_interim.py`. Future steps should add preprocessing that writes to `data/processed/`, a dataset/dataloader, model training, and evaluation.

## Key Directories
| Directory | Purpose |
|-----------|---------|
| `src/` | Core Python package. |
| `src/data/` | Extraction, sampling, stub dataset creation, validation. |
| `src/features/` | Placeholder for feature‑engineering functions. |
| `src/models/` | Stubs for training and inference. |
| `src/visualization/` | Placeholder for reporting/plotting. |
| `data/` | Holds `raw/`, `interim/`, `processed/`. **Do not edit** generated files under `interim/` or `processed/`. |
| `docs/` | Sphinx docs (`conf.py`, `Makefile`). |
| `notebooks/` | Jupyter notebooks for manual exploration. |
| `tests/` | Not present yet; create for unit tests. |

## Development Commands
```bash
# Install dev dependencies
make requirements               # pip install -r requirements.txt

# Editable install for local development
pip install -e .

# Lint source (flake8)
make lint

# Run data pipeline utilities
python src/data/extract.py          # unzip raw → data/interim/
python src/data/create_interim.py   # sample ≤100 files per archive → data/interim/

# Verify Python environment (CI helper)
make test_environment

# Build documentation (Sphinx)
make -C docs html
```
`Makefile` also provides `clean`, `sync_data_to_s3`, `sync_data_from_s3`, and a stub `make data` target.

## Code Conventions & Common Patterns
- **PEP‑8** enforced via `flake8`. 
- Use **`pathlib.Path`** for filesystem paths. 
- Functions guarded by `if __name__ == "__main__"`. 
- **Randomness**: `random.sample` used without a fixed seed – add `random.seed(<int>)` for reproducibility. 
- Error handling currently uses `print`; replace with `logging` or raise exceptions for production code. 
- No async constructs, dependency injection, or state‑management frameworks. 
- Minimal type hints; adding them improves IDE support.

## Important Files
- `architecture.md` – high‑level scaffold description and status. 
- `pyproject.toml` – metadata, entry point (`tb-detection-model = "tb_detection_model:main"`), **uv** build backend. 
- `requirements.txt` – runtime deps (`click`, `Sphinx`, `coverage`, `awscli`, `flake8`, `python-dotenv`). 
- `Makefile` – dev commands, env setup, S3 sync helpers. 
- `src/data/extract.py` & `src/data/create_interim.py` – core data‑pipeline utilities. 
- `src/models/train_model.py` & `src/models/predict_model.py` – empty placeholders. 
- `docs/conf.py` & `docs/Makefile` – Sphinx config and build script. 
- `test_environment.py` – sanity‑check script invoked by `make test_environment`. 
- `README.md` – project description and directory overview. 
- `AGENTS.md` – this guidelines document.

## Runtime / Tooling Preferences
- **Python ≥ 3.13** (declared in `pyproject.toml`).
- **uv** for building/distribution (`uv_build` backend).
- Editable install via `pip install -e .` after `make requirements`.
- Lint with **flake8** (`make lint`).
- Docs generated via **Sphinx** (`make -C docs html`).
- No Node/Bun runtime required.

## Testing & QA
- No test suite yet; create a `tests/` directory and use **pytest** for new functionality.
- Existing `test_environment.py` validates interpreter setup; run via `make test_environment`.
- After adding tests, use **coverage** (`coverage run -m pytest && coverage report`).
- Suggested test areas:
  * Data extraction & sampling correctness (deterministic seeds).
  * Validation logic in `src/data/validate_processed.py`.
  * Feature‑engineering utilities.
  * Model training pipeline and evaluation metrics.

---
*All sections reflect the repository state as of the latest analysis. Update this document when new modules, tests, or CI pipelines are added.*
