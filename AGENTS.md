# Project instructions

## Build & install
- Install Python dependencies: `make requirements` (installs from `requirements.txt`).
- Install the editable package: `pip install -e .` (already covered by the requirements step).

## Data pipeline
- Extract raw ZIP archives: `python src/data/extract.py` (populates `data/interim/`).
- Create a random sampled interim dataset (up to 100 entries per archive): `python src/data/create_interim.py`.
- Note: the Makefile `make data` target currently runs a stub; use the scripts directly.

## Development workflow
- Lint code: `make lint` (uses `flake8`).
- Run notebooks manually for exploration and preprocessing (`notebooks/*.ipynb`).
- No training or inference command exists yet; the repository is a scaffold for a TB chest‑X‑ray classifier.

## Conventions
- Keep public API changes backward compatible.
- Never edit any generated files under `data/interim/` or `data/processed/`.
- Ensure deterministic randomness: set a seed before using `random` or NumPy RNG in scripts.
- Add unit tests for any new functionality before merging.
- Follow the code style enforced by `flake8` (PEP‑8).

## Testing
- Currently there are no test suites; create tests under a `tests/` directory and run them with `pytest`.

## Documentation
- Update `architecture.md` when the pipeline or model implementation changes.
