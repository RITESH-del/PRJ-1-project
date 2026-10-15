# Full CLI conversion (without skill)

## Files changed

- `src/tb_detection_model/cli.py`: Typer app with `run`, `build-dataset`/`make-dataset`, `extract`, `interim`/`create-interim`, and `validate` commands.
- `src/tb_detection_model/config.py`: TOML loading and defaults/config/CLI precedence resolution.
- `src/tb_detection_model/__init__.py`: repaired `tb-detection-model` console entry point.
- `src/data/make_dataset.py`: configurable archive selection/label assignments, split ratios, and preprocessing settings while retaining existing processing functions.
- `src/data/extract.py`, `src/data/create_interim.py`: configurable paths, archives, sampling seed, and deterministic selection.
- `src/data/validate_processed.py`: repaired undefined configuration/helpers and exposed configurable validation/report paths.
- `tests/test_cli.py`: CLI discovery and TOML/CLI precedence coverage.
- `tb_detection.toml`: root-level default configuration example.
- `pyproject.toml`, `requirements.txt`: Typer runtime dependency.
- `AGENTS.md`, `CONTRIBUTING.md`: copy-pasteable CLI/config documentation.

## Implemented contract

The default precedence is built-in defaults < `tb_detection.toml` (or `--config PATH`) < explicit command-line options. Paths, seed, repeatable archive selection, `ARCHIVE=LABEL` assignments, split ratios, target size, normalization, image validity thresholds, ROI fraction, median blur kernel, and CLAHE clip limit/grid are configurable. The `run` command is the recommended direct ZIP-to-processed path; `extract` and `interim` remain optional legacy utilities.

## Explicitly unsupported

- Archive assignment values must be `TB` or `Normal` to participate in the existing stratified splitter; arbitrary labels are not a supported dataset contract.
- The CLI does not add model training, inference, cloud storage, or an extraction-to-builder intermediate handoff; the existing builder reads raw ZIP archives directly.
- ZIP extraction retains the legacy `extractall` behavior and is not a substitute for the builder's image validation.

## Verification

- Attempted targeted Python syntax compilation twice with `python -m py_compile` for all modified Python modules and `tests/test_cli.py`; the final check completed successfully.
- Full pytest, lint, package build, and installed-console smoke verification were intentionally deferred to the parent agent because this delegated run must not perform repository-wide verification and the isolated environment does not include Typer by default.
