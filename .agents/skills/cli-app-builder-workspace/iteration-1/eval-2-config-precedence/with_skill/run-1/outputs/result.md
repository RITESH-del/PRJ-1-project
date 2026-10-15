# Config-precedence implementation report

## Files changed

- `src/data/make_dataset.py`: accepts target image dimensions and split-ratio mappings while retaining the existing preprocessing implementation.
- `src/tb_detection_model/config.py`: typed `DatasetConfig`, TOML loading, defaults/config/CLI merge helper, JSON output, and boundary validation.
- `src/tb_detection_model/cli.py`, `src/tb_detection_model/__init__.py`: package CLI entry point with `data build` and `config init/show/validate`; legacy `data extract` and `data interim` wrappers; explicit unsupported `data validate` error.
- `tb-detection.toml`: checked-in example.
- `tests/test_config_cli.py`: precedence and invalid-ratio/dimension tests.
- `pyproject.toml`: Typer runtime dependency.
- `AGENTS.md`, `CONTRIBUTING.md`: synchronized CLI, configuration precedence, validation, and unsupported-command documentation.

## Contract

`tb-detection-model data build` uses typed defaults, then root `tb-detection.toml`, then an explicit `--config PATH`, then explicit build options. Supported build overrides are `--seed`, `--input-dir`, `--output-dir`, `--target-size HEIGHT WIDTH`, `--train-ratio`, `--val-ratio`, `--test-ratio`, and `--log-level`. Config keys are `[dataset]` (`input_dir`, `output_dir`, `seed`, `target_size`, nested `split_ratios`) and `[logging] level`.

`config init` refuses overwrite unless `--force`; `config show --json` emits effective values; `config validate` performs preflight validation. Dimensions must be positive integers, all three ratios must be strictly between 0 and 1 and sum to 1, and logging level must be recognized. `data validate` is intentionally unsupported because the existing validator is incomplete; it fails explicitly.

## Verification

Repository-wide tests, lint, and a real CLI fixture smoke run were intentionally deferred to the parent agent, as required for delegated work. Raw and generated data were not modified.
