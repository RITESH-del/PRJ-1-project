# Contributing to TB Detection Model

This guide covers the supported dataset workflow. `AGENTS.md` contains the
agent-facing repository contract; the installed CLI `--help` output is the
authoritative command reference.

## Installation and environment

From the repository root:

```bash
uv sync
```

The project requires Python 3.12 or newer. Use `.venv/bin/python` and
`.venv/bin/tb-detection-model` for repository commands.

## CLI discovery

Start with the root and grouped help pages:

```bash
  source .venv/bin/activate
  tb-detection-model --help
  tb-detection-model data --help
  tb-detection-model config --help
```

The root command also reports its version:

```bash
.venv/bin/tb-detection-model --version
```

## Recommended dataset build

The supported production path reads raw ZIP archives and writes deterministic,
patient-grouped, preprocessed PNGs:

```bash
.venv/bin/tb-detection-model data build
```

The output is `data/processed/{train,val,test}/{TB,Normal}/*.png` plus
`data/processed/manifest.csv`. Raw archives are read-only.

For source-segmented evaluation:

```bash
.venv/bin/tb-detection-model data build \
  --seed 42 \
  --train-archive nih.zip \
  --train-archive mont_139.zip \
  --validation-archive shenzen.zip \
  --test-archive shenzen.zip
```

All three archive groups are required when archive selection is used. An
archive listed for validation and test is split by patient group between those
outputs. Unlisted ZIP files are ignored.

## Configuration and override precedence

`tb-detection.toml` is the checked-in root-level example. Create a starter at
another path with:

```bash
.venv/bin/tb-detection-model config init --path /tmp/tb-experiment.toml
```

Its sections and keys are:

- `[dataset]`: `input_dir`, `output_dir`, `seed`, `target_size`,
  `split_ratios`, `train_archives`, `validation_archives`, `test_archives`,
  `manifest_filename`, and `archive_glob`;
- `[preprocessing]`: `minimum_image_size`, `minimum_image_mean`, ROI fraction
  ranges, `median_blur_kernel`, and CLAHE settings;
- `[logging]`: `level`, `scan_log_interval`, and `process_log_interval`.

Relative TOML paths are resolved from the TOML file directory. Effective
settings are applied in this order:

1. typed built-in defaults;
2. root `tb-detection.toml`, when present;
3. an explicitly supplied `--config PATH`;
4. explicit command-line options.

For example, CLI options take priority over both TOML layers:

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

Inspect or validate the effective configuration without processing images:

```bash
.venv/bin/tb-detection-model config show --json
.venv/bin/tb-detection-model config validate
```

`config init` refuses to overwrite an existing file unless `--force` is
provided. Validation rejects invalid dimensions, split ratios that are not
positive or do not sum to one, unsafe/equal input and output paths, incomplete
archive assignments, invalid preprocessing ranges, and unknown log levels.

## Validation and legacy commands

The command inventory is:

```text
tb-detection-model data build       recommended raw-ZIP-to-PNG workflow
tb-detection-model data validate    explicit unsupported failure
tb-detection-model data extract     legacy extraction compatibility command
tb-detection-model data interim     legacy sampling compatibility command
tb-detection-model config init
tb-detection-model config show
tb-detection-model config validate
```

`data validate` fails explicitly because `src/data/validate_processed.py`
currently references undefined helpers. It does not claim to validate data.
The legacy commands remain available for compatibility, but they are not part
of the current raw-ZIP-to-processed-dataset path.

## Tests and lint

Run the focused and regression suites from the repository root:

```bash
.venv/bin/python -m pytest -q
make lint
.venv/bin/python -m flake8 src tests
```

Use temporary ZIP fixtures for CLI smoke tests. Never commit generated
`data/interim/` or `data/processed/` content, and never edit `data/raw/`.
