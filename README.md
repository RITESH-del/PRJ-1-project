# TB Detection Model

A Python research codebase for classifying chest X-rays as **TB** or **Normal**. It provides a command-line workflow for preparing patient-grouped datasets, training classifiers, evaluating checkpoints, and running single-image inference.

> **Research prototype only.** This project is not a medical device and must not be used for diagnosis or clinical decisions.

## What it includes

- ZIP-archive dataset preparation with patient-grouped train, validation, and test splits.
- Configurable X-ray preprocessing and a CSV manifest for generated samples.
- Training workflows for DINOv2 and DenseNet121, including fine-tuning and early stopping options.
- Evaluation reports and command-line inference for individual images.
- TOML configuration with defaults, validation, and explicit CLI overrides.

No model-performance claims are made here; review the checked-in notebooks and reports for experiment details.

## Setup

Requires Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
.venv/bin/tb-detection-model --help
```

## Prepare a dataset

The default configuration reads ZIP archives from `data/raw` and writes processed data to `data/processed`. Adjust `tb-detection.toml` or pass CLI options for different paths and split settings.

```bash
.venv/bin/tb-detection-model config show --json
.venv/bin/tb-detection-model config validate
.venv/bin/tb-detection-model data build
```

For source-segmented evaluation, assign archives explicitly:

```bash
.venv/bin/tb-detection-model data build \
  --seed 42 \
  --train-archive nih.zip \
  --train-archive mont_139.zip \
  --validation-archive shenzen.zip \
  --test-archive shenzen.zip
```

Archive names must match files in the configured input directory. When archive selection is used, provide training, validation, and test assignments. Relative paths in TOML files are resolved from the file’s directory.

## Train and evaluate

After building the processed dataset, inspect model options and start training:

```bash
.venv/bin/tb-detection-model model train --help
.venv/bin/tb-detection-model model train \
  --model-name densenet121 \
  --epochs 3 \
  --batch-size 64 \
  --eval-split val \
  --output-dir models/densenet121 \
  --report-dir reports/densenet121
```

Evaluate a checkpoint or classify one image:

```bash
.venv/bin/tb-detection-model model evaluate \
  --model-path models/densenet121 \
  --split test

.venv/bin/tb-detection-model model predict \
  --image path/to/chest-xray.png \
  --model-path models/densenet121 \
  --json
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full command and configuration reference, including model and dataset options.

## Known limitations

- `data validate` is currently unsupported and exits with an explicit error; the processed-data validator is incomplete.
- `data extract` and `data interim` are legacy compatibility commands. `data build` is the recommended dataset workflow.

## Project references

- [Architecture notes](architecture.md)
- [User manual](user-manual.pdf)
- [Dataset references](references/references.md)
- [Contributing and test instructions](CONTRIBUTING.md)
