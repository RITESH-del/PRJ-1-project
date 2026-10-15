project_name
==============================

A short description of the project.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


Data preprocessing
-------------------

Install and discover the supported command interface:

```bash
uv sync
.venv/bin/tb-detection-model --help
```

Build the processed dataset with the recommended workflow:

```bash
.venv/bin/tb-detection-model data build
```

For source-segmented evaluation:

```bash
.venv/bin/tb-detection-model data build \
  --seed 42 \
  --train-archive nih.zip \
  --train-archive mont_139.zip \
  --validation-archive shenzen.zip \
  --test-archive shenzen.zip
```

The checked-in `tb-detection.toml` configures paths, seed, archive
assignments, split ratios, target size, preprocessing thresholds/options,
manifest output, and logging. Relative paths are resolved from the TOML file.
Effective precedence is built-in defaults, root `tb-detection.toml`, explicit
`--config PATH`, then explicit CLI options:

```bash
.venv/bin/tb-detection-model config show --json
.venv/bin/tb-detection-model config validate
.venv/bin/tb-detection-model data build \
  --input-dir /tmp/tb/raw \
  --output-dir /tmp/tb/processed \
  --target-size 192 192 \
  --seed 17 \
  --log-level WARNING
```

`config init` creates a documented starter file and refuses to overwrite one
unless `--force` is supplied:

```bash
.venv/bin/tb-detection-model config init --path /tmp/tb-detection.toml
```

The legacy `data extract` and `data interim` commands remain available for
compatibility. `data validate` is discoverable but intentionally fails with an
unsupported message because the existing processed-data validator is
incomplete. See `AGENTS.md` and `CONTRIBUTING.md` for the full command and
configuration contract.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
