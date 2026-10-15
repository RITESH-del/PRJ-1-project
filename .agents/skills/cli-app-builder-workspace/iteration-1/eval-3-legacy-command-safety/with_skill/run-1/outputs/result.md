# Legacy command safety (with `cli-app-builder`)

## Files changed

In the isolated worktree `/tmp/cli-app-eval-legacy-with-skill`:

- `src/tb_detection_model/__init__.py` — exports the installed `main` entrypoint.
- `src/tb_detection_model/cli.py` — Typer root app, `data` and `config` command groups, TOML loading/precedence, path safety, and explicit incomplete-command failure.
- `src/data/extract.py` — preserved legacy extraction API while adding configurable paths and safe ZIP-member destinations.
- `src/data/create_interim.py` — preserved legacy sampling API while adding configurable paths, deterministic seed support, and safe ZIP-member destinations.
- `pyproject.toml` — added the `typer>=0.16` runtime dependency and retained the declared console entrypoint.
- `uv.lock` — refreshed for Typer and its transitive dependencies.
- `AGENTS.md` — updated agent-facing installation, command inventory, config contract, legacy behavior, and QA commands.
- `CONTRIBUTING.md` — updated contributor-facing command walkthrough and unsupported validation behavior.

No raw, interim, or processed data was added to the worktree or repository.

## Command/config contract

The installed command is `tb-detection-model`:

```text
tb-detection-model data build
tb-detection-model data extract       # LEGACY
tb-detection-model data interim        # LEGACY
tb-detection-model data validate       # INCOMPLETE/UNSUPPORTED
tb-detection-model config init
tb-detection-model config show
tb-detection-model config validate
```

`data build` calls the existing `build_dataset` implementation. `data extract`
and `data interim` call the existing legacy implementations through Python
imports; they do not shell out to source scripts. Legacy commands default to
`data/interim` when no config or output override is supplied. `data validate`
exits with status 2 and says that `src/data/validate_processed.py` references
undefined configuration/helpers; it does not claim to validate anything.

The optional root-level `tb-detection.toml` uses:

```toml
[dataset]
input_path = "data/raw"
output_path = "data/processed"
seed = 42
```

Precedence is built-in defaults, the default `./tb-detection.toml` when
present, an explicitly supplied `--config PATH`, then explicit command-line
options. `config init` refuses overwrite unless `--force` is supplied.
`data build` and legacy commands reject an output path equal to or below the
raw input directory. Preprocessing constants, split ratios, dimensions, and
manifest schema remain fixed in the existing builder because its API does not
expose them as settings; they are intentionally unsupported by this adapter.

## Smoke verification

Executed in the isolated worktree with a temporary ZIP fixture under
`/tmp/tb-cli-smoke-legacy`:

- `uv sync` (installed the project and Typer entrypoint).
- `tb-detection-model --help`, `data --help`, and `config --help` (all command
groups and commands were discoverable).
- `tb-detection-model --version` (status 0 and version output).
- `config init`, `config show --json`, `config validate`, and a second `config
init` without `--force` (status 2, overwrite refusal).
- `data build --input ... --output ... --seed 7` (wrote `manifest.csv` and
processed output; the raw archive remained unchanged during the command).
- `data extract --input ... --output ...` (wrote interim extracted files).
- `data interim --input ... --output ... --seed 42 --sample-size 1` (wrote
interim sampled output).
- `data validate` (status 2 with explicit incomplete/unsupported explanation).
- An output nested under raw input was rejected with status 2 and an immutable
raw-data error.

## Deferred checks

Repository-wide pytest and lint were intentionally deferred to the parent
agent per the isolated evaluation protocol. The report does not claim those
checks passed.
