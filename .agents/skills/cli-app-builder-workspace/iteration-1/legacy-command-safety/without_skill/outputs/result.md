# Legacy command-safety run

## Files changed

- `pyproject.toml` — configures `uv_build` to package both `tb_detection_model` and the imported `data` module.
- `src/tb_detection_model/cli.py` — argparse command surface with `dataset`, legacy `extract`/`interim`, and explicit unsupported `validate`; direct Python imports only.
- `src/tb_detection_model/__init__.py` — exports the package `main` entry point declared by `pyproject.toml`.
- `src/tb_detection_model/__main__.py` — supports `python -m tb_detection_model`.
- `src/data/extract.py` — parameterized raw/interim paths and rejects unsafe archive traversal while streaming output.
- `src/data/create_interim.py` — parameterized raw/interim paths, sample size, and optional deterministic seed.
- `AGENTS.md`, `CONTRIBUTING.md`, `README.md`, `docs/commands.rst` — documented recommended, legacy, and unsupported command contracts.

## Command/config contract

- `tb-detection-model dataset [INPUT] [OUTPUT] [--input-dir PATH] [--output-dir PATH] [--seed N]` is the recommended raw-ZIP to processed-dataset workflow. Defaults are `data/raw` and `data/processed`; option paths override positional paths.
- `tb-detection-model extract ...` is explicitly **LEGACY** and extracts every raw ZIP into an interim output (default `data/interim`).
- `tb-detection-model interim ... [--sample-size N] [--seed N]` is explicitly **LEGACY** and samples archive members into interim output.
- `tb-detection-model validate [--input-dir PATH]` is discoverable but exits 2 with `UNSUPPORTED`; it performs no validation because `validate_processed.py` remains incomplete.
- `build-dataset`, `create-interim`, and `validate-processed` are accepted aliases. Every command and alias is available from `--help`.
- Raw archives are read only. Extraction rejects path-traversal members and all generated output is directed outside the raw directory.

## Explicit unsupported behavior

Validation is not implemented. The command does not import or execute the incomplete validation module and never reports success.

## Verification

Attempted:

- `PYTHONPATH=src python -m tb_detection_model --help`
- `PYTHONPATH=src python -m tb_detection_model dataset --help`
- `PYTHONPATH=src python -m tb_detection_model extract --help`
- `PYTHONPATH=src python -m tb_detection_model interim --help`
- `PYTHONPATH=src python -m tb_detection_model validate --help`
- Temporary ZIP smoke test for `extract` and `interim`, including an unsafe `../unsafe.txt` member: unsafe output was skipped and the raw ZIP SHA-256 remained unchanged.
- `PYTHONPATH=src python -m tb_detection_model validate`: observed exit 2 and explicit `UNSUPPORTED` message.

Deferred/intentionally not completed:

- End-to-end `dataset` smoke generation could not run in the available interpreter because `cv2` (and NumPy) are not installed; command invocation reported `ERROR: No module named 'cv2'`. The repository's configured environment should run it after dependencies are synchronized.
- Repository-wide test, build, lint, and formatter checks were deferred to the parent agent.
