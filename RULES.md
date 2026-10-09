# Sticky rules for the project

- **Never commit generated data** (`data/interim/`, `data/processed/`).
- **All new source code must be accompanied by unit tests** placed under a `tests/` directory.
- **Enforce code style** with `flake8` (run via `make lint`) before committing.
- **Use deterministic randomness**: set a fixed seed (e.g., `random.seed(42)` or `np.random.seed(42)`) in any script that samples data.
- **Do not edit generated files** directly; modify the source scripts instead.
- **Sanitize any external inputs** (e.g., file paths, user‑provided parameters) before processing.
- **Update documentation** (`architecture.md`, `AGENTS.md`) whenever the pipeline, model, or data layout changes.
