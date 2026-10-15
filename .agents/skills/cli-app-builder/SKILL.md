---
name: cli-app-builder
description: Build and evolve repository-native, user- and agent-friendly CLI applications that expose existing scripts and workflows without requiring source edits for routine adjustments. Use this skill whenever a user asks to create a CLI, wrap scripts in commands, make dataset or pipeline settings configurable, add config-file support, expose `--help`, repair a console entry point, or document commands for agents and contributors—even if they do not explicitly call it a CLI app.
compatibility: Python repositories with a package entry point; this skill standardizes on Typer, TOML configuration, pathlib, dataclasses, pytest, and the repository's existing environment/tooling.
---

# Build a configurable repository CLI

Create a thin command layer over the repository's real workflows. Users should adjust paths, seeds, split assignments, preprocessing, and other supported settings through a checked-in TOML file or explicit command-line options—not by editing Python. Agents should be able to discover commands from `--help`, pass deterministic flags, and inspect stable output.

## Operating rules

1. Read `AGENTS.md`, `CONTRIBUTING.md`, `pyproject.toml`, the package entry point, current scripts, and the relevant tests before editing. Treat repository guidance as the contract.
2. Inventory executable scripts, Make targets, public functions, dataclasses, and existing CLI parsers. Preserve the current production path and behavior; do not create a second implementation of the business logic.
3. Keep the CLI adapter thin. Move only genuinely reusable parsing/configuration code into small typed helpers. Existing functions remain the source of truth for dataset processing.
4. Treat `data/raw/` as immutable and never commit generated `data/interim/` or `data/processed/` content. Use temporary fixtures for smoke tests.
5. Do not expose unfinished model-training or prediction placeholders as working commands. Expose them only when a real implementation and contract exist.

## Standard interface

Use Typer with a root app and grouped subcommands. Keep names stable, lowercase, and verb-oriented. The root command must provide useful `--help` and version information.

Expose the repository's existing operational scripts as commands, adapting the names to the repository rather than inventing unrelated workflows. For this repository, the initial command inventory includes the recommended dataset builder plus the legacy extraction/interim utilities and validation entry point, for example:

```text
tb-detection-model data build
tb-detection-model data validate
tb-detection-model data extract
tb-detection-model data interim
tb-detection-model config init
tb-detection-model config show
tb-detection-model config validate
```

If a script cannot run safely because it is incomplete, make that fact explicit in help and fail with an actionable message; never silently pretend it succeeded. Keep legacy commands available only as compatibility paths and label the recommended path in help and docs.

Use the existing installed console entry point when one exists. If it points at a missing `main`, implement the smallest package-level entry point that calls the Typer app. Preserve the project package name and add no duplicate launcher unless the repository already uses one.

## Configuration contract

Use a root-level `tb-detection.toml` for this repository. For a reusable repository, derive the filename from the project name only when the project already has an established convention; otherwise prefer a clear root-level `<project>-<purpose>.toml` file.

Apply settings in this order:

1. Built-in typed defaults.
2. The default config file, when present.
3. An explicitly supplied `--config PATH` file.
4. Explicit CLI options.

Make the effective configuration visible with `config show`; support a machine-readable format when practical so agents do not need to parse prose. `config init` should create a documented starter file, refuse to overwrite an existing file unless the user explicitly requests replacement, and preserve comments only when the chosen TOML writer supports that safely. `config validate` should fail before expensive processing and identify the setting and reason for every invalid value.

Map TOML values into existing typed configuration objects, such as `DatasetConfig`, rather than maintaining parallel constants. Validate cross-field invariants at the boundary: paths, split ratios, labels, dimensions, odd blur kernels, positive standard deviations, archive assignments, and any source-specific constraints. CLI overrides and TOML keys must use one documented naming convention.

Every setting users are expected to adjust must have one of these paths:

- a documented TOML key;
- a typed CLI option with the same semantic name; or
- an explicit reason why it is intentionally not configurable.

For this dataset pipeline, cover at least input/output paths, archive selection, seed, split ratios, target size, preprocessing thresholds/options, manifest settings, and logging. Keep fixed scientific constants fixed unless the repository's existing API already treats them as user-tunable.

## Implementation sequence

1. Build an inventory and write down the command/config contract before coding.
2. Add or refine the Typer app and typed config loader. Reuse `DatasetConfig` and existing script functions.
3. Wire every command to the real implementation, including clear exit codes and logging. Avoid `os.chdir`, global mutable configuration, shelling out to Python scripts, and duplicated preprocessing.
4. Add the installed console entry point and dependency using the repository's package manager. Update the lockfile when dependency metadata changes.
5. Add focused tests for command discovery, config precedence, invalid configuration, installed/module invocation, and one end-to-end temporary-fixture smoke path. Test behavior, not help-text formatting alone.
6. Update **both** `AGENTS.md` and `CONTRIBUTING.md` with the canonical invocation examples, command inventory, config location, precedence, and the rule that `--help` is authoritative. Keep examples copy-pasteable and synchronized.
7. Run the repository's required tests and lint, then execute the actual CLI on a small temporary fixture. Verify output files, manifest/provenance, exit status, and that a config-only change affects behavior without a source edit.

## Documentation contract

Document commands in this order:

1. installation/environment setup;
2. `tb-detection-model --help` and subcommand discovery;
3. recommended dataset build;
4. source-segmented build example;
5. config-file example and override precedence;
6. validation and legacy commands;
7. tests and lint.

`AGENTS.md` is the agent-facing operational reference. `CONTRIBUTING.md` is the contributor-facing walkthrough. Update both in the same change; do not leave one with stale command syntax. Do not claim commands work until the smoke run proves them.

## Completion report

Report:

- the CLI entry point and command tree;
- the config file path, schema/keys, and precedence;
- files changed, including both command documents;
- tests, lint, and real CLI smoke commands run;
- any intentionally unsupported or incomplete command, with its exact reason.

Do not finish with a scaffold, placeholder, undocumented flag, or a claim based only on compilation.
