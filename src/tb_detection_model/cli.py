"""Command-line interface for the repository's data workflows."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from .data.create_interim import create_interim_samples
from .data.extract import INTERIM_DIR, RAW_DIR, extract_all_datasets
from .data.make_dataset import build_dataset
from .config import (
    DEFAULT_CONFIG_PATH,
    ConfigFileError,
    as_dict,
    load_config,
    write_starter_config,
)


app = typer.Typer(
    name="tb-detection-model",
    help="Prepare and inspect tuberculosis chest-X-ray datasets.",
    no_args_is_help=True,
)
data_app = typer.Typer(help="Dataset workflows; `data build` is recommended.")
config_app = typer.Typer(help="Create and inspect TOML configuration.")
app.add_typer(data_app, name="data")
app.add_typer(config_app, name="config")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo("tb-detection-model 0.1.0")
        raise typer.Exit()


@app.callback()
def root_callback(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            help="Show the installed version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """Discover grouped data and configuration commands."""


def _config(
    config_path: Path | None,
    **overrides: object,
):
    return load_config(
        config_path,
        overrides={
            key: value
            for key, value in overrides.items()
            if value is not None
        },
    )


def _fail(error: Exception) -> None:
    typer.echo(f"Error: {error}", err=True)
    raise typer.Exit(code=2)


def _configure_logging(config) -> None:
    logging.basicConfig(
        level=config.log_level,
        format=config.log_format,
        force=True,
    )


@data_app.command("build")
def build_command(
    config_path: Annotated[
        Optional[Path],
        typer.Option("--config", help="TOML configuration file."),
    ] = None,
    input_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--input-dir",
            help="Directory containing raw ZIP archives.",
        ),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option("--output-dir", help="Directory for processed PNGs."),
    ] = None,
    seed: Annotated[
        Optional[int],
        typer.Option("--seed", help="Deterministic patient-group split seed."),
    ] = None,
    target_size: Annotated[
        Optional[tuple[int, int]],
        typer.Option("--target-size", metavar="HEIGHT WIDTH"),
    ] = None,
    train_ratio: Annotated[
        Optional[float],
        typer.Option("--train-ratio", help="Training split ratio."),
    ] = None,
    val_ratio: Annotated[
        Optional[float],
        typer.Option("--val-ratio", help="Validation split ratio."),
    ] = None,
    test_ratio: Annotated[
        Optional[float],
        typer.Option("--test-ratio", help="Test split ratio."),
    ] = None,
    log_level: Annotated[
        Optional[str],
        typer.Option(
            "--log-level",
            help="CRITICAL, ERROR, WARNING, INFO, DEBUG.",
        ),
    ] = None,
    train_archives: Annotated[
        Optional[list[str]],
        typer.Option(
            "--train-archive",
            "--train-archives",
            help="Repeat for archives assigned to training.",
        ),
    ] = None,
    validation_archives: Annotated[
        Optional[list[str]],
        typer.Option(
            "--validation-archive",
            "--validation-archives",
            help="Repeat for archives assigned to validation.",
        ),
    ] = None,
    test_archives: Annotated[
        Optional[list[str]],
        typer.Option(
            "--test-archive",
            "--test-archives",
            help="Repeat for archives assigned to test.",
        ),
    ] = None,
) -> None:
    """Build patient-grouped, preprocessed PNGs from raw ZIP archives."""
    overrides: dict[str, object] = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "seed": seed,
        "target_size": target_size,
        "log_level": log_level,
    }
    ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }
    if any(value is not None for value in ratios.values()):
        overrides["split_ratios"] = {
            name: value
            for name, value in ratios.items()
            if value is not None
        }
    if any(
        value is not None
        for value in (train_archives, validation_archives, test_archives)
    ):
        overrides.update(
            {
                "train_archives": train_archives or (),
                "validation_archives": validation_archives or (),
                "test_archives": test_archives or (),
            }
        )
    try:
        config = _config(config_path, **overrides)
        if not config.raw_dir.is_dir():
            raise FileNotFoundError(
                f"input directory does not exist: {config.raw_dir}"
            )
        _configure_logging(config)
        counts = build_dataset(
            config.raw_dir,
            config.processed_dir,
            seed=config.default_seed,
            train_archives=config.train_archives,
            validation_archives=config.validation_archives,
            test_archives=config.test_archives,
            config=config,
        )
    except (
        ConfigFileError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        _fail(exc)
    typer.echo(json.dumps(counts, sort_keys=True))


@data_app.command("extract")
def extract_command(
    input_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--input-dir",
            help="Directory containing raw ZIP archives.",
        ),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output-dir",
            help="Directory for extracted interim data.",
        ),
    ] = None,
) -> None:
    """LEGACY: extract every raw ZIP into interim directories."""
    try:
        extract_all_datasets(
            raw_dir=input_dir or RAW_DIR,
            interim_dir=output_dir or INTERIM_DIR,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        _fail(exc)


@data_app.command("interim")
def interim_command(
    input_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--input-dir",
            help="Directory containing raw ZIP archives.",
        ),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output-dir",
            help="Directory for sampled interim data.",
        ),
    ] = None,
    sample_size: Annotated[
        int,
        typer.Option(
            "--sample-size",
            min=1,
            help="Maximum members per archive.",
        ),
    ] = 100,
    seed: Annotated[
        Optional[int],
        typer.Option("--seed", help="Optional deterministic sampling seed."),
    ] = None,
) -> None:
    """LEGACY: sample archive members into interim directories."""
    try:
        create_interim_samples(
            raw_dir=input_dir or RAW_DIR,
            interim_dir=output_dir or INTERIM_DIR,
            sample_size=sample_size,
            seed=seed,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        _fail(exc)


@data_app.command("validate")
def validate_command() -> None:
    """UNSUPPORTED: the existing processed-data validator is incomplete."""
    _fail(
        RuntimeError(
            "UNSUPPORTED: data validation is not wired because "
            "src/data/validate_processed.py is incomplete"
        )
    )


@config_app.command("init")
def config_init_command(
    path: Annotated[
        Path,
        typer.Option("--path", help="Starter configuration path."),
    ] = DEFAULT_CONFIG_PATH,
    force: Annotated[
        bool,
        typer.Option("--force", help="Replace an existing configuration."),
    ] = False,
) -> None:
    """Create a documented starter TOML configuration."""
    try:
        write_starter_config(path, force=force)
    except (ConfigFileError, OSError) as exc:
        _fail(exc)
    typer.echo(f"Wrote configuration to {path}")


@config_app.command("show")
def config_show_command(
    config_path: Annotated[
        Optional[Path],
        typer.Option("--config", help="Explicit TOML configuration file."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """Show effective configuration after applying all precedence layers."""
    try:
        config = load_config(config_path)
    except (ConfigFileError, OSError, ValueError) as exc:
        _fail(exc)
    text = json.dumps(as_dict(config), indent=2, sort_keys=True)
    if json_output:
        typer.echo(text)
    else:
        typer.echo("Effective configuration:")
        typer.echo(text)


@config_app.command("validate")
def config_validate_command(
    config_path: Annotated[
        Optional[Path],
        typer.Option("--config", help="Explicit TOML configuration file."),
    ] = None,
) -> None:
    """Validate configuration before dataset processing starts."""
    try:
        config = load_config(config_path)
    except (ConfigFileError, OSError, ValueError) as exc:
        _fail(exc)
    typer.echo(
        f"Configuration is valid: {config.raw_dir} -> "
        f"{config.processed_dir}"
    )
