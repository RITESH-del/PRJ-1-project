"""TOML configuration for the dataset-building CLI."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import logging
import tomllib
from pathlib import Path
from typing import Mapping

from .data.make_dataset import (
    CONFIG as DEFAULT_DATASET_CONFIG,
    ConfigurationError,
    DatasetConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "tb-detection.toml"


class ConfigFileError(ValueError):
    """Raised when a TOML configuration cannot be loaded or mapped."""


@dataclass(frozen=True)
class ModelConfig:
    """All user-tunable model training and evaluation parameters."""

    model_name: str = "facebook/dinov2-small"
    data_dir: Path = DEFAULT_DATASET_CONFIG.processed_dir
    output_dir: Path = PROJECT_ROOT / "models" / "dinov2"
    report_dir: Path = PROJECT_ROOT / "reports"
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-4
    freeze_backbone: bool = False
    backbone_lr: float | None = None
    train_top_k: int = 2
    patience: int = 3
    auto_class_weights: bool = False
    pos_weight: float | None = None
    augment: bool = True
    device: str = "auto"
    seed: int = 42
    num_workers: int = 2
    dual_representation: bool = True


STARTER_CONFIG = """# TB Detection Model experiment configuration.
# Relative paths are resolved from this file's directory.

[dataset]
input_dir = "data/raw"
output_dir = "data/processed"
seed = 42
target_size = [518, 518]
manifest_filename = "manifest.csv"
archive_glob = "*.zip"
train_archives = []
validation_archives = []
test_archives = []

[dataset.split_ratios]
train = 0.70
val = 0.15
test = 0.15

[preprocessing]
minimum_image_size = [50, 50]
minimum_image_mean = 5.0
roi_mode = "anatomical"
roi_height_fraction = [0.02, 0.98]
roi_width_fraction = [0.02, 0.98]
median_blur_kernel = 0
bilateral_filter = true
clahe_clip_limit = 1.5
clahe_tile_grid_size = [16, 16]

[logging]
level = "INFO"
scan_log_interval = 1000
process_log_interval = 100

[model]
name = "facebook/dinov2-small"
data_dir = "data/processed"
output_dir = "models/dinov2"
report_dir = "reports"
epochs = 5
batch_size = 32
learning_rate = 0.0001
freeze_backbone = false
train_top_k = 2
patience = 3
auto_class_weights = false
device = "auto"
seed = 42
num_workers = 2
dual_representation = true
"""


_LEVELS = {
    name: getattr(logging, name)
    for name in ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")
}


def _default_mapping() -> dict[str, object]:
    config = DEFAULT_DATASET_CONFIG
    return {
        "dataset": {
            "input_dir": str(config.raw_dir),
            "output_dir": str(config.processed_dir),
            "seed": config.default_seed,
            "target_size": list(config.target_size),
            "manifest_filename": config.manifest_filename,
            "archive_glob": config.archive_glob,
            "train_archives": [],
            "validation_archives": [],
            "test_archives": [],
            "split_ratios": dict(config.split_ratios),
        },
        "preprocessing": {
            "minimum_image_size": list(config.minimum_image_size),
            "minimum_image_mean": config.minimum_image_mean,
            "roi_mode": config.roi_mode,
            "roi_height_fraction": list(config.roi_height_fraction),
            "roi_width_fraction": list(config.roi_width_fraction),
            "median_blur_kernel": config.median_blur_kernel,
            "bilateral_filter": config.bilateral_filter,
            "bilateral_d": config.bilateral_d,
            "bilateral_sigma_color": config.bilateral_sigma_color,
            "bilateral_sigma_space": config.bilateral_sigma_space,
            "clahe_clip_limit": config.clahe_clip_limit,
            "clahe_tile_grid_size": list(config.clahe_tile_grid_size),
        },
        "logging": {
            "level": logging.getLevelName(config.log_level),
            "scan_log_interval": config.scan_log_interval,
            "process_log_interval": config.process_log_interval,
        },
        "model": {
            "name": "facebook/dinov2-small",
            "data_dir": str(config.processed_dir),
            "output_dir": str((PROJECT_ROOT / "models" / "dinov2").resolve()),
            "report_dir": str((PROJECT_ROOT / "reports").resolve()),
            "epochs": 5,
            "batch_size": 32,
            "learning_rate": 1e-4,
            "freeze_backbone": False,
            "train_top_k": 2,
            "patience": 3,
            "auto_class_weights": False,
            "pos_weight": None,
            "backbone_lr": None,
            "augment": True,
            "device": "auto",
            "seed": 42,
            "num_workers": 2,
            "dual_representation": True,
        },
    }


def _merge(base: dict[str, object], overlay: Mapping[str, object]) -> None:
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _merge(base[key], value)  # type: ignore[arg-type]
        else:
            base[key] = copy.deepcopy(value)


def _resolve_paths(
    table: dict[str, object],
    keys: tuple[str, ...],
    prefix: str,
    base_dir: Path,
) -> None:
    for key in keys:
        value = table.get(key)
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ConfigFileError(
                    f"{prefix}.{key} must be a non-empty path"
                )
            table[key] = str((base_dir / value).resolve())


def _read_layer(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as config_file:
            values = tomllib.load(config_file)
    except FileNotFoundError as exc:
        raise ConfigFileError(f"configuration file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigFileError(f"invalid TOML in {path}: {exc}") from exc

    if not isinstance(values, dict):
        raise ConfigFileError(f"configuration must be a TOML table: {path}")
    values = copy.deepcopy(values)
    dataset = values.get("dataset", {})
    if not isinstance(dataset, dict):
        raise ConfigFileError("[dataset] must be a TOML table")
    _resolve_paths(
        dataset, ("input_dir", "output_dir"), "dataset", path.parent
    )

    model = values.get("model", {})
    if isinstance(model, dict):
        keys = ("data_dir", "output_dir", "report_dir")
        _resolve_paths(model, keys, "model", path.parent)
    return values


def _section(values: Mapping[str, object], name: str) -> Mapping[str, object]:
    section = values.get(name, {})
    if not isinstance(section, Mapping):
        raise ConfigFileError(f"[{name}] must be a TOML table")
    return section


def _two_ints(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigFileError(f"{name} must contain exactly two integers")
    if any(
        not isinstance(item, int) or isinstance(item, bool)
        for item in value
    ):
        raise ConfigFileError(f"{name} must contain exactly two integers")
    return int(value[0]), int(value[1])


def _two_numbers(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigFileError(f"{name} must contain exactly two numbers")
    if any(
        not isinstance(item, (int, float)) or isinstance(item, bool)
        for item in value
    ):
        raise ConfigFileError(f"{name} must contain exactly two numbers")
    return float(value[0]), float(value[1])


def _archives(value: object, name: str) -> tuple[str, ...] | None:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ConfigFileError(
            f"dataset.{name} must be a list of archive names"
        )
    return tuple(value) or None


def _logging_level(value: object) -> int:
    if not isinstance(value, str) or value.upper() not in _LEVELS:
        choices = ", ".join(sorted(_LEVELS))
        raise ConfigFileError(f"logging.level must be one of: {choices}")
    return _LEVELS[value.upper()]


def _apply_overrides(
    values: dict[str, object], overrides: Mapping[str, object]
) -> None:
    dataset = values["dataset"]
    logging_values = values["logging"]
    preprocessing = values["preprocessing"]
    assert isinstance(dataset, dict)
    assert isinstance(logging_values, dict)
    assert isinstance(preprocessing, dict)
    preprocessing_keys = {
        "minimum_image_size",
        "minimum_image_mean",
        "roi_mode",
        "roi_height_fraction",
        "roi_width_fraction",
        "median_blur_kernel",
        "bilateral_filter",
        "bilateral_d",
        "bilateral_sigma_color",
        "bilateral_sigma_space",
        "clahe_clip_limit",
        "clahe_tile_grid_size",
        "scan_log_interval",
        "process_log_interval",
    }
    model = values.get("model")
    for key, value in overrides.items():
        if key.startswith("model.") and isinstance(model, dict):
            model[key.removeprefix("model.")] = value
        elif key == "log_level":
            logging_values["level"] = value
        elif key == "split_ratios":
            if not isinstance(value, Mapping):
                raise ConfigFileError("split_ratios override must be a table")
            ratios = dataset["split_ratios"]
            if not isinstance(ratios, dict):
                raise ConfigFileError(
                    "dataset.split_ratios must be a TOML table"
                )
            ratios.update(value)
        elif key in preprocessing_keys:
            preprocessing[key] = value
        else:
            dataset[key] = value


def _make_config(values: Mapping[str, object]) -> DatasetConfig:
    dataset = _section(values, "dataset")
    ratios = dataset.get("split_ratios")
    if not isinstance(ratios, Mapping):
        raise ConfigFileError("dataset.split_ratios must be a TOML table")
    preprocessing = _section(values, "preprocessing")
    logging_values = _section(values, "logging")
    try:
        return DatasetConfig(
            raw_dir=Path(dataset["input_dir"]),
            processed_dir=Path(dataset["output_dir"]),
            archive_glob=str(dataset["archive_glob"]),
            manifest_filename=str(dataset["manifest_filename"]),
            split_ratios=tuple(
                (name, float(ratios[name]))
                for name in ("train", "val", "test")
            ),
            default_seed=dataset["seed"],
            target_size=_two_ints(
                dataset["target_size"], "dataset.target_size"
            ),
            minimum_image_size=_two_ints(
                preprocessing["minimum_image_size"],
                "preprocessing.minimum_image_size",
            ),
            minimum_image_mean=float(preprocessing["minimum_image_mean"]),
            roi_mode=str(preprocessing.get("roi_mode", "anatomical")),
            roi_height_fraction=_two_numbers(
                preprocessing["roi_height_fraction"],
                "preprocessing.roi_height_fraction",
            ),
            roi_width_fraction=_two_numbers(
                preprocessing["roi_width_fraction"],
                "preprocessing.roi_width_fraction",
            ),
            median_blur_kernel=int(preprocessing.get("median_blur_kernel", 0)),
            bilateral_filter=bool(
                preprocessing.get("bilateral_filter", True)
            ),
            bilateral_d=int(preprocessing.get("bilateral_d", 5)),
            bilateral_sigma_color=float(
                preprocessing.get("bilateral_sigma_color", 30.0)
            ),
            bilateral_sigma_space=float(
                preprocessing.get("bilateral_sigma_space", 30.0)
            ),
            clahe_clip_limit=float(preprocessing["clahe_clip_limit"]),
            clahe_tile_grid_size=_two_ints(
                preprocessing["clahe_tile_grid_size"],
                "preprocessing.clahe_tile_grid_size",
            ),
            log_level=_logging_level(logging_values["level"]),
            scan_log_interval=logging_values["scan_log_interval"],
            process_log_interval=logging_values["process_log_interval"],
            train_archives=_archives(
                dataset.get("train_archives", []), "train_archives"
            ),
            validation_archives=_archives(
                dataset.get("validation_archives", []),
                "validation_archives",
            ),
            test_archives=_archives(
                dataset.get("test_archives", []), "test_archives"
            ),
        )
    except (ConfigurationError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ConfigurationError):
            raise ConfigFileError(str(exc)) from exc
        raise ConfigFileError(f"invalid dataset configuration: {exc}") from exc


def load_config(
    config_path: Path | None = None,
    *,
    overrides: Mapping[str, object] | None = None,
) -> DatasetConfig:
    """Load defaults, TOML layers, then explicit CLI overrides."""
    values = _default_mapping()
    if DEFAULT_CONFIG_PATH.exists():
        _merge(values, _read_layer(DEFAULT_CONFIG_PATH))
    if config_path is not None:
        _merge(values, _read_layer(config_path))
    if overrides:
        _apply_overrides(values, overrides)
    return _make_config(values)


def _validate_model_config_bounds(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    train_top_k: int,
    patience: int,
    pos_weight: float | None,
) -> None:
    if epochs <= 0:
        raise ConfigFileError("model.epochs must be positive")
    if batch_size <= 0:
        raise ConfigFileError("model.batch_size must be positive")
    if learning_rate <= 0:
        raise ConfigFileError("model.learning_rate must be positive")
    if train_top_k < 0:
        raise ConfigFileError("model.train_top_k must be non-negative")
    if patience < 0:
        raise ConfigFileError("model.patience must be non-negative")
    if pos_weight is not None and pos_weight <= 0:
        raise ConfigFileError("model.pos_weight must be positive")


def _make_model_config(values: Mapping[str, object]) -> ModelConfig:
    model = _section(values, "model")
    default_data_dir = DEFAULT_DATASET_CONFIG.processed_dir
    try:
        data_dir = Path(str(model.get("data_dir", default_data_dir)))
        output_dir = Path(
            str(model.get("output_dir", PROJECT_ROOT / "models" / "dinov2"))
        )
        report_dir = Path(
            str(model.get("report_dir", PROJECT_ROOT / "reports"))
        )
        epochs = int(model["epochs"])  # type: ignore
        batch_size = int(model["batch_size"])  # type: ignore
        learning_rate = float(model["learning_rate"])  # type: ignore
        freeze_backbone = bool(model.get("freeze_backbone", False))
        train_top_k = int(model.get("train_top_k", 2))  # type: ignore
        patience = int(model.get("patience", 3))  # type: ignore
        auto_class_weights = bool(model.get("auto_class_weights", False))
        pos_weight_val = model.get("pos_weight")
        pos_weight = (
            float(pos_weight_val) if pos_weight_val is not None else None
        )
        device = str(model.get("device", "auto"))
        seed = int(model.get("seed", 42))  # type: ignore
        num_workers = int(model.get("num_workers", 2))  # type: ignore
        model_name = str(
            model.get("name", model.get("model_name", "facebook/dinov2-small"))
        )
        backbone_lr_val = model.get("backbone_lr")
        backbone_lr = (
            float(backbone_lr_val) if backbone_lr_val is not None else None
        )
        augment = bool(model.get("augment", True))
        dual_representation = bool(model.get("dual_representation", True))

        _validate_model_config_bounds(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            train_top_k=train_top_k,
            patience=patience,
            pos_weight=pos_weight,
        )
        return ModelConfig(
            model_name=model_name,
            data_dir=data_dir,
            output_dir=output_dir,
            report_dir=report_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            freeze_backbone=freeze_backbone,
            train_top_k=train_top_k,
            patience=patience,
            auto_class_weights=auto_class_weights,
            pos_weight=pos_weight,
            backbone_lr=backbone_lr,
            augment=augment,
            device=device,
            seed=seed,
            num_workers=num_workers,
            dual_representation=dual_representation,
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ConfigFileError):
            raise
        raise ConfigFileError(f"invalid model configuration: {exc}") from exc


def load_model_config(
    config_path: Path | None = None,
    *,
    overrides: Mapping[str, object] | None = None,
) -> ModelConfig:
    """Load defaults, TOML layers, and explicit overrides for model."""
    values = _default_mapping()
    if DEFAULT_CONFIG_PATH.exists():
        _merge(values, _read_layer(DEFAULT_CONFIG_PATH))
    if config_path is not None:
        _merge(values, _read_layer(config_path))
    if overrides:
        _apply_overrides(values, overrides)
    return _make_model_config(values)


def as_model_dict(config: ModelConfig) -> dict[str, object]:
    """Return stable, JSON-serializable model configuration values."""
    return {
        "model": {
            "name": config.model_name,
            "data_dir": str(config.data_dir),
            "output_dir": str(config.output_dir),
            "report_dir": str(config.report_dir),
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "freeze_backbone": config.freeze_backbone,
            "train_top_k": config.train_top_k,
            "patience": config.patience,
            "auto_class_weights": config.auto_class_weights,
            "pos_weight": config.pos_weight,
            "device": config.device,
            "seed": config.seed,
            "num_workers": config.num_workers,
        }
    }


def as_dict(config: DatasetConfig) -> dict[str, object]:
    """Return stable, JSON-serializable effective configuration values."""
    return {
        "dataset": {
            "input_dir": str(config.raw_dir),
            "output_dir": str(config.processed_dir),
            "seed": config.default_seed,
            "target_size": list(config.target_size),
            "manifest_filename": config.manifest_filename,
            "archive_glob": config.archive_glob,
            "train_archives": list(config.train_archives or ()),
            "validation_archives": list(config.validation_archives or ()),
            "test_archives": list(config.test_archives or ()),
            "split_ratios": dict(config.split_ratios),
        },
        "preprocessing": {
            "minimum_image_size": list(config.minimum_image_size),
            "minimum_image_mean": config.minimum_image_mean,
            "roi_mode": config.roi_mode,
            "roi_height_fraction": list(config.roi_height_fraction),
            "roi_width_fraction": list(config.roi_width_fraction),
            "median_blur_kernel": config.median_blur_kernel,
            "bilateral_filter": config.bilateral_filter,
            "bilateral_d": config.bilateral_d,
            "bilateral_sigma_color": config.bilateral_sigma_color,
            "bilateral_sigma_space": config.bilateral_sigma_space,
            "clahe_clip_limit": config.clahe_clip_limit,
            "clahe_tile_grid_size": list(config.clahe_tile_grid_size),
        },
        "logging": {
            "level": logging.getLevelName(config.log_level),
            "scan_log_interval": config.scan_log_interval,
            "process_log_interval": config.process_log_interval,
        },
    }


def write_starter_config(path: Path, *, force: bool = False) -> None:
    """Write the documented starter TOML, refusing accidental overwrite."""
    if path.exists() and not force:
        raise ConfigFileError(
            f"configuration already exists: {path} (use --force to replace)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(STARTER_CONFIG)
