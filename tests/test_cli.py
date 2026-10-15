import csv
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from typer.testing import CliRunner

from src.tb_detection_model.cli import app
from src.tb_detection_model.config import load_config, write_starter_config


runner = CliRunner()


def _png_bytes(value: int) -> bytes:
    image = np.full((80, 100), value, dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


def _write_archive(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        for label, suffix, value, offset in (
            ("TB", "1", 80, 0),
            ("Normal", "0", 160, 100),
        ):
            for patient in range(1, 7):
                archive.writestr(
                    (
                        f"dataset/{label}/CHNCXR_"
                        f"{patient + offset:04d}_{suffix}.png"
                    ),
                    _png_bytes(value + patient),
                )


def test_command_discovery_and_module_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "tb-detection-model 0.1.0"

    result = runner.invoke(app, ["data", "--help"])
    assert result.exit_code == 0
    assert "build" in result.stdout
    assert "interim" in result.stdout


def test_config_precedence_and_starter_overwrite(tmp_path):
    config_path = tmp_path / "experiment.toml"
    config_path.write_text(
        """[dataset]
input_dir = "raw"
output_dir = "processed"
seed = 7
target_size = [96, 80]
[dataset.split_ratios]
train = 0.6
val = 0.2
test = 0.2
[logging]
level = "WARNING"
"""
    )

    config = load_config(
        config_path,
        overrides={"seed": 19, "target_size": (64, 72), "log_level": "DEBUG"},
    )
    assert config.default_seed == 19
    assert config.target_size == (64, 72)
    assert config.log_level == 10
    assert config.raw_dir == (tmp_path / "raw").resolve()
    assert dict(config.split_ratios) == {"train": 0.6, "val": 0.2, "test": 0.2}

    starter = tmp_path / "starter.toml"
    write_starter_config(starter)
    refused = runner.invoke(app, ["config", "init", "--path", str(starter)])
    assert refused.exit_code == 2
    replaced = runner.invoke(
        app, ["config", "init", "--path", str(starter), "--force"]
    )
    assert replaced.exit_code == 0


def test_invalid_config_fails_before_processing(tmp_path):
    config_path = tmp_path / "invalid.toml"
    output_dir = tmp_path / "processed"
    config_path.write_text(
        f"""[dataset]
input_dir = "{tmp_path / 'raw'}"
output_dir = "{output_dir}"
target_size = [0, 224]
[dataset.split_ratios]
train = 0.7
val = 0.7
test = -0.4
"""
    )

    result = runner.invoke(
        app,
        ["config", "validate", "--config", str(config_path)],
    )
    assert result.exit_code == 2
    assert "target_size" in result.output
    assert not output_dir.exists()


def test_config_only_build_uses_target_size_and_manifest(tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()
    _write_archive(raw_dir / "fixture.zip")
    config_path = tmp_path / "experiment.toml"
    config_path.write_text(
        f"""[dataset]
input_dir = "{raw_dir}"
output_dir = "{output_dir}"
target_size = [64, 72]
[dataset.split_ratios]
train = 0.7
val = 0.15
test = 0.15
[logging]
level = "ERROR"
"""
    )

    result = runner.invoke(
        app, ["data", "build", "--config", str(config_path)]
    )
    assert result.exit_code == 0, result.stdout
    counts = json.loads(result.stdout)
    assert sum(sum(split.values()) for split in counts.values()) == 12
    rows = list(csv.DictReader((output_dir / "manifest.csv").open()))
    assert len(rows) == 12
    image = cv2.imread(
        str(output_dir / rows[0]["filename"]),
        cv2.IMREAD_GRAYSCALE,
    )
    assert image.shape == (64, 72)
