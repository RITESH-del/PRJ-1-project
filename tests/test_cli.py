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


def test_model_command_discovery():
    result = runner.invoke(app, ["model", "--help"])
    assert result.exit_code == 0
    assert "train" in result.stdout
    assert "evaluate" in result.stdout
    assert "predict" in result.stdout

    train_help = runner.invoke(app, ["model", "train", "--help"])
    assert train_help.exit_code == 0
    assert "--epochs" in train_help.stdout
    assert "--lr" in train_help.stdout
    assert "--k8s-gpu" in train_help.stdout


def test_model_train_and_predict_cli(tmp_path):
    data_dir = tmp_path / "processed"
    for split in ("train", "val", "test"):
        for label in ("TB", "Normal"):
            folder = data_dir / split / label
            folder.mkdir(parents=True)
            for i in range(2):
                img = np.full((224, 224), 80 + i * 30, dtype=np.uint8)
                cv2.imwrite(str(folder / f"sample_{i}.png"), img)

    output_dir = tmp_path / "model_out"
    report_dir = tmp_path / "reports_out"

    train_res = runner.invoke(
        app,
        [
            "model",
            "train",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--report-dir",
            str(report_dir),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--freeze-backbone",
            "--num-workers",
            "0",
        ],
    )
    assert train_res.exit_code == 0, train_res.stdout
    assert (output_dir / "best_model.pt").exists()
    assert (report_dir / "metrics_test.json").exists()
    assert (report_dir / "dinov2_evaluation_report_test.md").exists()

    test_img = data_dir / "test" / "TB" / "sample_0.png"
    pred_res = runner.invoke(
        app,
        [
            "model",
            "predict",
            "--image",
            str(test_img),
            "--model-path",
            str(output_dir),
            "--device",
            "cpu",
            "--json",
        ],
    )
    assert pred_res.exit_code == 0, pred_res.stdout
    pred_data = json.loads(pred_res.stdout)
    assert pred_data["label"] in ("TB", "Normal")
    assert "probabilities" in pred_data


def test_densenet121_train_and_predict_cli(tmp_path):
    data_dir = tmp_path / "processed"
    for split in ("train", "val", "test"):
        for label in ("TB", "Normal"):
            folder = data_dir / split / label
            folder.mkdir(parents=True)
            for i in range(2):
                img = np.full((224, 224), 90 + i * 20, dtype=np.uint8)
                cv2.imwrite(str(folder / f"sample_{i}.png"), img)

    output_dir = tmp_path / "densenet_out"
    report_dir = tmp_path / "reports_out"

    train_res = runner.invoke(
        app,
        [
            "model",
            "train",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--report-dir",
            str(report_dir),
            "--model-name",
            "densenet121",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--freeze-backbone",
            "--num-workers",
            "0",
        ],
    )
    assert train_res.exit_code == 0, train_res.stdout
    assert (output_dir / "best_model.pt").exists()
    assert (report_dir / "densenet121_evaluation_report_test.md").exists()
    assert (report_dir / "metrics_densenet121_test.json").exists()

    test_img = data_dir / "test" / "TB" / "sample_0.png"
    pred_res = runner.invoke(
        app,
        [
            "model",
            "predict",
            "--image",
            str(test_img),
            "--model-path",
            str(output_dir),
            "--device",
            "cpu",
            "--json",
        ],
    )
    assert pred_res.exit_code == 0, pred_res.stdout
    pred_data = json.loads(pred_res.stdout)
    assert pred_data["label"] in ("TB", "Normal")
    assert "probabilities" in pred_data


def test_model_train_top_k_and_eval_val_cli(tmp_path):
    data_dir = tmp_path / "processed"
    for split in ("train", "val", "test"):
        for label in ("TB", "Normal"):
            folder = data_dir / split / label
            folder.mkdir(parents=True)
            for i in range(2):
                img = np.full((224, 224), 80 + i * 30, dtype=np.uint8)
                cv2.imwrite(str(folder / f"sample_{i}.png"), img)

    output_dir = tmp_path / "model_out"
    report_dir = tmp_path / "reports_out"

    train_res = runner.invoke(
        app,
        [
            "model",
            "train",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--report-dir",
            str(report_dir),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--train-top-k",
            "2",
            "--patience",
            "2",
            "--auto-class-weights",
            "--eval-split",
            "val",
            "--device",
            "cpu",
            "--num-workers",
            "0",
        ],
    )
    assert train_res.exit_code == 0, train_res.stdout
    assert (output_dir / "best_model.pt").exists()
    assert (report_dir / "metrics_val.json").exists()
    with open(report_dir / "metrics_val.json", encoding="utf-8") as f:
        metrics = json.load(f)
    assert "sensitivity" in metrics
    assert "specificity" in metrics


def test_model_list_cli():
    res = runner.invoke(app, ["model", "list"])
    assert res.exit_code == 0, res.stdout
    assert "Available Models & Performance Benchmarks" in res.stdout
    assert "dinov2_518" in res.stdout

    res_json = runner.invoke(app, ["model", "list", "--json"])
    assert res_json.exit_code == 0, res_json.stdout
    data = json.loads(res_json.stdout)
    assert isinstance(data, list)
    assert len(data) > 0
    names = [m["name"] for m in data]
    assert "dinov2_518" in names
    assert "densenet121" in names

    res_val = runner.invoke(app, ["model", "list", "--split", "val"])
    assert res_val.exit_code == 0, res_val.stdout
    assert "Split: val" in res_val.stdout


def test_model_info_cli():
    res = runner.invoke(app, ["model", "info", "dinov2_518"])
    assert res.exit_code == 0, res.stdout
    assert "Model: dinov2_518" in res.stdout
    assert "Architecture: facebook/dinov2-small" in res.stdout
    assert "Checkpoints:" in res.stdout
    assert "TEST Split Metrics" in res.stdout

    res_json = runner.invoke(app, ["model", "info", "dinov2_518", "--json"])
    assert res_json.exit_code == 0, res_json.stdout
    info = json.loads(res_json.stdout)
    assert info["name"] == "dinov2_518"
    assert "metrics" in info
    assert "test" in info["metrics"]

    res_err = runner.invoke(app, ["model", "info", "nonexistent_model_xyz"])
    assert res_err.exit_code != 0


def test_model_benchmark_cli():
    res = runner.invoke(app, ["model", "benchmark"])
    assert res.exit_code == 0, res.stdout
    assert "Reference Model Benchmark Summary" in res.stdout
    assert "Held-Out Test Set Performance Benchmark" in res.stdout
    assert "Clinical & Engineering Insights" in res.stdout

    res_json = runner.invoke(app, ["model", "benchmark", "--json"])
    assert res_json.exit_code == 0, res_json.stdout
    bench = json.loads(res_json.stdout)
    assert "benchmarks_test" in bench
    assert "dataset_splits" in bench
    assert len(bench["benchmarks_test"]) >= 5


def test_model_list_custom_temp_dir(tmp_path):
    mock_models = tmp_path / "custom_models"
    mock_reports = tmp_path / "custom_reports"
    model_dir = mock_models / "mock_vit"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "dinov2", "image_size": 224})
    )
    (model_dir / "best_model.pt").write_bytes(b"mock_checkpoint_data")

    res = runner.invoke(
        app,
        [
            "model",
            "list",
            "--models-dir",
            str(mock_models),
            "--reports-dir",
            str(mock_reports),
            "--json",
        ],
    )
    assert res.exit_code == 0, res.stdout
    data = json.loads(res.stdout)
    assert len(data) == 1
    assert data[0]["name"] == "mock_vit"
    assert data[0]["resolution"] == "224x224"
    assert data[0]["checkpoint_count"] == 1
