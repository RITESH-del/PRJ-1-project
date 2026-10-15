"""Command-line interface for the repository's data workflows."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import subprocess
import sys
import uuid
from typing import Annotated, Optional

import typer
from torch.utils.data import DataLoader

from .models.predict_model import (
    evaluate_model,
    load_trained_model,
    predict_image,
    save_evaluation_report,
)
from .models.metadata import (
    format_benchmark_report,
    format_model_info,
    format_models_table,
    get_benchmark_reference,
    get_model_details,
    list_models,
)
from .models.train_model import (
    TBDataset,
    TrainConfig,
    get_device,
    train_model,
)
from .data.create_interim import create_interim_samples
from .data.extract import INTERIM_DIR, RAW_DIR, extract_all_datasets
from .data.make_dataset import build_dataset
from .config import (
    DEFAULT_CONFIG_PATH,
    ConfigFileError,
    as_dict,
    load_config,
    load_model_config,
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
model_app = typer.Typer(
    help="Model training, evaluation, and inference workflows."
)
app.add_typer(model_app, name="model")


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
    roi_mode: Annotated[
        Optional[str],
        typer.Option(
            "--roi-mode",
            help="Thorax ROI mode: 'anatomical' or 'fractional'.",
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
    if roi_mode is not None:
        overrides["roi_mode"] = roi_mode
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


def _run_k8s_gpu_job(cmd_args: list[str]) -> None:
    """Launch training inside an NVIDIA H200 MIG GPU pod on Kubernetes."""
    pod_name = f"tb-train-{uuid.uuid4().hex[:8]}"
    namespace = "dgx-s-bmu-soet-240609-restricted"
    bin_path = Path(sys.executable).parent / "tb-detection-model"
    full_cmd = [str(bin_path), "model", "train"] + cmd_args

    manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": pod_name, "namespace": namespace},
        "spec": {
            "restartPolicy": "Never",
            "volumes": [
                {
                    "name": "home-dir",
                    "hostPath": {"path": "/home/dgx-s-bmu-soet-240609"},
                },
                {"name": "dshm", "emptyDir": {"medium": "Memory"}},
            ],
            "containers": [
                {
                    "name": "trainer",
                    "image": "nvidia/cuda:12.4.1-base-ubuntu22.04",
                    "workingDir": (
                        "/home/dgx-s-bmu-soet-240609/projects/"
                        "TB_Detection_Model"
                    ),
                    "command": full_cmd,
                    "env": [
                        {"name": "PYTHONUNBUFFERED", "value": "1"},
                        {
                            "name": "HOME",
                            "value": "/home/dgx-s-bmu-soet-240609",
                        },
                        {
                            "name": "MPLCONFIGDIR",
                            "value": "/tmp/matplotlib",
                        },
                        {
                            "name": "HF_HOME",
                            "value": (
                                "/home/dgx-s-bmu-soet-240609/.cache/"
                                "huggingface"
                            ),
                        },
                    ],
                    "volumeMounts": [
                        {
                            "name": "home-dir",
                            "mountPath": "/home/dgx-s-bmu-soet-240609",
                        },
                        {"name": "dshm", "mountPath": "/dev/shm"},
                    ],
                    "resources": {
                        "limits": {
                            "nvidia.com/mig-1g.18gb": "1",
                            "cpu": "4",
                            "memory": "16Gi",
                        },
                        "requests": {
                            "nvidia.com/mig-1g.18gb": "1",
                            "cpu": "4",
                            "memory": "16Gi",
                        },
                    },
                }
            ],
        },
    }

    manifest_json = json.dumps(manifest)
    k8s_prefix = (
        "source /etc/profile.d/modules.sh && "
        "module load kubernetes/BMU-Cluster/1.32.11-1.1 && "
    )
    typer.echo(f"Submitting GPU training pod: {pod_name}")
    apply_cmd = f"{k8s_prefix}kubectl apply -f -"
    sub = subprocess.run(
        ["bash", "-c", apply_cmd],
        input=manifest_json,
        text=True,
        capture_output=True,
    )
    if sub.returncode != 0:
        raise RuntimeError(f"Failed to submit GPU pod: {sub.stderr}")

    try:
        ready_cmd = (
            f"{k8s_prefix}kubectl wait --for=condition=Ready "
            f"pod/{pod_name} -n {namespace} --timeout=120s"
        )
        subprocess.run(["bash", "-c", ready_cmd], check=False)

        stream_cmd = f"{k8s_prefix}kubectl logs -f {pod_name} -n {namespace}"
        subprocess.run(["bash", "-c", stream_cmd], check=False)

        wait_done_cmd = (
            f"{k8s_prefix}kubectl wait "
            f"--for=jsonpath='{{.status.phase}}'=Succeeded "
            f"pod/{pod_name} -n {namespace} --timeout=30s"
        )
        res = subprocess.run(
            ["bash", "-c", wait_done_cmd],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            status_cmd = (
                f"{k8s_prefix}kubectl get pod {pod_name} -n {namespace} "
                "-o jsonpath='{.status.phase}'"
            )
            status = subprocess.run(
                ["bash", "-c", status_cmd],
                text=True,
                capture_output=True,
                check=False,
            ).stdout.strip().strip("'")
            if status != "Succeeded":
                raise RuntimeError(
                    f"Pod {pod_name} finished with status: {status}"
                )
    finally:
        del_cmd = f"{k8s_prefix}kubectl delete pod {pod_name} -n {namespace}"
        subprocess.run(["bash", "-c", del_cmd], check=False)


def _collect_k8s_args(
    config_path: Path | None,
    data_dir: Path | None,
    output_dir: Path | None,
    report_dir: Path | None,
    model_name: str | None,
    epochs: int | None,
    batch_size: int | None,
    learning_rate: float | None,
    backbone_lr: float | None,
    freeze_backbone: bool | None,
    train_top_k: int | None,
    patience: int | None,
    auto_class_weights: bool | None,
    pos_weight: float | None,
    augment: bool | None,
    seed: int | None,
    num_workers: int | None,
    evaluate_test: bool,
    eval_split: str,
    dual_representation: bool | None = None,
) -> list[str]:
    args: list[str] = ["--no-k8s-gpu", "--device", "cuda"]
    options = [
        ("--config", config_path),
        ("--data-dir", data_dir),
        ("--output-dir", output_dir),
        ("--report-dir", report_dir),
        ("--model-name", model_name),
        ("--epochs", epochs),
        ("--batch-size", batch_size),
        ("--lr", learning_rate),
        ("--backbone-lr", backbone_lr),
        ("--train-top-k", train_top_k),
        ("--patience", patience),
        ("--pos-weight", pos_weight),
        ("--eval-split", eval_split),
        ("--seed", seed),
        ("--num-workers", num_workers),
    ]
    for flag, val in options:
        if val is not None:
            args.extend([flag, str(val)])
    if freeze_backbone is not None:
        args.append("--freeze-backbone" if freeze_backbone else "--fine-tune")
    bool_flags = [
        (
            auto_class_weights,
            "--auto-class-weights",
            "--no-auto-class-weights",
        ),
        (augment, "--augment", "--no-augment"),
        (
            dual_representation,
            "--dual-representation",
            "--no-dual-representation",
        ),
    ]
    for val, pos, neg in bool_flags:
        if val is True:
            args.append(pos)
        elif val is False:
            args.append(neg)
    if not evaluate_test:
        args.append("--no-evaluate-test")
    return args


def _collect_overrides(
    data_dir: Path | None,
    output_dir: Path | None,
    report_dir: Path | None,
    model_name: str | None,
    epochs: int | None,
    batch_size: int | None,
    learning_rate: float | None,
    freeze_backbone: bool | None,
    backbone_lr: float | None,
    train_top_k: int | None,
    patience: int | None,
    auto_class_weights: bool | None,
    pos_weight: float | None,
    augment: bool | None,
    device: str | None,
    seed: int | None,
    num_workers: int | None,
    dual_representation: bool | None = None,
) -> dict[str, object]:
    mapping: dict[str, object] = {
        "model.data_dir": str(data_dir) if data_dir else None,
        "model.output_dir": str(output_dir) if output_dir else None,
        "model.report_dir": str(report_dir) if report_dir else None,
        "model.name": model_name,
        "model.epochs": epochs,
        "model.batch_size": batch_size,
        "model.learning_rate": learning_rate,
        "model.freeze_backbone": freeze_backbone,
        "model.backbone_lr": backbone_lr,
        "model.train_top_k": train_top_k,
        "model.patience": patience,
        "model.auto_class_weights": auto_class_weights,
        "model.pos_weight": pos_weight,
        "model.augment": augment,
        "model.device": device,
        "model.seed": seed,
        "model.num_workers": num_workers,
        "model.dual_representation": dual_representation,
    }
    return {k: v for k, v in mapping.items() if v is not None}


def _run_post_train_eval(
    model: object,
    train_cfg: TrainConfig,
    report_dir: Path,
    history: dict[str, list[float]],
    split: str = "val",
) -> None:
    typer.echo(f"Evaluating on {split} split...")
    dev = get_device(train_cfg.device)
    ds = TBDataset(
        train_cfg.data_dir,
        split=split,
        dual_representation=train_cfg.dual_representation,
        target_size=train_cfg.target_size,
    )
    loader = DataLoader(
        ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
    )
    metrics, y_true, _, y_prob = evaluate_model(
        model, loader, dev  # type: ignore[arg-type]
    )
    report_file = save_evaluation_report(
        metrics=metrics,
        y_true=y_true,
        y_prob=y_prob,
        history=history,
        report_dir=report_dir,
        split_name=split,
        model_name=train_cfg.model_name,
    )
    typer.echo(f"Evaluation report saved to {report_file}")
    acc = metrics["accuracy"]
    sens = metrics.get("sensitivity", metrics["recall"])
    spec = metrics.get("specificity", 0.0)
    f1 = metrics["f1"]
    auc = metrics["roc_auc"]
    typer.echo(
        f"{split.capitalize()} Accuracy: {acc:.4f} | Sens: {sens:.4f} | "
        f"Spec: {spec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}"
    )


@model_app.command("train")
def train_command(
    config_path: Annotated[
        Optional[Path],
        typer.Option("--config", help="Path to TOML configuration file."),
    ] = None,
    data_dir: Annotated[
        Optional[Path],
        typer.Option("--data-dir", help="Path to processed data directory."),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option("--output-dir", help="Directory to save model."),
    ] = None,
    report_dir: Annotated[
        Optional[Path],
        typer.Option("--report-dir", help="Directory to save reports."),
    ] = None,
    model_name: Annotated[
        Optional[str],
        typer.Option("--model-name", help="Hugging Face model identifier."),
    ] = None,
    epochs: Annotated[
        Optional[int],
        typer.Option("--epochs", help="Number of training epochs."),
    ] = None,
    batch_size: Annotated[
        Optional[int],
        typer.Option("--batch-size", help="Batch size for training."),
    ] = None,
    learning_rate: Annotated[
        Optional[float],
        typer.Option("--lr", help="Learning rate for AdamW optimizer."),
    ] = None,
    freeze_backbone: Annotated[
        Optional[bool],
        typer.Option(
            "--freeze-backbone/--fine-tune",
            help="Freeze DINOv2 backbone or fine-tune all weights.",
        ),
    ] = None,
    backbone_lr: Annotated[
        Optional[float],
        typer.Option(
            "--backbone-lr",
            help="Learning rate for backbone when fine-tuning.",
        ),
    ] = None,
    train_top_k: Annotated[
        Optional[int],
        typer.Option(
            "--train-top-k",
            help="Final Transformer blocks to fine-tune (e.g. 2, 3, 4).",
        ),
    ] = None,
    patience: Annotated[
        Optional[int],
        typer.Option(
            "--patience",
            help="Early stopping patience in epochs (0 disables).",
        ),
    ] = None,
    auto_class_weights: Annotated[
        Optional[bool],
        typer.Option(
            "--auto-class-weights/--no-auto-class-weights",
            help="Apply inverse class frequency weights in loss.",
        ),
    ] = None,
    pos_weight: Annotated[
        Optional[float],
        typer.Option(
            "--pos-weight",
            help="Loss multiplier for positive class (TB).",
        ),
    ] = None,
    eval_split: Annotated[
        str,
        typer.Option(
            "--eval-split",
            help="Split to evaluate on post-training ('val' or 'test').",
        ),
    ] = "test",
    augment: Annotated[
        Optional[bool],
        typer.Option(
            "--augment/--no-augment",
            help="Apply spatial and contrast augmentations during training.",
        ),
    ] = None,
    device: Annotated[
        Optional[str],
        typer.Option(
            "--device", help="Compute device ('auto', 'cuda', 'cpu')."
        ),
    ] = None,
    seed: Annotated[
        Optional[int],
        typer.Option("--seed", help="Random seed for reproducibility."),
    ] = None,
    num_workers: Annotated[
        Optional[int],
        typer.Option("--num-workers", help="Dataloader worker processes."),
    ] = None,
    dual_representation: Annotated[
        Optional[bool],
        typer.Option(
            "--dual-representation/--no-dual-representation",
            help="3-channel dual representation (raw, lung-isolated, edge).",
        ),
    ] = None,
    k8s_gpu: Annotated[
        bool,
        typer.Option(
            "--k8s-gpu/--no-k8s-gpu",
            help="Run training on cluster worker node with NVIDIA GPU.",
        ),
    ] = False,
    evaluate_test: Annotated[
        bool,
        typer.Option(
            "--evaluate-test/--no-evaluate-test",
            help="Evaluate on test split after training and generate report.",
        ),
    ] = True,
) -> None:
    """Train DINOv2 model on preprocessed chest X-ray images."""
    if k8s_gpu:
        args = _collect_k8s_args(
            config_path, data_dir, output_dir, report_dir, model_name,
            epochs, batch_size, learning_rate, backbone_lr,
            freeze_backbone, train_top_k, patience, auto_class_weights,
            pos_weight, augment, seed, num_workers, evaluate_test,
            eval_split, dual_representation,
        )
        try:
            _run_k8s_gpu_job(args)
            return
        except Exception as exc:
            _fail(exc)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    overrides = _collect_overrides(
        data_dir, output_dir, report_dir, model_name,
        epochs, batch_size, learning_rate, freeze_backbone,
        backbone_lr, train_top_k, patience, auto_class_weights,
        pos_weight, augment, device, seed, num_workers, dual_representation,
    )
    try:
        model_cfg = load_model_config(config_path, overrides=overrides)
    except (ConfigFileError, OSError, ValueError) as exc:
        _fail(exc)

    train_cfg = TrainConfig(
        data_dir=model_cfg.data_dir,
        output_dir=model_cfg.output_dir,
        model_name=model_cfg.model_name,
        epochs=model_cfg.epochs,
        batch_size=model_cfg.batch_size,
        learning_rate=model_cfg.learning_rate,
        freeze_backbone=model_cfg.freeze_backbone,
        train_top_k=model_cfg.train_top_k,
        patience=model_cfg.patience,
        auto_class_weights=model_cfg.auto_class_weights,
        pos_weight=model_cfg.pos_weight,
        backbone_lr=model_cfg.backbone_lr,
        augment=model_cfg.augment,
        device=model_cfg.device,
        num_workers=model_cfg.num_workers,
        seed=model_cfg.seed,
        dual_representation=model_cfg.dual_representation,
    )

    try:
        model, history = train_model(train_cfg)
        typer.echo(
            f"Training completed successfully. Saved to {train_cfg.output_dir}"
        )
        if evaluate_test:
            _run_post_train_eval(
                model, train_cfg, model_cfg.report_dir, history,
                split=eval_split,
            )
    except Exception as exc:
        _fail(exc)


@model_app.command("evaluate")
def evaluate_command(
    model_path: Annotated[
        Path,
        typer.Option(
            "--model-path",
            help="Path to trained model directory or checkpoint .pt file.",
        ),
    ] = Path("models/dinov2"),
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", help="Path to processed data directory."),
    ] = Path("data/processed"),
    split: Annotated[
        str,
        typer.Option(
            "--split", help="Dataset split to evaluate ('test', 'val')."
        ),
    ] = "test",
    report_dir: Annotated[
        Path,
        typer.Option(
            "--report-dir", help="Directory to save evaluation reports."
        ),
    ] = Path("reports"),
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Batch size for evaluation."),
    ] = 32,
    device: Annotated[
        str,
        typer.Option(
            "--device", help="Compute device ('auto', 'cuda', 'cpu')."
        ),
    ] = "auto",
    dual_representation: Annotated[
        bool,
        typer.Option(
            "--dual-representation/--no-dual-representation",
            help="3-channel dual representation (raw, lung-isolated, edge).",
        ),
    ] = True,
) -> None:
    """Evaluate a trained model checkpoint and output performance metrics."""
    try:
        dev = get_device(device)
        model = load_trained_model(model_path, device=dev)
        dataset = TBDataset(
            data_dir, split=split, dual_representation=dual_representation
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        metrics, y_true, _, y_prob = evaluate_model(model, dataloader, dev)
        detected_name = (
            "densenet121"
            if "densenet" in str(model_path).lower()
            else "facebook/dinov2-small"
        )
        report_file = save_evaluation_report(
            metrics=metrics,
            y_true=y_true,
            y_prob=y_prob,
            report_dir=report_dir,
            split_name=split,
            model_name=detected_name,
        )
        typer.echo(f"Report generated: {report_file}")
        typer.echo(json.dumps(metrics, indent=2))
    except Exception as exc:
        _fail(exc)


@model_app.command("predict")
def predict_command(
    image_path: Annotated[
        Path,
        typer.Option(
            "--image", help="Path to chest X-ray image for classification."
        ),
    ],
    model_path: Annotated[
        Path,
        typer.Option(
            "--model-path",
            help="Path to trained model directory or checkpoint .pt file.",
        ),
    ] = Path("models/dinov2"),
    device: Annotated[
        str,
        typer.Option(
            "--device", help="Compute device ('auto', 'cuda', 'cpu')."
        ),
    ] = "auto",
    json_output: Annotated[
        bool,
        typer.Option(
            "--json/--plain", help="Output predictions in JSON format."
        ),
    ] = False,
    dual_representation: Annotated[
        bool,
        typer.Option(
            "--dual-representation/--no-dual-representation",
            help="3-channel dual representation (raw, lung-isolated, edge).",
        ),
    ] = True,
) -> None:
    """Predict tuberculosis classification for an individual chest X-ray."""
    try:
        dev = get_device(device)
        model = load_trained_model(model_path, device=dev)
        result = predict_image(
            model,
            image_path,
            device=dev,
            dual_representation=dual_representation,
        )
        if json_output:
            typer.echo(json.dumps(result, indent=2))
        else:
            typer.echo(
                f"Prediction: {result['label']} "
                f"(confidence: {result['probability'] * 100:.2f}%)"
            )
    except Exception as exc:
        _fail(exc)


@model_app.command("list")
def list_models_command(
    models_dir: Annotated[
        Path,
        typer.Option(
            "--models-dir", help="Directory containing trained models."
        ),
    ] = Path("models"),
    reports_dir: Annotated[
        Path,
        typer.Option(
            "--reports-dir", help="Directory containing evaluation reports."
        ),
    ] = Path("reports"),
    split: Annotated[
        str,
        typer.Option("--split", help="Evaluation split ('test', 'val')."),
    ] = "test",
    json_output: Annotated[
        bool,
        typer.Option(
            "--json/--plain",
            help="Output models and benchmarks in JSON format.",
        ),
    ] = False,
) -> None:
    """List trained models, checkpoints, and performance benchmarks."""
    try:
        models = list_models(models_dir, reports_dir, split=split)
        if json_output:
            typer.echo(json.dumps(models, indent=2))
        else:
            typer.echo(format_models_table(models, split=split))
    except Exception as exc:
        _fail(exc)


@model_app.command("info")
def info_model_command(
    model_identifier: Annotated[
        str,
        typer.Argument(
            help="Model directory name or path (e.g. 'dinov2_518')."
        ),
    ],
    models_dir: Annotated[
        Path,
        typer.Option(
            "--models-dir", help="Directory containing trained models."
        ),
    ] = Path("models"),
    reports_dir: Annotated[
        Path,
        typer.Option(
            "--reports-dir", help="Directory containing evaluation reports."
        ),
    ] = Path("reports"),
    json_output: Annotated[
        bool,
        typer.Option(
            "--json/--plain",
            help="Output detailed model metadata in JSON format.",
        ),
    ] = False,
) -> None:
    """Inspect detailed metadata, checkpoints, and metrics for a model."""
    try:
        details = get_model_details(model_identifier, models_dir, reports_dir)
        if json_output:
            typer.echo(json.dumps(details, indent=2))
        else:
            typer.echo(format_model_info(details))
    except Exception as exc:
        _fail(exc)


@model_app.command("benchmark")
def benchmark_command(
    reports_dir: Annotated[
        Path,
        typer.Option(
            "--reports-dir", help="Directory containing evaluation reports."
        ),
    ] = Path("reports"),
    json_output: Annotated[
        bool,
        typer.Option(
            "--json/--plain",
            help="Output benchmark reference data in JSON format.",
        ),
    ] = False,
) -> None:
    """Display reference model benchmarks and clinical comparison analysis."""
    try:
        benchmarks = get_benchmark_reference(reports_dir)
        if json_output:
            clean_bench = dict(benchmarks)
            clean_bench.pop("raw_markdown", None)
            typer.echo(json.dumps(clean_bench, indent=2))
        else:
            typer.echo(format_benchmark_report(benchmarks))
    except Exception as exc:
        _fail(exc)


if __name__ == "__main__":
    app()
