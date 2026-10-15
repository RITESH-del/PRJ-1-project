"""Model metadata inspection, discovery, and benchmark utilities."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass
class CheckpointMeta:
    """Metadata for an individual model checkpoint file."""

    name: str
    path: str
    size_bytes: int
    size_mb: float
    epoch: int | None = None
    val_loss: float | None = None
    val_acc: float | None = None
    model_name: str | None = None


def find_metrics_file(
    reports_dir: Path, model_name: str, split: str = "test"
) -> Path | None:
    """Find the metrics JSON file corresponding to a model and split."""
    candidates = [
        reports_dir / model_name / f"metrics_{split}.json",
        reports_dir / model_name / f"metrics_{model_name}_{split}.json",
        reports_dir / f"metrics_{model_name}_{split}.json",
    ]
    if model_name in ("dinov2", "default"):
        candidates.append(reports_dir / f"metrics_{split}.json")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def inspect_checkpoint(ckpt_path: Path) -> CheckpointMeta:
    """Inspect checkpoint file size and embedded training metadata."""
    stat = ckpt_path.stat()
    size_bytes = stat.st_size
    size_mb = round(size_bytes / (1024 * 1024), 2)

    meta = CheckpointMeta(
        name=ckpt_path.name,
        path=str(ckpt_path),
        size_bytes=size_bytes,
        size_mb=size_mb,
    )

    if ckpt_path.suffix == ".pt":
        try:
            import torch

            data = torch.load(
                ckpt_path, map_location="cpu", weights_only=False
            )
            if isinstance(data, dict):
                meta.epoch = data.get("epoch")
                meta.val_loss = data.get("val_loss")
                meta.val_acc = data.get("val_acc")
                meta.model_name = data.get("model_name")
        except Exception as exc:
            LOGGER.debug(
                "Could not parse torch checkpoint %s: %s", ckpt_path, exc
            )

    return meta


def load_metrics(metrics_file: Path) -> dict[str, Any]:
    """Load and normalize model evaluation metrics from JSON file."""
    with open(metrics_file, encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)

    cm = raw.get("confusion_matrix", [[0, 0], [0, 0]])
    tn = cm[0][0] if len(cm) > 0 and len(cm[0]) > 0 else 0
    fp = cm[0][1] if len(cm) > 0 and len(cm[0]) > 1 else 0
    fn = cm[1][0] if len(cm) > 1 and len(cm[1]) > 0 else 0
    tp = cm[1][1] if len(cm) > 1 and len(cm[1]) > 1 else 0

    specificity = raw.get("specificity")
    if specificity is None:
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    sensitivity = raw.get("sensitivity", raw.get("recall", 0.0))
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "accuracy": raw.get("accuracy", 0.0),
        "sensitivity": sensitivity,
        "recall": sensitivity,
        "specificity": specificity,
        "precision": raw.get("precision", 0.0),
        "f1": raw.get("f1", 0.0),
        "roc_auc": raw.get("roc_auc", 0.0),
        "sample_count": raw.get("sample_count", tn + fp + fn + tp),
        "confusion_matrix": cm,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "false_negative_rate": fnr,
        "metrics_file": str(metrics_file),
    }


def _detect_architecture(model_dir: Path, cfg: dict[str, Any]) -> str:
    """Detect human-readable model architecture from config and path."""
    name_hint = model_dir.name.lower()
    if "densenet" in name_hint or cfg.get("model_type") == "densenet121":
        return "densenet121"
    arch = cfg.get("architectures", [])
    if arch and "dinov2" in arch[0].lower():
        return "facebook/dinov2-small"
    if cfg.get("model_type") == "dinov2":
        return "facebook/dinov2-small"
    return cfg.get("model_name", "facebook/dinov2-small")


def _detect_resolution(model_dir: Path, cfg: dict[str, Any]) -> str:
    """Determine input image resolution for the model."""
    if "518" in model_dir.name:
        return "518x518"
    if "target_size" in cfg:
        ts = cfg["target_size"]
        return f"{ts[0]}x{ts[1]}"
    return "224x224"


def collect_model_summary(
    model_dir: Path, reports_dir: Path, split: str = "test"
) -> dict[str, Any]:
    """Collect summary metadata and metrics for a given model directory."""
    cfg_path = model_dir / "config.json"
    cfg: dict[str, Any] = {}
    if cfg_path.is_file():
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as exc:
            LOGGER.debug("Could not read config %s: %s", cfg_path, exc)

    ckpts = [
        inspect_checkpoint(p)
        for p in sorted(model_dir.iterdir())
        if p.is_file() and p.suffix in (".pt", ".safetensors", ".bin")
    ]

    best_ckpt = next((c for c in ckpts if c.name == "best_model.pt"), None)
    metrics_file = find_metrics_file(reports_dir, model_dir.name, split)
    metrics = load_metrics(metrics_file) if metrics_file else None

    arch = _detect_architecture(model_dir, cfg)
    res = _detect_resolution(model_dir, cfg)

    return {
        "name": model_dir.name,
        "path": str(model_dir),
        "architecture": arch,
        "resolution": res,
        "config": cfg,
        "checkpoints": [c.__dict__ for c in ckpts],
        "checkpoint_count": len(ckpts),
        "best_epoch": best_ckpt.epoch if best_ckpt else None,
        "best_val_loss": best_ckpt.val_loss if best_ckpt else None,
        "best_val_acc": best_ckpt.val_acc if best_ckpt else None,
        "split": split,
        "metrics": metrics,
    }


def list_models(
    models_dir: Path, reports_dir: Path, split: str = "test"
) -> list[dict[str, Any]]:
    """Discover all trained models and summarize their performance metrics."""
    if not models_dir.is_dir():
        return []

    summaries = []
    for entry in sorted(models_dir.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            summary = collect_model_summary(entry, reports_dir, split)
            summaries.append(summary)

    def sort_key(item: dict[str, Any]) -> tuple[float, float, str]:
        m = item.get("metrics")
        if m:
            return (
                m.get("roc_auc", 0.0),
                m.get("accuracy", 0.0),
                item["name"],
            )
        return (0.0, 0.0, item["name"])

    summaries.sort(key=sort_key, reverse=True)
    return summaries


def get_model_details(
    model_identifier: str, models_dir: Path, reports_dir: Path
) -> dict[str, Any]:
    """Retrieve in-depth configuration, multi-split metrics, and figures."""
    target_dir = Path(model_identifier)
    if not target_dir.is_dir():
        target_dir = models_dir / model_identifier

    if not target_dir.is_dir():
        raise FileNotFoundError(
            f"Model directory '{model_identifier}' not found in "
            f"'{models_dir}'."
        )

    summary_test = collect_model_summary(target_dir, reports_dir, "test")
    summary_val = collect_model_summary(target_dir, reports_dir, "val")

    figs_dir = reports_dir / target_dir.name / "figures"
    figures: list[str] = []
    if figs_dir.is_dir():
        figures = [str(f) for f in sorted(figs_dir.glob("*.png"))]

    rep_candidates = [
        reports_dir / target_dir.name / (
            f"{target_dir.name}_evaluation_report_test.md"
        ),
        reports_dir / target_dir.name / "dinov2_evaluation_report_test.md",
        reports_dir / target_dir.name / (
            "densenet121_evaluation_report_test.md"
        ),
        reports_dir / f"{target_dir.name}_evaluation_report_test.md",
    ]
    report_file = next((str(r) for r in rep_candidates if r.is_file()), None)

    return {
        "name": target_dir.name,
        "path": str(target_dir),
        "architecture": summary_test["architecture"],
        "resolution": summary_test["resolution"],
        "config": summary_test["config"],
        "checkpoints": summary_test["checkpoints"],
        "best_epoch": summary_test["best_epoch"],
        "best_val_loss": summary_test["best_val_loss"],
        "best_val_acc": summary_test["best_val_acc"],
        "metrics": {
            "test": summary_test.get("metrics"),
            "val": summary_val.get("metrics"),
        },
        "figures": figures,
        "report_markdown": report_file,
    }


def get_benchmark_reference(reports_dir: Path) -> dict[str, Any]:
    """Load or synthesize reference benchmarks and comparative results."""
    comp_md = reports_dir / "model_comparison_report.md"
    raw_markdown = (
        comp_md.read_text(encoding="utf-8") if comp_md.is_file() else ""
    )

    benchmarks_test = [
        {
            "model": "dinov2_518",
            "architecture": "DINOv2 ViT-S/14",
            "resolution": "518x518 (Dual-Rep)",
            "accuracy": 0.7674,
            "sensitivity": 0.7857,
            "specificity": 0.7485,
            "f1": 0.7742,
            "roc_auc": 0.8307,
            "missed_tb": "36 / 168",
        },
        {
            "model": "densenet121_518",
            "architecture": "DenseNet121",
            "resolution": "518x518 (CNN)",
            "accuracy": 0.6677,
            "sensitivity": 0.7619,
            "specificity": 0.5706,
            "f1": 0.6995,
            "roc_auc": 0.7214,
            "missed_tb": "40 / 168",
        },
        {
            "model": "dinov2_top2",
            "architecture": "DINOv2 ViT-S/14",
            "resolution": "224x224 (Top-2 unfreezing)",
            "accuracy": 0.6284,
            "sensitivity": 0.6369,
            "specificity": 0.6196,
            "f1": 0.6350,
            "roc_auc": 0.6979,
            "missed_tb": "61 / 168",
        },
        {
            "model": "dinov2_frozen",
            "architecture": "DINOv2 ViT-S/14",
            "resolution": "224x224 (Frozen backbone)",
            "accuracy": 0.6254,
            "sensitivity": 0.4762,
            "specificity": 0.7791,
            "f1": 0.5634,
            "roc_auc": 0.7140,
            "missed_tb": "88 / 168",
        },
        {
            "model": "dinov2_top4",
            "architecture": "DINOv2 ViT-S/14",
            "resolution": "224x224 (Top-4 unfreezing)",
            "accuracy": 0.5952,
            "sensitivity": 0.7857,
            "specificity": 0.3988,
            "f1": 0.6633,
            "roc_auc": 0.6677,
            "missed_tb": "36 / 168",
        },
        {
            "model": "densenet121",
            "architecture": "DenseNet121",
            "resolution": "224x224 (Supervised baseline)",
            "accuracy": 0.5891,
            "sensitivity": 0.3274,
            "specificity": 0.8589,
            "f1": 0.4472,
            "roc_auc": 0.7046,
            "missed_tb": "113 / 168",
        },
    ]

    insights = [
        (
            "High-Resolution Token Density (518x518): At 518x518 with "
            "patch size 14, DINOv2 generates 1,369 tokens (5.35x higher than "
            "224x224), capturing apical opacities and lifting test "
            "ROC-AUC to 0.8307."
        ),
        (
            "Clinical Triage Superiority: In screening, false negatives are "
            "critical. DINOv2-518 reduces missed cases to 36/168 (78.57% "
            "sensitivity), while DenseNet121-224 misses 113/168 (32.74% "
            "sensitivity)."
        ),
        (
            "Differential Learning Rates: lr_head=5e-4 and lr_backbone=5e-5 "
            "prevent representation collapse while adapting top ViT blocks."
        ),
    ]

    return {
        "dataset_splits": {
            "train": "N=7,138 (NIH ChestX-ray14 + Montgomery)",
            "val": "N=331 (Shenzhen Hospital, 0% patient leakage)",
            "test": "N=331 (Shenzhen Hospital, 0% patient leakage)",
        },
        "benchmarks_test": benchmarks_test,
        "clinical_insights": insights,
        "markdown_report_path": str(comp_md) if comp_md.is_file() else None,
        "raw_markdown": raw_markdown,
    }


def format_models_table(
    models: list[dict[str, Any]], split: str = "test"
) -> str:
    """Format model summaries into a clean tabular ASCII display."""
    if not models:
        return f"No trained models found for evaluation split '{split}'."

    headers = [
        "Model Name",
        "Architecture",
        "Resolution",
        "Checkpoints",
        "Accuracy",
        "Sens (TB)",
        "Spec (Norm)",
        "F1",
        "ROC-AUC",
    ]

    rows: list[list[str]] = []
    for m in models:
        metrics = m.get("metrics")
        ckpts = f"{m['checkpoint_count']} files"
        if metrics:
            acc = f"{metrics['accuracy'] * 100:.2f}%"
            sens = f"{metrics['sensitivity'] * 100:.2f}%"
            spec = f"{metrics['specificity'] * 100:.2f}%"
            f1 = f"{metrics['f1']:.4f}"
            auc = f"{metrics['roc_auc']:.4f}"
        else:
            acc, sens, spec, f1, auc = ("N/A", "N/A", "N/A", "N/A", "N/A")

        rows.append(
            [
                m["name"],
                m["architecture"],
                m["resolution"],
                ckpts,
                acc,
                sens,
                spec,
                f1,
                auc,
            ]
        )

    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(val))

    def make_row(items: list[str]) -> str:
        return "  ".join(
            item.ljust(col_widths[idx]) for idx, item in enumerate(items)
        )

    separator = "  ".join("-" * w for w in col_widths)
    output_lines = [
        f"Available Models & Performance Benchmarks (Split: {split}):",
        "",
        make_row(headers),
        separator,
    ]
    for r in rows:
        output_lines.append(make_row(r))

    return "\n".join(output_lines)


def format_model_info(info: dict[str, Any]) -> str:
    """Format single model details into an inspection report card."""
    lines = [
        f"Model: {info['name']}",
        f"Path: {info['path']}",
        f"Architecture: {info['architecture']}",
        f"Resolution: {info['resolution']}",
        "",
        "Checkpoints:",
    ]

    for ckpt in info.get("checkpoints", []):
        ep_info = f" | Epoch: {ckpt['epoch']}" if ckpt.get("epoch") else ""
        acc_info = (
            f" | Val Acc: {ckpt['val_acc'] * 100:.2f}%"
            if ckpt.get("val_acc")
            else ""
        )
        lines.append(
            f"  - {ckpt['name']} ({ckpt['size_mb']} MB){ep_info}{acc_info}"
        )

    for split in ("test", "val"):
        m = info.get("metrics", {}).get(split)
        if m:
            lines.extend(
                [
                    "",
                    f"{split.upper()} Split Metrics (N={m['sample_count']}):",
                    f"  Accuracy:    {m['accuracy'] * 100:.2f}%",
                    f"  Sensitivity: {m['sensitivity'] * 100:.2f}% "
                    f"({m['tp']}/{m['tp'] + m['fn']} TB detected)",
                    f"  Specificity: {m['specificity'] * 100:.2f}% "
                    f"({m['tn']}/{m['tn'] + m['fp']} Normal detected)",
                    f"  Precision:   {m['precision'] * 100:.2f}%",
                    f"  F1-Score:    {m['f1']:.4f}",
                    f"  ROC-AUC:     {m['roc_auc']:.4f}",
                    f"  Confusion:   TN={m['tn']}, FP={m['fp']}, "
                    f"FN={m['fn']}, TP={m['tp']}",
                    f"  Missed TB:   {m['fn']} cases (FNR: "
                    f"{m['false_negative_rate'] * 100:.2f}%)",
                ]
            )

    figures = info.get("figures", [])
    if figures:
        lines.extend(["", "Figures & Diagnostics:"])
        for fig in figures:
            lines.append(f"  - {fig}")

    if info.get("report_markdown"):
        lines.extend(["", f"Evaluation Report: {info['report_markdown']}"])

    return "\n".join(lines)


def format_benchmark_report(benchmark: dict[str, Any]) -> str:
    """Format benchmark reference data into human-readable comparison text."""
    lines = [
        "===========================================================",
        " Tuberculosis Detection: Reference Model Benchmark Summary",
        "===========================================================",
        "",
        "Evaluation Cohort:",
    ]
    for split, desc in benchmark.get("dataset_splits", {}).items():
        lines.append(f"  - {split.capitalize():5s}: {desc}")

    lines.extend(
        [
            "",
            "Held-Out Test Set Performance Benchmark "
            "(N=331, Zero Patient Leakage):",
            "",
        ]
    )

    headers = [
        "Model",
        "Architecture",
        "Accuracy",
        "Sens (TB)",
        "Spec (Norm)",
        "F1",
        "ROC-AUC",
        "Missed TB",
    ]
    rows = []
    for b in benchmark.get("benchmarks_test", []):
        rows.append(
            [
                b["model"],
                b["architecture"],
                f"{b['accuracy'] * 100:.2f}%",
                f"{b['sensitivity'] * 100:.2f}%",
                f"{b['specificity'] * 100:.2f}%",
                f"{b['f1']:.4f}",
                f"{b['roc_auc']:.4f}",
                b["missed_tb"],
            ]
        )

    col_widths = [len(h) for h in headers]
    for r in rows:
        for idx, val in enumerate(r):
            col_widths[idx] = max(col_widths[idx], len(val))

    def make_row(items: list[str]) -> str:
        return "  ".join(
            item.ljust(col_widths[idx]) for idx, item in enumerate(items)
        )

    lines.append(make_row(headers))
    lines.append("  ".join("-" * w for w in col_widths))
    for r in rows:
        lines.append(make_row(r))

    lines.extend(["", "Key Clinical & Engineering Insights:"])
    for idx, insight in enumerate(benchmark.get("clinical_insights", []), 1):
        lines.append(f"  {idx}. {insight}")

    return "\n".join(lines)
