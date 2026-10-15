"""Inference, evaluation, and reporting for DINOv2 TB models."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
import transformers
from transformers import AutoModelForImageClassification

from tb_detection_model.data.make_dataset import (
    compute_high_pass_residual,
    generate_lung_mask,
    normalize_xray,
)

LOGGER = logging.getLogger(__name__)

LABEL_TO_ID = {"Normal": 0, "TB": 1}
ID_TO_LABEL = {0: "Normal", 1: "TB"}


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    """Compute classification metrics from predictions and probabilities."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except (ValueError, IndexError):
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred).tolist()
    tn = cm[0][0] if len(cm) > 0 and len(cm[0]) > 0 else 0
    fp = cm[0][1] if len(cm) > 0 and len(cm[0]) > 1 else 0
    fn = cm[1][0] if len(cm) > 1 and len(cm[1]) > 0 else 0
    tp = cm[1][1] if len(cm) > 1 and len(cm[1]) > 1 else 0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    clf_rep = classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "TB"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "classification_report": clf_rep,
        "sample_count": int(len(y_true)),
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model on dataloader and compute performance metrics."""
    model.eval()
    all_targets: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            outputs = model(pixel_values=images)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

            tb_probs = probs[:, LABEL_TO_ID["TB"]].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_targets.extend(labels.numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(tb_probs.tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    return metrics, y_true, y_pred, y_prob


def _load_from_directory(
    path: Path,
    model_name: str,
    device: torch.device,
) -> nn.Module:
    """Load model from saved directory."""
    from tb_detection_model.models.train_model import (
        build_model,
        is_densenet_model,
    )

    for ckpt_name in ("best_model.pt", "final_model.pt", "model.pt"):
        candidate = path / ckpt_name
        if candidate.is_file():
            return _load_from_file(candidate, model_name, device)

    config_file = path / "config.json"
    is_densenet = is_densenet_model(model_name)
    if config_file.exists():
        try:
            with open(config_file, encoding="utf-8") as f:
                cfg = json.load(f)
            cfg_name = str(cfg.get("model_name", cfg.get("model_type", "")))
            if is_densenet_model(cfg_name):
                is_densenet = True
        except Exception:
            pass

    if is_densenet:
        model = build_model("densenet121", num_labels=2)
        pt_file = path / "model.pt"
        if pt_file.exists():
            state = torch.load(
                pt_file, map_location=device, weights_only=False
            )
            model.load_state_dict(state)
        return model

    return AutoModelForImageClassification.from_pretrained(
        path,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )


def _load_from_file(
    path: Path,
    model_name: str,
    device: torch.device,
) -> nn.Module:
    """Load model from checkpoint file."""
    from tb_detection_model.models.train_model import build_model

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    target_name = model_name
    if isinstance(checkpoint, dict) and "model_name" in checkpoint:
        target_name = checkpoint["model_name"]

    model = build_model(target_name, num_labels=2)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    return model


def load_trained_model(
    model_path_or_dir: Path | str,
    model_name: str = "facebook/dinov2-small",
    device: torch.device | None = None,
) -> nn.Module:
    """Load model weights from checkpoint file or saved directory."""
    path = Path(model_path_or_dir)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformers.logging.set_verbosity_error()

    if path.is_dir():
        model = _load_from_directory(path, model_name, device)
    elif path.is_file():
        model = _load_from_file(path, model_name, device)
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    model.to(device)
    model.eval()
    return model


def predict_image(
    model: nn.Module,
    image_path: Path | str,
    device: torch.device | None = None,
    target_size: tuple[int, int] | None = None,
    dual_representation: bool = True,
) -> dict[str, Any]:
    """Classify a single chest X-ray image file."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image at {path}")

    if target_size is not None and image.shape != target_size:
        image = cv2.resize(
            image,
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_AREA,
        )
    elif max(image.shape) > 518:
        image = cv2.resize(
            image, (518, 518), interpolation=cv2.INTER_AREA
        )

    if dual_representation:
        ch0 = normalize_xray(image)
        mask = generate_lung_mask(image)
        lung_isolated = cv2.bitwise_and(image, image, mask=mask)
        ch1 = normalize_xray(lung_isolated)
        edge_map = compute_high_pass_residual(image)
        ch2 = normalize_xray(edge_map)
        tensor = (
            torch.from_numpy(np.stack([ch0, ch1, ch2], axis=0))
            .unsqueeze(0)
            .float()
            .to(device)
        )
    else:
        normalized = normalize_xray(image)
        tensor = (
            torch.from_numpy(normalized)
            .unsqueeze(0)
            .repeat(3, 1, 1)
            .unsqueeze(0)
            .float()
            .to(device)
        )

    model.eval()
    with torch.no_grad():
        output = model(pixel_values=tensor)
        probs = torch.softmax(output.logits, dim=1).squeeze(0).cpu().numpy()
        pred_id = int(np.argmax(probs))

    pred_label = ID_TO_LABEL[pred_id]
    return {
        "image_path": str(path),
        "label": pred_label,
        "probability": float(probs[pred_id]),
        "probabilities": {
            "Normal": float(probs[LABEL_TO_ID["Normal"]]),
            "TB": float(probs[LABEL_TO_ID["TB"]]),
        },
    }


def plot_confusion_matrix(
    cm: list[list[int]],
    output_path: Path,
    labels: tuple[str, ...] = ("Normal", "TB"),
) -> None:
    """Save confusion matrix heatmap plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_arr = np.array(cm)
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    fig.colorbar(im)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=12)

    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j,
                i,
                f"{cm_arr[i, j]}",
                ha="center",
                va="center",
                color="white" if cm_arr[i, j] > thresh else "black",
                fontsize=14,
            )

    ax.set_title("Confusion Matrix", fontsize=14)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    auc_score: float,
) -> None:
    """Save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"ROC curve (AUC = {auc_score:.4f})",
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_training_curves(
    history: dict[str, list[float]],
    output_path: Path,
) -> None:
    """Save training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "s-", label="Val Loss")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training & Validation Loss", fontsize=13)
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.plot(epochs, history["train_acc"], "o-", label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "s-", label="Val Acc")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Training & Validation Accuracy", fontsize=13)
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_evaluation_report(
    metrics: dict[str, Any],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    history: dict[str, list[float]] | None = None,
    report_dir: Path | str = "reports",
    split_name: str = "test",
    model_name: str = "facebook/dinov2-small",
) -> Path:
    """Save evaluation JSON, Markdown report, and figures."""
    report_path = Path(report_dir)
    figures_path = report_path / "figures"
    report_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    model_slug = (
        "densenet121"
        if "densenet" in model_name.lower()
        else "dinov2"
        if "dino" in model_name.lower()
        else model_name.split("/")[-1].replace("-", "_")
    )
    display_title = (
        "DenseNet121"
        if "densenet" in model_name.lower()
        else "DINOv2"
        if "dino" in model_name.lower()
        else model_name
    )

    # Save metrics JSON (both model-specific and split default)
    json_path = report_path / f"metrics_{model_slug}_{split_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(
        report_path / f"metrics_{split_name}.json", "w", encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=2)

    # Generate figures
    cm_fig_path = (
        figures_path / f"confusion_matrix_{model_slug}_{split_name}.png"
    )
    plot_confusion_matrix(metrics["confusion_matrix"], cm_fig_path)
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        figures_path / f"confusion_matrix_{split_name}.png",
    )

    roc_fig_path = figures_path / f"roc_curve_{model_slug}_{split_name}.png"
    plot_roc_curve(y_true, y_prob, roc_fig_path, metrics["roc_auc"])
    plot_roc_curve(
        y_true,
        y_prob,
        figures_path / f"roc_curve_{split_name}.png",
        metrics["roc_auc"],
    )

    if history and history.get("train_loss"):
        history_fig_path = figures_path / f"training_curves_{model_slug}.png"
        plot_training_curves(history, history_fig_path)
        plot_training_curves(history, figures_path / "training_curves.png")

    # Generate markdown report
    md_path = report_path / f"{model_slug}_evaluation_report_{split_name}.md"
    cm = metrics["confusion_matrix"]
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    header = (
        f"# {display_title} Tuberculosis Detection - "
        f"{split_name.capitalize()} Evaluation Report"
    )
    acc_pct = metrics['accuracy'] * 100
    report_text = classification_report(
        y_true,
        (y_prob >= 0.5).astype(int),
        target_names=["Normal", "TB"],
        zero_division=0,
    )

    lines = [
        header,
        "",
        f"- **Model Architecture**: `{model_name}`",
        f"- **Evaluation Split**: `{split_name}`",
        f"- **Total Samples**: {metrics['sample_count']}",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **Accuracy** | {metrics['accuracy']:.4f} ({acc_pct:.2f}%) |",
        f"| **Precision (TB)** | {metrics['precision']:.4f} |",
        f"| **Recall / Sensitivity (TB)** | {metrics['recall']:.4f} |",
        f"| **Specificity (Normal)** | {specificity:.4f} |",
        f"| **F1-Score** | {metrics['f1']:.4f} |",
        f"| **ROC-AUC** | {metrics['roc_auc']:.4f} |",
        "",
        "## Confusion Matrix",
        "",
        "| | Predicted Normal | Predicted TB | Total True |",
        "|---|---|---|---|",
        f"| **True Normal** | {tn} (TN) | {fp} (FP) | {tn + fp} |",
        f"| **True TB** | {fn} (FN) | {tp} (TP) | {fn + tp} |",
        (
            f"| **Total Predicted** | {tn + fn} | {fp + tp} | "
            f"{metrics['sample_count']} |"
        ),
        "",
        "## Classification Breakdown",
        "",
        "```text",
        f"{report_text}",
        "```",
        "## Visualizations",
        "",
        f"- Confusion Matrix: `figures/{cm_fig_path.name}`",
        f"- ROC Curve: `figures/{roc_fig_path.name}`",
    ]
    if history and history.get("train_loss"):
        lines.append(f"- Training Curves: `figures/{history_fig_path.name}`")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    LOGGER.info("Saved evaluation report to %s", md_path)
    return md_path
