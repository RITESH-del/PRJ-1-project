"""Training pipeline for DINOv2 chest X-ray classification."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as tv_models
from transformers import AutoModelForImageClassification

from tb_detection_model.data.make_dataset import (
    DEFAULT_PROCESSED_DIR,
    compute_high_pass_residual,
    generate_lung_mask,
    normalize_xray,
)

LOGGER = logging.getLogger(__name__)

LABEL_TO_ID = {"Normal": 0, "TB": 1}
ID_TO_LABEL = {0: "Normal", 1: "TB"}


@dataclass
class TrainConfig:
    """Configuration for model training."""

    data_dir: Path = DEFAULT_PROCESSED_DIR
    output_dir: Path = Path("models/dinov2")
    model_name: str = "facebook/dinov2-small"
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-4
    freeze_backbone: bool = False
    train_top_k: int = 2
    backbone_lr: float | None = None
    augment: bool = True
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    device: str = "auto"
    num_workers: int = 2
    seed: int = 42
    patience: int = 3
    min_delta: float = 1e-4
    auto_class_weights: bool = False
    pos_weight: float | None = None
    class_weights: list[float] | None = None
    dual_representation: bool = True
    target_size: tuple[int, int] | None = None


class EarlyStopping:
    """Early stopping handler when validation loss plateaus."""
    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False

    def step(self, score: float) -> bool:
        """Evaluate score; return True if improved, False otherwise."""
        if self.patience <= 0:
            return True
        if self.best_score is None:
            self.best_score = score
            return True

        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return False


def compute_class_weights(
    dataset: TBDataset,
    pos_weight: float | None = None,
    explicit_weights: list[float] | None = None,
) -> torch.Tensor | None:
    """Compute class weights for cross-entropy loss."""
    if explicit_weights is not None:
        return torch.tensor(explicit_weights, dtype=torch.float32)

    if pos_weight is not None:
        return torch.tensor([1.0, float(pos_weight)], dtype=torch.float32)

    counts = [0, 0]
    for _, label_id in dataset.samples:
        counts[label_id] += 1
    total = sum(counts)
    if counts[0] == 0 or counts[1] == 0:
        return None
    w0 = total / (2.0 * counts[0])
    w1 = total / (2.0 * counts[1])
    return torch.tensor([w0, w1], dtype=torch.float32)


class TBDataset(Dataset):

    def __init__(
        self,
        data_dir: Path | str,
        split: str = "train",
        normalize: bool = True,
        augment: bool = False,
        target_size: tuple[int, int] | None = None,
        dual_representation: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.normalize = normalize
        self.augment = augment
        self.target_size = target_size
        self.dual_representation = dual_representation
        self.samples: list[tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory does not exist: {split_dir}"
            )

        for label_name, label_id in sorted(LABEL_TO_ID.items()):
            label_dir = split_dir / label_name
            if not label_dir.is_dir():
                continue
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                for img_path in sorted(label_dir.glob(ext)):
                    self.samples.append((img_path, label_id))

        if not self.samples:
            raise ValueError(
                f"No valid images found in {split_dir}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        img_path, label_id = self.samples[idx]
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image at {img_path}")

        if self.target_size is not None and image.shape != self.target_size:
            target_h, target_w = self.target_size
            image = cv2.resize(
                image, (target_w, target_h), interpolation=cv2.INTER_AREA
            )
        img_h, img_w = image.shape[:2]

        if self.split == "train" and self.augment:
            angle = float(np.random.uniform(-7.0, 7.0))
            scale = float(np.random.uniform(0.95, 1.05))
            center = (float(img_w) / 2.0, float(img_h) / 2.0)
            rot = cv2.getRotationMatrix2D(center, angle, scale)
            image = cv2.warpAffine(
                image,
                rot,
                (img_w, img_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            contrast = float(np.random.uniform(0.85, 1.15))
            brightness = float(np.random.uniform(-15.0, 15.0))
            image = np.clip(
                image.astype(np.float32) * contrast + brightness, 0, 255
            ).astype(np.uint8)
        if self.normalize:
            ch0 = normalize_xray(image)
        else:
            ch0 = image.astype(np.float32) / 255.0

        if self.dual_representation:
            mask = generate_lung_mask(image)
            lung_isolated = cv2.bitwise_and(image, image, mask=mask)
            edge_map = compute_high_pass_residual(image)
            if self.normalize:
                ch1 = normalize_xray(lung_isolated)
                ch2 = normalize_xray(edge_map)
            else:
                ch1 = lung_isolated.astype(np.float32) / 255.0
                ch2 = edge_map.astype(np.float32) / 255.0
            tensor_3ch = torch.from_numpy(
                np.stack([ch0, ch1, ch2], axis=0)
            )
        else:
            tensor_3ch = (
                torch.from_numpy(ch0).unsqueeze(0).repeat(3, 1, 1)
            )

        return tensor_3ch.float(), label_id, str(img_path)


class DenseNet121Classifier(nn.Module):
    """DenseNet121 baseline wrapper adhering to classification contract."""

    def __init__(
        self,
        num_labels: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        weights = (
            tv_models.DenseNet121_Weights.DEFAULT if pretrained else None
        )
        self.densenet = tv_models.densenet121(weights=weights)
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(in_features, num_labels)
        self.classifier = self.densenet.classifier

        if freeze_backbone:
            for param in self.densenet.features.parameters():
                param.requires_grad = False

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        x = pixel_values if pixel_values is not None else images
        if x is None:
            raise ValueError(
                "Forward pass requires pixel_values or images tensor"
            )
        logits = self.densenet(x)
        return SimpleNamespace(logits=logits)

    def save_pretrained(self, save_directory: Path | str) -> None:
        """Save model weights and configuration to directory."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path / "model.pt")
        config_data = {
            "model_type": "densenet121",
            "model_name": "densenet121",
            "num_labels": 2,
            "id2label": ID_TO_LABEL,
            "label2id": LABEL_TO_ID,
        }
        with open(save_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)


def is_densenet_model(model_name: str) -> bool:
    """Check if model name indicates a DenseNet121 architecture."""
    name = model_name.lower()
    return (
        "densenet121" in name
        or "densenet-121" in name
        or "densenet" in name
    )


def build_model(
    model_name: str = "facebook/dinov2-small",
    num_labels: int = 2,
    freeze_backbone: bool = False,
    train_top_k: int = 2,
) -> nn.Module:
    """Build and initialize model (DINOv2 or DenseNet121)."""
    if is_densenet_model(model_name):
        return DenseNet121Classifier(
            num_labels=num_labels,
            pretrained=True,
            freeze_backbone=freeze_backbone,
        )

    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    if freeze_backbone or train_top_k <= 0:
        for name, param in model.named_parameters():
            param.requires_grad = ("classifier" in name)
    else:
        num_layers = 12
        if hasattr(model, "dinov2") and hasattr(model.dinov2, "encoder"):
            num_layers = len(model.dinov2.encoder.layer)
        start_idx = max(0, num_layers - train_top_k)
        unfreeze_prefixes = tuple(
            f"dinov2.encoder.layer.{idx}."
            for idx in range(start_idx, num_layers)
        )
        for name, param in model.named_parameters():
            is_target = (
                "classifier" in name
                or "dinov2.layernorm" in name
                or any(name.startswith(p) for p in unfreeze_prefixes)
            )
            param.requires_grad = is_target
    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    max_grad_norm: float = 1.0,
) -> tuple[float, float]:
    """Train the model for one epoch; return (average_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_cuda = device.type == "cuda"

    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_cuda):
            outputs = model(pixel_values=images)
            logits = outputs.logits
            loss = criterion(logits, labels)

        if scaler is not None and use_cuda:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate model on dataloader; return (average_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    use_cuda = device.type == "cuda"
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                outputs = model(pixel_values=images)
                logits = outputs.logits
                loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def get_device(requested: str = "auto") -> torch.device:
    """Resolve compute device based on request and availability."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _setup_optimizer(
    model: nn.Module, config: TrainConfig
) -> torch.optim.Optimizer:
    """Create optimizer with differential learning rates for fine-tuning."""
    if config.freeze_backbone or config.train_top_k <= 0:
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params, lr=config.learning_rate, weight_decay=config.weight_decay
        )
    backbone_lr = (
        config.backbone_lr
        if config.backbone_lr is not None
        else config.learning_rate * 0.1
    )
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        param_groups.append(
            {"params": head_params, "lr": config.learning_rate}
        )

    return torch.optim.AdamW(
        param_groups,
        weight_decay=config.weight_decay,
    )


def _build_criterion(
    config: TrainConfig,
    train_ds: TBDataset,
    device: torch.device,
) -> nn.Module:
    """Build cross-entropy loss with optional class weighting."""
    weights: torch.Tensor | None = None
    if config.auto_class_weights:
        weights = compute_class_weights(train_ds)
        if weights is not None:
            weights = weights.to(device)
            LOGGER.info(
                "Using auto-computed class weights: %s", weights.tolist()
            )
    elif config.pos_weight is not None:
        weights = compute_class_weights(
            train_ds, pos_weight=config.pos_weight
        )
        if weights is not None:
            weights = weights.to(device)
            LOGGER.info(
                "Using positive class weight (TB=%.2f): %s",
                config.pos_weight,
                weights.tolist(),
            )
    elif config.class_weights is not None:
        weights = compute_class_weights(
            train_ds, explicit_weights=config.class_weights
        )
        if weights is not None:
            weights = weights.to(device)
            LOGGER.info("Using explicit class weights: %s", weights.tolist())

    return nn.CrossEntropyLoss(
        weight=weights, label_smoothing=config.label_smoothing
    )


def train_model(
    config: TrainConfig,
) -> tuple[nn.Module, dict[str, list[float]]]:
    """Execute complete training pipeline and save best model."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = get_device(config.device)
    LOGGER.info("Using compute device: %s", device)

    train_ds = TBDataset(
        config.data_dir,
        split="train",
        augment=config.augment,
        target_size=config.target_size,
        dual_representation=config.dual_representation,
    )
    val_ds = TBDataset(
        config.data_dir,
        split="val",
        augment=False,
        target_size=config.target_size,
        dual_representation=config.dual_representation,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(
        model_name=config.model_name,
        num_labels=2,
        freeze_backbone=config.freeze_backbone,
        train_top_k=config.train_top_k,
    ).to(device)

    optimizer = _setup_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    criterion = _build_criterion(config, train_ds, device)
    early_stopping = EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta,
        mode="min",
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = config.output_dir / "best_model.pt"
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    LOGGER.info(
        "Starting training for %d epochs (batch_size=%d, lr=%.2e)",
        config.epochs,
        config.batch_size,
        config.learning_rate,
    )

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler=scaler,
        )
        scheduler.step()
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        LOGGER.info(
            "Epoch %02d/%02d - Train Loss: %.4f, Acc: %.4f | "
            "Val Loss: %.4f, Acc: %.4f",
            epoch,
            config.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        is_best = early_stopping.step(val_loss)
        if is_best:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "model_name": config.model_name,
                },
                best_model_path,
            )
            LOGGER.info(
                "Saved new best model checkpoint to %s", best_model_path
            )

        if early_stopping.early_stop:
            LOGGER.info(
                "Early stopping triggered at epoch %02d (patience=%d)",
                epoch,
                config.patience,
            )
            break
    final_model_path = config.output_dir / "final_model.pt"
    torch.save(
        {
            "epoch": config.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "model_name": config.model_name,
        },
        final_model_path,
    )
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        LOGGER.info(
            "Restored best checkpoint from epoch %d (val_acc=%.4f)",
            checkpoint["epoch"],
            checkpoint["val_acc"],
        )

    # Save Hugging Face pretrained directory format for easy loading
    model.save_pretrained(config.output_dir)

    return model, history


def main() -> None:
    """Entry point for standalone execution."""
    logging.basicConfig(level=logging.INFO)
    config = TrainConfig()
    train_model(config)


if __name__ == "__main__":
    main()
