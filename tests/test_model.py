"""Unit tests for DINOv2 TB classification models and datasets."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from tb_detection_model.models.predict_model import (
    compute_metrics,
    predict_image,
)
from tb_detection_model.models.train_model import (
    LABEL_TO_ID,
    TBDataset,
    EarlyStopping,
    TrainConfig,
    _setup_optimizer,
    build_model,
    compute_class_weights,
    train_one_epoch,
)


@pytest.fixture
def dummy_processed_dataset(tmp_path: Path) -> Path:
    """Create a minimal processed dataset structure with train/val/test."""
    root = tmp_path / "processed"
    for split in ("train", "val", "test"):
        for label in ("TB", "Normal"):
            folder = root / split / label
            folder.mkdir(parents=True)
            for i in range(2):
                img = np.full((224, 224), 100 + i * 20, dtype=np.uint8)
                cv2.imwrite(str(folder / f"sample_{i}.png"), img)
    return root


def test_tb_dataset_len_and_item(dummy_processed_dataset: Path) -> None:
    dataset = TBDataset(dummy_processed_dataset, split="train")
    assert len(dataset) == 4
    image, label, path = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert image.dtype == torch.float32
    assert label in (LABEL_TO_ID["Normal"], LABEL_TO_ID["TB"])
    assert Path(path).exists()


def test_tb_dataset_dual_representation(tmp_path: Path) -> None:
    root = tmp_path / "processed"
    folder = root / "train" / "TB"
    folder.mkdir(parents=True)
    img = np.full((200, 200), 160, dtype=np.uint8)
    img[40:150, 30:80] = 50
    img[40:150, 120:170] = 50
    cv2.imwrite(str(folder / "sample_0.png"), img)

    ds_dual = TBDataset(
        root,
        split="train",
        dual_representation=True,
        target_size=(518, 518),
    )
    img_dual, _, _ = ds_dual[0]
    assert img_dual.shape == (3, 518, 518)
    assert not torch.equal(img_dual[0], img_dual[1])
    assert not torch.equal(img_dual[0], img_dual[2])

    ds_single = TBDataset(
        root,
        split="train",
        dual_representation=False,
        target_size=(518, 518),
    )
    img_single, _, _ = ds_single[0]
    assert img_single.shape == (3, 518, 518)
    assert torch.equal(img_single[0], img_single[1])
    assert torch.equal(img_single[0], img_single[2])


def test_build_model_forward() -> None:
    model = build_model(
        model_name="facebook/dinov2-small",
        num_labels=2,
        freeze_backbone=True,
    )
    # Check that backbone is frozen
    for name, param in model.named_parameters():
        if "classifier" not in name:
            assert not param.requires_grad
        else:
            assert param.requires_grad

    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(pixel_values=dummy_input)
    assert output.logits.shape == (2, 2)


def test_compute_metrics() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])
    metrics = compute_metrics(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_train_one_epoch_smoke(dummy_processed_dataset: Path) -> None:
    dataset = TBDataset(dummy_processed_dataset, split="train")
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    model = build_model("facebook/dinov2-small", freeze_backbone=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")

    loss, acc = train_one_epoch(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )
    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_predict_image_smoke(
    dummy_processed_dataset: Path,
    tmp_path: Path,
) -> None:
    model = build_model("facebook/dinov2-small", freeze_backbone=True)
    device = torch.device("cpu")
    sample_img = (
        dummy_processed_dataset / "train" / "Normal" / "sample_0.png"
    )
    result = predict_image(model, sample_img, device=device)
    assert "label" in result
    assert result["label"] in ("Normal", "TB")
    assert 0.0 <= result["probability"] <= 1.0
    assert "Normal" in result["probabilities"]
    assert "TB" in result["probabilities"]


def test_build_densenet121_forward_and_freeze() -> None:
    model = build_model("densenet121", num_labels=2, freeze_backbone=True)
    for name, param in model.named_parameters():
        if "classifier" in name:
            assert param.requires_grad
        else:
            assert not param.requires_grad

    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(pixel_values=dummy_input)
    assert hasattr(output, "logits")
    assert output.logits.shape == (2, 2)


def test_train_densenet121_one_epoch_smoke(
    dummy_processed_dataset: Path,
) -> None:
    dataset = TBDataset(dummy_processed_dataset, split="train")
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    model = build_model("densenet121", freeze_backbone=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")

    loss, acc = train_one_epoch(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )
    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_densenet121_predict_and_load_smoke(
    dummy_processed_dataset: Path,
    tmp_path: Path,
) -> None:
    model = build_model("densenet121", freeze_backbone=True)
    device = torch.device("cpu")
    sample_img = dummy_processed_dataset / "train" / "Normal" / "sample_0.png"
    pred = predict_image(model, sample_img, device=device)
    assert pred["label"] in ("Normal", "TB")
    assert 0.0 <= pred["probability"] <= 1.0

    save_dir = tmp_path / "densenet_test"
    save_dir.mkdir()
    ckpt_file = save_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": "densenet121",
            "val_loss": 0.5,
            "val_acc": 0.8,
        },
        ckpt_file,
    )
    from tb_detection_model.models.predict_model import load_trained_model

    loaded = load_trained_model(save_dir, device=device)
    pred_loaded = predict_image(loaded, sample_img, device=device)
    assert pred_loaded["label"] == pred["label"]


def test_early_stopping_behavior() -> None:
    es = EarlyStopping(patience=2, min_delta=0.01, mode="min")
    assert es.step(1.0) is True
    assert not es.early_stop

    assert es.step(1.005) is False
    assert es.counter == 1
    assert not es.early_stop

    assert es.step(1.01) is False
    assert es.counter == 2
    assert es.early_stop


def test_compute_class_weights(dummy_processed_dataset: Path) -> None:
    dataset = TBDataset(dummy_processed_dataset, split="train")
    weights = compute_class_weights(dataset)
    assert weights is not None
    assert len(weights) == 2
    assert torch.isclose(weights[0], weights[1])

    pos_weighted = compute_class_weights(dataset, pos_weight=2.0)
    assert pos_weighted is not None
    assert torch.isclose(pos_weighted[0], torch.tensor(1.0))
    assert torch.isclose(pos_weighted[1], torch.tensor(2.0))


def test_dinov2_layer_unfreezing_2_and_4_blocks() -> None:
    # Test 2 blocks
    m2 = build_model(
        "facebook/dinov2-small",
        num_labels=2,
        freeze_backbone=False,
        train_top_k=2,
    )
    param_status_2 = {n: p.requires_grad for n, p in m2.named_parameters()}
    assert param_status_2["classifier.weight"] is True
    assert param_status_2["classifier.bias"] is True
    assert param_status_2["dinov2.layernorm.weight"] is True
    assert any(
        n.startswith("dinov2.encoder.layer.11.") and req
        for n, req in param_status_2.items()
    )
    assert any(
        n.startswith("dinov2.encoder.layer.10.") and req
        for n, req in param_status_2.items()
    )
    assert all(
        not req
        for n, req in param_status_2.items()
        if n.startswith("dinov2.encoder.layer.9.")
    )
    assert all(
        not req
        for n, req in param_status_2.items()
        if n.startswith("dinov2.embeddings.")
    )

    # Test 4 blocks
    m4 = build_model(
        "facebook/dinov2-small",
        num_labels=2,
        freeze_backbone=False,
        train_top_k=4,
    )
    param_status_4 = {n: p.requires_grad for n, p in m4.named_parameters()}
    assert param_status_4["classifier.weight"] is True
    assert param_status_4["dinov2.layernorm.weight"] is True
    for layer_idx in (8, 9, 10, 11):
        assert any(
            n.startswith(f"dinov2.encoder.layer.{layer_idx}.") and req
            for n, req in param_status_4.items()
        )
    for layer_idx in range(8):
        assert all(
            not req
            for n, req in param_status_4.items()
            if n.startswith(f"dinov2.encoder.layer.{layer_idx}.")
        )


def test_setup_optimizer_differential_lr() -> None:
    model = build_model(
        "facebook/dinov2-small",
        num_labels=2,
        freeze_backbone=False,
        train_top_k=2,
    )
    config = TrainConfig(
        learning_rate=1e-3,
        backbone_lr=1e-4,
        freeze_backbone=False,
        train_top_k=2,
    )
    optimizer = _setup_optimizer(model, config)
    assert len(optimizer.param_groups) == 2
    lrs = {group["lr"] for group in optimizer.param_groups}
    assert lrs == {1e-3, 1e-4}


def test_compute_metrics_sensitivity_and_specificity() -> None:
    # 2 true Normal, 2 true TB
    # Predictions: 1 TN, 1 FP (for Normal: specificity = 1/2 = 0.5)
    # Predictions: 2 TP, 0 FN (for TB: sensitivity = 2/2 = 1.0)
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array([0.2, 0.7, 0.8, 0.9])
    metrics = compute_metrics(y_true, y_pred, y_prob)

    assert "sensitivity" in metrics
    assert "specificity" in metrics
    assert np.isclose(metrics["sensitivity"], 1.0)
    assert np.isclose(metrics["specificity"], 0.5)
