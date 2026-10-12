import csv
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

from src.data.make_dataset import (
    build_dataset,
    normalize_xray,
    preprocess_xray,
)


def _png_bytes(value: int) -> bytes:
    image = np.full((80, 100), value, dtype=np.uint8)
    image[20:60, 30:70] = min(value + 30, 255)
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


def _write_fixture_archive(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        for label, suffix, start, offset in (
            ("TB", "1", 40, 0),
            ("Normal", "0", 120, 20),
        ):
            for patient_number in range(1, 7):
                patient = f"CHNCXR_{patient_number + offset:04d}"
                archive.writestr(
                    f"dataset/{label}/{patient}_{suffix}.png",
                    _png_bytes(start + patient_number),
                )
                if patient_number == 1:
                    archive.writestr(
                        f"dataset/{label}/{patient}_2.png",
                        _png_bytes(start + patient_number + 1),
                    )


def test_preprocess_xray_preserves_shape_and_normalizes_on_request():
    image = np.full((80, 100), 100, dtype=np.uint8)
    processed = preprocess_xray(image)
    assert processed is not None
    assert processed.shape == (224, 224)
    assert processed.dtype == np.uint8

    normalized = preprocess_xray(image, normalize=True)
    assert normalized is not None
    assert normalized.shape == (224, 224)
    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized, normalize_xray(processed))


def test_build_dataset_is_patient_grouped_stratified_and_reproducible(
    tmp_path,
):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()
    _write_fixture_archive(raw_dir / "fixture.zip")

    first_counts = build_dataset(raw_dir, output_dir, seed=17)
    first_manifest = (output_dir / "manifest.csv").read_text()
    second_counts = build_dataset(raw_dir, output_dir, seed=17)
    assert first_counts == second_counts
    assert first_manifest == (output_dir / "manifest.csv").read_text()

    rows = list(csv.DictReader((output_dir / "manifest.csv").open()))
    assert {row["label"] for row in rows} == {"TB", "Normal"}
    assert {row["split"] for row in rows} == {"train", "val", "test"}

    patient_splits = {}
    for row in rows:
        key = (row["label"], row["patient_id"])
        patient_splits.setdefault(key, set()).add(row["split"])
    assert all(len(splits) == 1 for splits in patient_splits.values())

    for row in rows:
        image_path = output_dir / row["filename"]
        assert image_path.is_file()
        decoded = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        assert decoded is not None
        assert decoded.shape == (224, 224)

    for split in ("train", "val", "test"):
        for label in ("TB", "Normal"):
            assert (output_dir / split / label).is_dir()
