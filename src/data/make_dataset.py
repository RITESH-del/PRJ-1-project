"""Build a patient-grouped dataset from raw ZIP archives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}
SPLITS = ("train", "val", "test")
LABELS = ("TB", "Normal")
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
TARGET_SIZE = (224, 224)
TRAIN_MEAN = 0.61313236
TRAIN_STD = 0.25896055


@dataclass(frozen=True)
class ImageRecord:
    """A source image reference and metadata, without retaining image bytes."""

    archive: Path
    member_name: str
    filename: str
    label: str
    patient_id: str
    patient_key: str
    digest: str
    source: str


def _label_for_member(member_name: str) -> str | None:
    """Infer a label from a path component, then from the filename suffix."""
    parts = Path(member_name).parts
    for part in parts[:-1]:
        normalized = part.casefold()
        if normalized == "tb":
            return "TB"
        if normalized in {"normal", "healthy"}:
            return "Normal"

    stem = Path(member_name).stem
    suffix = stem.rsplit("_", maxsplit=1)[-1]
    if suffix == "1":
        return "TB"
    if suffix == "0":
        return "Normal"
    return None


def _patient_id(filename: str) -> str:
    """Return the image-series identifier before its view/index suffix."""
    stem = Path(filename).stem
    match = re.match(r"^(?P<patient>.+)_\d+$", stem)
    return match.group("patient") if match else stem


def _patient_namespace(filename: str, archive: Path) -> str:
    """Prevent unrelated datasets with the same patient number from mixing."""
    stem = Path(filename).stem
    if stem.startswith("MCUCXR_"):
        return "montgomery"
    if stem.startswith("CHNCXR_"):
        return "shenzhen"
    if re.match(r"^\d{8}_\d+$", stem):
        return "nih"
    return archive.stem


def _decode_image(data: bytes) -> np.ndarray | None:
    encoded = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)


def is_valid_image(image: np.ndarray | None) -> bool:
    """Apply the notebook's minimum-size and nearly-black image checks."""
    if image is None or image.ndim != 2:
        return False
    if image.shape[0] < 50 or image.shape[1] < 50:
        return False
    return float(image.mean()) >= 5.0


def resize_with_padding(
    image: np.ndarray, target_size: tuple[int, int]
) -> np.ndarray:
    """Resize to ``(height, width)`` without changing the aspect ratio."""
    height, width = image.shape[:2]
    target_height, target_width = target_size
    scale = min(target_height / height, target_width / width)
    new_height = max(1, int(height * scale))
    new_width = max(1, int(width * scale))
    resized = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_AREA
    )

    pad_height = target_height - new_height
    pad_width = target_width - new_width
    top, bottom = pad_height // 2, pad_height - pad_height // 2
    left, right = pad_width // 2, pad_width - pad_width // 2
    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=0,
    )


def normalize_xray(
    image: np.ndarray,
    train_mean: float = TRAIN_MEAN,
    train_std: float = TRAIN_STD,
) -> np.ndarray:
    """Scale uint8 pixels with the notebook's fixed training statistics."""
    if train_std <= 0:
        raise ValueError("train_std must be positive")
    scaled = image.astype(np.float32) / 255.0
    return (scaled - train_mean) / train_std


def preprocess_xray(
    image_source: str | Path | np.ndarray,
    target_size: tuple[int, int] = TARGET_SIZE,
    train_mean: float = TRAIN_MEAN,
    train_std: float = TRAIN_STD,
    normalize: bool = False,
) -> np.ndarray | None:
    """Run the notebook pipeline and return a PNG-compatible image.

    ``normalize=False`` returns the uint8 output after CLAHE for storage.
    Set ``normalize=True`` to additionally return the notebook's float32
    z-score values for model input.
    """
    if isinstance(image_source, (str, Path)):
        image = cv2.imread(str(image_source), cv2.IMREAD_GRAYSCALE)
    else:
        image = image_source
    if not is_valid_image(image):
        return None

    height, width = image.shape
    top, bottom = int(0.10 * height), int(0.90 * height)
    left, right = int(0.10 * width), int(0.90 * width)
    roi = image[top:bottom, left:right]
    image = resize_with_padding(roi, target_size)
    image = cv2.medianBlur(image, 3)
    image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    image = image.astype(np.uint8, copy=False)
    return normalize_xray(image, train_mean, train_std) if normalize else image


def _record_from_entry(
    zip_file: ZipFile,
    archive: Path,
    info,
    seen_digests: dict[str, ImageRecord],
    seen_logical_keys: dict[tuple[str, str, str], ImageRecord],
) -> ImageRecord | None:
    member_path = Path(info.filename)
    if info.is_dir() or member_path.suffix.casefold() not in IMAGE_SUFFIXES:
        return None

    label = _label_for_member(info.filename)
    if label is None:
        LOGGER.warning(
            "Skipping image with unknown label: %s:%s",
            archive.name,
            info.filename,
        )
        return None

    filename = member_path.name
    patient_id = _patient_id(filename)
    patient_key = f"{_patient_namespace(filename, archive)}:{patient_id}"
    logical_key = (label, patient_key, filename)
    previous_logical = seen_logical_keys.get(logical_key)
    if previous_logical is not None:
        LOGGER.debug(
            "Skipping duplicate patient image: %s:%s (already from %s)",
            archive.name,
            info.filename,
            previous_logical.source,
        )
        return None

    data = zip_file.read(info)
    digest = hashlib.sha256(data).hexdigest()
    image = _decode_image(data)
    if not is_valid_image(image):
        LOGGER.warning(
            "Skipping invalid image: %s:%s", archive.name, info.filename
        )
        return None

    previous_digest = seen_digests.get(digest)
    if previous_digest is not None:
        if previous_digest.label != label:
            raise ValueError(
                "The same image bytes have conflicting labels: "
                f"{previous_digest.label} and {label}"
            )
        return None

    record = ImageRecord(
        archive=archive,
        member_name=info.filename,
        filename=filename,
        label=label,
        patient_id=patient_id,
        patient_key=patient_key,
        digest=digest,
        source=archive.stem,
    )
    seen_digests[digest] = record
    seen_logical_keys[logical_key] = record
    return record


def _unique_filenames(records: list[ImageRecord]) -> list[ImageRecord]:
    filename_counts: dict[str, int] = defaultdict(int)
    for record in records:
        filename_counts[record.filename] += 1

    unique_records: list[ImageRecord] = []
    for record in records:
        if filename_counts[record.filename] == 1:
            unique_records.append(record)
            continue
        path = Path(record.filename)
        unique_records.append(
            ImageRecord(
                archive=record.archive,
                member_name=record.member_name,
                filename=(
                    f"{path.stem}_{record.digest[:8]}{path.suffix.casefold()}"
                ),
                label=record.label,
                patient_id=record.patient_id,
                patient_key=record.patient_key,
                digest=record.digest,
                source=record.source,
            )
        )
    return unique_records


def _source_records(raw_dir: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    seen_digests: dict[str, ImageRecord] = {}
    seen_logical_keys: dict[tuple[str, str, str], ImageRecord] = {}

    archives = sorted(
        raw_dir.glob("*.zip"), key=lambda path: path.name.casefold()
    )
    if not archives:
        raise FileNotFoundError(f"No ZIP files found in {raw_dir}")

    for archive in archives:
        with ZipFile(archive) as zip_file:
            entries = sorted(
                zip_file.infolist(), key=lambda info: info.filename.casefold()
            )
            for info in entries:
                record = _record_from_entry(
                    zip_file,
                    archive,
                    info,
                    seen_digests,
                    seen_logical_keys,
                )
                if record is not None:
                    records.append(record)

    if not records:
        raise ValueError(f"No valid labelled images found in {raw_dir}")
    return _unique_filenames(records)


def _stable_sort_key(seed: int, label: str, patient_key: str) -> bytes:
    value = f"{seed}:{label}:{patient_key}".encode("utf-8")
    return hashlib.sha256(value).digest()


def _assign_groups(records: list[ImageRecord], seed: int) -> dict[str, str]:
    """Assign complete patient groups to deterministic, stratified splits."""
    groups: dict[tuple[str, str], list[ImageRecord]] = defaultdict(list)
    for record in records:
        groups[(record.label, record.patient_key)].append(record)
    patient_labels: dict[str, str] = {}
    for record in records:
        previous_label = patient_labels.setdefault(
            record.patient_key, record.label
        )
        if previous_label != record.label:
            raise ValueError(
                f"Patient group {record.patient_key} contains both "
                f"{previous_label} and {record.label} images"
            )

    assignments: dict[str, str] = {}
    for label in LABELS:
        label_groups = [
            (patient_key, group)
            for (group_label, patient_key), group in groups.items()
            if group_label == label
        ]
        label_groups.sort(
            key=lambda item: _stable_sort_key(seed, label, item[0])
        )
        total_images = sum(len(group) for _, group in label_groups)
        targets = {
            split: total_images * SPLIT_RATIOS[split] for split in SPLITS
        }
        current = {split: 0 for split in SPLITS}
        group_counts = {split: 0 for split in SPLITS}

        for index, (patient_key, group) in enumerate(label_groups):
            remaining_groups = len(label_groups) - index
            empty_splits = [
                split for split in SPLITS if group_counts[split] == 0
            ]
            candidates = (
                empty_splits
                if remaining_groups <= len(empty_splits)
                else list(SPLITS)
            )
            split = min(
                candidates,
                key=lambda candidate: (
                    current[candidate] / targets[candidate]
                    if targets[candidate]
                    else float("inf"),
                    SPLITS.index(candidate),
                ),
            )
            assignments[patient_key] = split
            current[split] += len(group)
            group_counts[split] += 1

    return assignments


def _prepare_output(output_dir: Path) -> None:
    for split in SPLITS:
        for label in LABELS:
            label_dir = output_dir / split / label
            label_dir.mkdir(parents=True, exist_ok=True)
            for image_path in label_dir.glob("*.png"):
                image_path.unlink()
    manifest = output_dir / "manifest.csv"
    if manifest.exists():
        manifest.unlink()


def build_dataset(
    raw_dir: Path, output_dir: Path, seed: int = 42
) -> dict[str, dict[str, int]]:
    """Convert all unique raw images into the requested folder structure."""
    records = _source_records(raw_dir)
    assignments = _assign_groups(records, seed)
    _prepare_output(output_dir)

    counts = {split: {label: 0 for label in LABELS} for split in SPLITS}
    manifest_rows: list[dict[str, str]] = []
    for record in records:
        split = assignments[record.patient_key]
        output_path = output_dir / split / record.label / record.filename
        with ZipFile(record.archive) as zip_file:
            image = _decode_image(zip_file.read(record.member_name))
        processed = preprocess_xray(image)
        if processed is None or not cv2.imwrite(str(output_path), processed):
            raise RuntimeError(
                (
                    "Could not preprocess or write "
                    f"{record.archive}:{record.member_name}"
                )
            )

        relative_path = output_path.relative_to(output_dir).as_posix()
        manifest_rows.append(
            {
                "filename": relative_path,
                "label": record.label,
                "split": split,
                "patient_id": record.patient_id,
                "source": record.source,
            }
        )
        counts[split][record.label] += 1

    with (output_dir / "manifest.csv").open("w", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=["filename", "label", "split", "patient_id", "source"],
        )
        writer.writeheader()
        writer.writerows(
            sorted(manifest_rows, key=lambda row: row["filename"])
        )

    LOGGER.info("Wrote %d unique images to %s", len(records), output_dir)
    for split in SPLITS:
        LOGGER.info("%s: %s", split, counts[split])
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_filepath",
        nargs="?",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory containing raw ZIP archives (default: data/raw)",
    )
    parser.add_argument(
        "output_filepath",
        nargs="?",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Processed dataset directory (default: data/processed)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Deterministic split seed"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    build_dataset(args.input_filepath, args.output_filepath, seed=args.seed)


if __name__ == "__main__":
    main()
