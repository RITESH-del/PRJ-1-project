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
from typing import Sequence
from zipfile import ZipFile

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}
SPLITS = ("train", "val", "test")
LABELS = ("TB", "Normal")
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
TARGET_SIZE = (518, 518)
TRAIN_MEAN = 0.61313236
TRAIN_STD = 0.25896055
LOG_LEVELS = frozenset(
    {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
)


class ConfigurationError(ValueError):
    """Raised when configuration values cannot describe a dataset run."""


@dataclass(frozen=True)
class DatasetConfig:
    """All user-tunable dataset-building and preprocessing parameters."""

    raw_dir: Path = DEFAULT_RAW_DIR
    processed_dir: Path = DEFAULT_PROCESSED_DIR
    archive_glob: str = "*.zip"
    output_suffix: str = ".png"
    manifest_filename: str = "manifest.csv"
    image_suffixes: frozenset[str] = frozenset(IMAGE_SUFFIXES)
    splits: tuple[str, ...] = SPLITS
    labels: tuple[str, ...] = LABELS
    split_ratios: tuple[tuple[str, float], ...] = (
        ("train", 0.70),
        ("val", 0.15),
        ("test", 0.15),
    )
    default_seed: int = 42
    target_size: tuple[int, int] = TARGET_SIZE
    train_mean: float = TRAIN_MEAN
    train_std: float = TRAIN_STD
    minimum_image_size: tuple[int, int] = (50, 50)
    minimum_image_mean: float = 5.0
    roi_mode: str = "anatomical"
    roi_height_fraction: tuple[float, float] = (0.02, 0.98)
    roi_width_fraction: tuple[float, float] = (0.02, 0.98)
    median_blur_kernel: int = 0
    bilateral_filter: bool = True
    bilateral_d: int = 5
    bilateral_sigma_color: float = 30.0
    bilateral_sigma_space: float = 30.0
    clahe_clip_limit: float = 1.5
    clahe_tile_grid_size: tuple[int, int] = (16, 16)
    scan_log_interval: int = 1000
    process_log_interval: int = 100
    manifest_fields: tuple[str, ...] = (
        "filename",
        "label",
        "split",
        "patient_id",
        "source",
    )
    log_level: int = logging.INFO
    log_format: str = "%(asctime)s - %(levelname)s - %(message)s"
    train_archives: tuple[str, ...] | None = None
    validation_archives: tuple[str, ...] | None = None
    test_archives: tuple[str, ...] | None = None
    tb_directory_names: frozenset[str] = frozenset({"tb"})
    normal_directory_names: frozenset[str] = frozenset(
        {"normal", "healthy"}
    )
    tb_filename_suffix: str = "1"
    normal_filename_suffix: str = "0"
    montgomery_filename_prefix: str = "MCUCXR_"
    shenzhen_filename_prefix: str = "CHNCXR_"
    nih_filename_pattern: str = r"^\d{8}_\d+$"

    def __post_init__(self) -> None:
        errors = (
            self._validate_paths()
            + self._validate_dataset_settings()
            + self._validate_preprocessing()
            + self._validate_output_settings()
            + self._validate_archives()
        )
        if errors:
            raise ConfigurationError("; ".join(errors))

    def _validate_paths(self) -> list[str]:
        errors: list[str] = []
        if not isinstance(self.raw_dir, Path):
            errors.append("input_dir must be a path")
        if not isinstance(self.processed_dir, Path):
            errors.append("output_dir must be a path")
        if isinstance(self.raw_dir, Path) and isinstance(
            self.processed_dir, Path
        ):
            raw_path = self.raw_dir.resolve()
            output_path = self.processed_dir.resolve()
            if raw_path == output_path:
                errors.append("input_dir and output_dir must differ")
            elif output_path.is_relative_to(raw_path):
                errors.append("output_dir must not be inside input_dir")
        return errors

    def _validate_dataset_settings(self) -> list[str]:
        errors: list[str] = []
        if not isinstance(self.default_seed, int) or isinstance(
            self.default_seed, bool
        ):
            errors.append("seed must be an integer")
        if not self._positive_int_pair(self.target_size):
            errors.append("target_size must contain two positive integers")
        try:
            ratios = dict(self.split_ratios)
        except (TypeError, ValueError):
            return errors + ["split_ratios must contain train, val, and test"]
        if set(ratios) != set(self.splits):
            errors.append("split_ratios must contain train, val, and test")
        elif any(
            not isinstance(ratio, (int, float))
            or isinstance(ratio, bool)
            or ratio <= 0
            for ratio in ratios.values()
        ):
            errors.append("split ratios must all be positive")
        elif not np.isclose(sum(ratios.values()), 1.0):
            errors.append("split ratios must sum to 1.0")
        return errors

    def _validate_preprocessing_rois(self) -> list[str]:
        errors: list[str] = []
        if self.roi_mode not in {"anatomical", "fractional"}:
            errors.append("roi_mode must be 'anatomical' or 'fractional'")
        if not self._valid_fraction(self.roi_height_fraction):
            errors.append("roi_height_fraction must be an increasing range")
        if not self._valid_fraction(self.roi_width_fraction):
            errors.append("roi_width_fraction must be an increasing range")
        return errors

    def _validate_preprocessing_filters(self) -> list[str]:
        errors: list[str] = []
        if self.median_blur_kernel != 0 and (
            self.median_blur_kernel < 1 or self.median_blur_kernel % 2 == 0
        ):
            errors.append(
                "median_blur_kernel must be 0 or a positive odd integer"
            )
        if self.bilateral_d < 0:
            errors.append("bilateral_d must be non-negative")
        if self.bilateral_sigma_color <= 0:
            errors.append("bilateral_sigma_color must be positive")
        if self.bilateral_sigma_space <= 0:
            errors.append("bilateral_sigma_space must be positive")
        return errors

    def _validate_preprocessing(self) -> list[str]:
        errors = (
            self._validate_preprocessing_rois()
            + self._validate_preprocessing_filters()
        )
        if not self._positive_int_pair(self.minimum_image_size):
            errors.append(
                "minimum_image_size must contain two positive integers"
            )
        if self.train_std <= 0:
            errors.append("train_std must be positive")
        if not self._positive_int_pair(self.clahe_tile_grid_size):
            errors.append("clahe_tile_grid_size must be positive integers")
        if self.minimum_image_mean < 0:
            errors.append("minimum_image_mean must be non-negative")
        if self.clahe_clip_limit <= 0:
            errors.append("clahe_clip_limit must be positive")
        if self.scan_log_interval < 0 or self.process_log_interval < 0:
            errors.append("log intervals must be non-negative")
        return errors

    def _validate_output_settings(self) -> list[str]:
        errors: list[str] = []
        if self.output_suffix != ".png":
            errors.append("output_suffix must be .png")
        if not self.manifest_filename or Path(self.manifest_filename).name != (
            self.manifest_filename
        ):
            errors.append("manifest_filename must be a file name")
        valid_levels = {getattr(logging, level) for level in LOG_LEVELS}
        if self.log_level not in valid_levels:
            errors.append("log_level must be a standard logging level")
        return errors

    def _validate_archives(self) -> list[str]:
        options = (
            self.train_archives,
            self.validation_archives,
            self.test_archives,
        )
        if all(option is None for option in options):
            return []
        if any(option is None for option in options):
            return ["train, validation, and test archives must all be set"]
        if not all(options):
            return ["archive assignments must not be empty"]
        train = set(self.train_archives or ())
        evaluation = set(
            (self.validation_archives or ()) + (self.test_archives or ())
        )
        if train & evaluation:
            return ["an archive cannot be assigned to train and evaluation"]
        return []

    @staticmethod
    def _positive_int_pair(value: object) -> bool:
        return (
            isinstance(value, (tuple, list))
            and len(value) == 2
            and all(
                isinstance(item, int)
                and not isinstance(item, bool)
                and item > 0
                for item in value
            )
        )

    @staticmethod
    def _valid_fraction(value: object) -> bool:
        return (
            isinstance(value, (tuple, list))
            and len(value) == 2
            and all(isinstance(item, (int, float)) for item in value)
            and 0 <= value[0] < value[1] <= 1
        )


CONFIG = DatasetConfig()


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


def _label_for_member(
    member_name: str, config: DatasetConfig = CONFIG
) -> str | None:
    """Infer a label from a path component, then from the filename suffix."""
    parts = Path(member_name).parts
    for part in parts[:-1]:
        normalized = part.casefold()
        if normalized in config.tb_directory_names:
            return config.labels[0]
        if normalized in config.normal_directory_names:
            return config.labels[1]

    stem = Path(member_name).stem
    suffix = stem.rsplit("_", maxsplit=1)[-1]
    if suffix == config.tb_filename_suffix:
        return config.labels[0]
    if suffix == config.normal_filename_suffix:
        return config.labels[1]
    return None


def _patient_id(filename: str) -> str:
    """Return the image-series identifier before its view/index suffix."""
    stem = Path(filename).stem
    match = re.match(r"^(?P<patient>.+)_\d+$", stem)
    return match.group("patient") if match else stem


def _patient_namespace(
    filename: str, archive: Path, config: DatasetConfig = CONFIG
) -> str:
    """Prevent unrelated datasets with the same patient number from mixing."""
    stem = Path(filename).stem
    if stem.startswith(config.montgomery_filename_prefix):
        return "montgomery"
    if stem.startswith(config.shenzhen_filename_prefix):
        return "shenzhen"
    if re.match(config.nih_filename_pattern, stem):
        return "nih"
    return archive.stem


def _decode_image(data: bytes) -> np.ndarray | None:
    encoded = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)


def is_valid_image(
    image: np.ndarray | None, config: DatasetConfig = CONFIG
) -> bool:
    """Apply the notebook's minimum-size and nearly-black image checks."""
    if image is None or image.ndim != 2:
        return False
    minimum_height, minimum_width = config.minimum_image_size
    if (
        image.shape[0] < minimum_height
        or image.shape[1] < minimum_width
    ):
        return False
    return float(image.mean()) >= config.minimum_image_mean


def resize_with_padding(
    image: np.ndarray, target_size: tuple[int, int]
) -> np.ndarray:
    """Resize to ``(height, width)`` without changing the aspect ratio."""
    height, width = image.shape[:2]
    target_height, target_width = target_size
    scale = min(target_height / height, target_width / width)
    new_height = max(1, int(height * scale))
    new_width = max(1, int(width * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    resized = cv2.resize(
        image, (new_width, new_height), interpolation=interp
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
    train_mean: float | None = None,
    train_std: float | None = None,
    *,
    config: DatasetConfig = CONFIG,
) -> np.ndarray:
    """Scale uint8 pixels with the notebook's fixed training statistics."""
    train_mean = config.train_mean if train_mean is None else train_mean
    train_std = config.train_std if train_std is None else train_std
    if train_std <= 0:
        raise ValueError("train_std must be positive")
    scaled = image.astype(np.float32) / 255.0
    return (scaled - train_mean) / train_std


def _fallback_roi(
    height: int,
    width: int,
    h_frac: tuple[float, float],
    w_frac: tuple[float, float],
) -> tuple[int, int, int, int]:
    top = max(0, min(height - 1, int(h_frac[0] * height)))
    bottom = max(top + 1, min(height, int(h_frac[1] * height)))
    left = max(0, min(width - 1, int(w_frac[0] * width)))
    right = max(left + 1, min(width, int(w_frac[1] * width)))
    return top, bottom, left, right


def _find_body_bbox(
    image: np.ndarray,
    scale: float,
) -> tuple[int, int, int, int] | None:
    sh, sw = image.shape[:2]
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    k = max(5, int(min(sh, sw) * 0.05))
    k = k if k % 2 == 1 else k + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 0.15 * float(sh * sw):
        return None
    x, y, w, h = cv2.boundingRect(largest)
    return int(x / scale), int(y / scale), int(w / scale), int(h / scale)


def detect_thorax_roi(
    image: np.ndarray,
    *,
    config: DatasetConfig | None = None,
) -> tuple[int, int, int, int]:
    """Detect anatomical thoracic bounding box preserving lung apices."""
    height, width = image.shape[:2]
    h_frac = config.roi_height_fraction if config else (0.02, 0.98)
    w_frac = config.roi_width_fraction if config else (0.02, 0.98)
    fallback = _fallback_roi(height, width, h_frac, w_frac)
    if min(height, width) < 10 or float(np.std(image)) < 1.0:
        return fallback

    max_dim = max(height, width)
    scale = 512.0 / max_dim if max_dim > 512 else 1.0
    sw, sh = max(1, int(width * scale)), max(1, int(height * scale))
    small = (
        cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else image
    )
    bbox = _find_body_bbox(small, scale)
    if bbox is None:
        return fallback

    orig_x, orig_y, orig_w, orig_h = bbox
    m_y, m_x = int(0.02 * height), int(0.02 * width)
    top, bottom = max(0, orig_y - m_y), min(height, orig_y + orig_h + m_y)
    left, right = max(0, orig_x - m_x), min(width, orig_x + orig_w + m_x)
    if (bottom - top) < int(0.2 * height) or (right - left) < int(0.2 * width):
        return fallback
    return top, bottom, left, right


def _find_lung_contours(
    thresh: np.ndarray,
    h: int,
    w: int,
) -> list[np.ndarray]:
    k = max(5, int(min(h, w) * 0.03))
    k = k if k % 2 == 1 else k + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    k_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_sm)
    contours, _ = cv2.findContours(
        opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    total_area = float(h * w)
    valid = [
        c for c in contours
        if 0.02 * total_area < cv2.contourArea(c) < 0.75 * total_area
    ]
    valid.sort(key=cv2.contourArea, reverse=True)
    return valid


def generate_lung_mask(image: np.ndarray) -> np.ndarray:
    """Generate anatomical lung field mask."""
    h, w = image.shape[:2]
    fallback = np.full_like(image, 255)
    if min(h, w) < 20 or float(np.std(image)) < 1.0:
        return fallback

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    by, bx = max(1, int(0.04 * h)), max(1, int(0.04 * w))
    thresh[:by, :] = 0
    thresh[-by:, :] = 0
    thresh[:, :bx] = 0
    thresh[:, -bx:] = 0

    valid = _find_lung_contours(thresh, h, w)
    if not valid:
        return fallback

    mask = np.zeros_like(image)
    cv2.drawContours(mask, valid[:2], -1, 255, thickness=cv2.FILLED)
    d_k = max(3, int(min(h, w) * 0.02))
    d_k = d_k if d_k % 2 == 1 else d_k + 1
    d_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d_k, d_k))
    return cv2.dilate(mask, d_kernel)


def compute_high_pass_residual(image: np.ndarray) -> np.ndarray:
    """Compute high-contrast edge residual highlighting cavitary borders."""
    blurred = cv2.GaussianBlur(image, (9, 9), 2.0)
    diff = cv2.absdiff(image, blurred)
    return cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)


def _ensure_radiologic_polarity(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    ch, cw = max(1, h // 10), max(1, w // 10)
    corners = (
        image[:ch, :cw],
        image[:ch, -cw:],
        image[-ch:, :cw],
        image[-ch:, -cw:],
    )
    corner_mean = float(np.mean([c.mean() for c in corners]))
    center = image[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    center_mean = float(center.mean()) if center.size > 0 else 0.0
    if corner_mean > 150.0 and corner_mean > (center_mean + 20.0):
        return 255 - image
    return image


def _normalize_dynamic_range(image: np.ndarray) -> np.ndarray:
    non_zero = image[image > 0]
    if non_zero.size == 0:
        return image
    p_low = float(np.percentile(non_zero, 0.5))
    p_high = float(np.percentile(non_zero, 99.5))
    if p_high <= p_low:
        return image
    clipped = np.clip(image, p_low, p_high)
    stretched = (clipped - p_low) / (p_high - p_low) * 255.0
    return stretched.astype(np.uint8)


def harmonize_xray_radiometry(
    image: np.ndarray,
    *,
    config: DatasetConfig | None = None,
) -> np.ndarray:
    """Harmonize digital detector dynamic range and contrast across cohorts."""
    image = _ensure_radiologic_polarity(image)
    image = _normalize_dynamic_range(image)

    h, w = image.shape[:2]
    clip_limit = config.clahe_clip_limit if config else 1.5
    tile_grid = config.clahe_tile_grid_size if config else None
    if tile_grid is None:
        th = max(4, int(round(h / 32.0)))
        tw = max(4, int(round(w / 32.0)))
        tile_grid = (th, tw)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(image)


def preprocess_xray(
    image_source: str | Path | np.ndarray,
    target_size: tuple[int, int] | None = None,
    train_mean: float | None = None,
    train_std: float | None = None,
    normalize: bool = False,
    *,
    config: DatasetConfig = CONFIG,
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
    if not is_valid_image(image, config):
        return None

    target_size = (
        config.target_size if target_size is None else target_size
    )
    train_mean = config.train_mean if train_mean is None else train_mean
    train_std = config.train_std if train_std is None else train_std
    height, width = image.shape
    if config.roi_mode == "anatomical":
        top, bottom, left, right = detect_thorax_roi(image, config=config)
    else:
        height_start, height_end = config.roi_height_fraction
        width_start, width_end = config.roi_width_fraction
        top, bottom = int(height_start * height), int(height_end * height)
        left, right = int(width_start * width), int(width_end * width)
    roi = image[top:bottom, left:right]
    if config.bilateral_filter and roi.size > 0:
        roi = cv2.bilateralFilter(
            roi,
            d=config.bilateral_d,
            sigmaColor=config.bilateral_sigma_color,
            sigmaSpace=config.bilateral_sigma_space,
        )
    image = resize_with_padding(roi, target_size)
    if config.median_blur_kernel > 0:
        image = cv2.medianBlur(image, config.median_blur_kernel)
    image = harmonize_xray_radiometry(image, config=config)
    image = image.astype(np.uint8, copy=False)
    return (
        normalize_xray(
            image,
            train_mean,
            train_std,
            config=config,
        )
        if normalize
        else image
    )


def _record_from_entry(
    zip_file: ZipFile,
    archive: Path,
    info,
    seen_digests: dict[str, ImageRecord],
    seen_logical_keys: dict[tuple[str, str, str], ImageRecord],
    config: DatasetConfig = CONFIG,
) -> ImageRecord | None:
    member_path = Path(info.filename)
    if (
        info.is_dir()
        or member_path.suffix.casefold() not in config.image_suffixes
    ):
        return None

    label = _label_for_member(info.filename, config)
    if label is None:
        LOGGER.warning(
            "Skipping image with unknown label: %s:%s",
            archive.name,
            info.filename,
        )
        return None

    filename = member_path.name
    patient_id = _patient_id(filename)
    patient_key = (
        f"{_patient_namespace(filename, archive, config)}:{patient_id}"
    )
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
    if not is_valid_image(image, config):
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


def _unique_filenames(
    records: list[ImageRecord], config: DatasetConfig = CONFIG
) -> list[ImageRecord]:
    output_names = [
        f"{Path(record.filename).stem}{config.output_suffix}"
        for record in records
    ]
    filename_counts: dict[str, int] = defaultdict(int)
    for filename in output_names:
        filename_counts[filename] += 1

    unique_records: list[ImageRecord] = []
    for record, output_filename in zip(records, output_names):
        filename = output_filename
        if filename_counts[output_filename] > 1:
            path = Path(output_filename)
            filename = (
                f"{path.stem}_{record.digest[:8]}{config.output_suffix}"
            )
        unique_records.append(
            ImageRecord(
                archive=record.archive,
                member_name=record.member_name,
                filename=filename,
                label=record.label,
                patient_id=record.patient_id,
                patient_key=record.patient_key,
                digest=record.digest,
                source=record.source,
            )
        )
    return unique_records


def _selected_archive_names(
    train_archives: Sequence[str] | None,
    validation_archives: Sequence[str] | None,
    test_archives: Sequence[str] | None,
) -> set[str] | None:
    configured = (
        train_archives,
        validation_archives,
        test_archives,
    )
    if all(archives is None for archives in configured):
        return None
    return {
        Path(archive).name
        for archives in configured
        if archives is not None
        for archive in archives
    }


def _source_records(
    raw_dir: Path,
    archive_names: set[str] | None = None,
    config: DatasetConfig = CONFIG,
) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    seen_digests: dict[str, ImageRecord] = {}
    seen_logical_keys: dict[tuple[str, str, str], ImageRecord] = {}

    archives = sorted(
        (
            path
            for path in raw_dir.glob(config.archive_glob)
            if archive_names is None or path.name in archive_names
        ),
        key=lambda path: path.name.casefold(),
    )
    if not archives:
        raise FileNotFoundError(f"No ZIP files found in {raw_dir}")

    for archive in archives:
        LOGGER.info("Scanning archive %s", archive.name)
        records_before = len(records)
        with ZipFile(archive) as zip_file:
            entries = sorted(
                zip_file.infolist(), key=lambda info: info.filename.casefold()
            )
            for index, info in enumerate(entries, start=1):
                record = _record_from_entry(
                    zip_file,
                    archive,
                    info,
                    seen_digests,
                    seen_logical_keys,
                    config,
                )
                if record is not None:
                    records.append(record)
                if (
                    config.scan_log_interval
                    and index % config.scan_log_interval == 0
                ):
                    LOGGER.info(
                        "Scanning %s: %d/%d entries",
                        archive.name,
                        index,
                        len(entries),
                    )
        LOGGER.info(
            "Scanned %s: %d valid images",
            archive.name,
            len(records) - records_before,
        )

    if not records:
        raise ValueError(f"No valid labelled images found in {raw_dir}")
    return _unique_filenames(records, config)


def _stable_sort_key(seed: int, label: str, patient_key: str) -> bytes:
    value = f"{seed}:{label}:{patient_key}".encode("utf-8")
    return hashlib.sha256(value).digest()


def _patient_groups(
    records: list[ImageRecord],
) -> dict[tuple[str, str], list[ImageRecord]]:
    groups: dict[tuple[str, str], list[ImageRecord]] = defaultdict(list)
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
        groups[(record.label, record.patient_key)].append(record)
    return groups


def _assign_groups(
    records: list[ImageRecord],
    seed: int,
    config: DatasetConfig = CONFIG,
) -> dict[str, str]:
    """Assign complete patient groups to deterministic, stratified splits."""
    groups = _patient_groups(records)

    assignments: dict[str, str] = {}
    split_ratios = dict(config.split_ratios)
    for label in config.labels:
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
            split: total_images * split_ratios[split]
            for split in config.splits
        }
        current = {split: 0 for split in config.splits}
        group_counts = {split: 0 for split in config.splits}

        for index, (patient_key, group) in enumerate(label_groups):
            remaining_groups = len(label_groups) - index
            empty_splits = [
                split
                for split in config.splits
                if group_counts[split] == 0
            ]
            candidates = (
                empty_splits
                if remaining_groups <= len(empty_splits)
                else list(config.splits)
            )
            split = min(
                candidates,
                key=lambda candidate: (
                    current[candidate] / targets[candidate]
                    if targets[candidate]
                    else float("inf"),
                    config.splits.index(candidate),
                ),
            )
            assignments[patient_key] = split
            current[split] += len(group)
            group_counts[split] += 1

    return assignments


def _archive_split_options(
    records: list[ImageRecord],
    train_archives: Sequence[str] | None,
    validation_archives: Sequence[str] | None,
    test_archives: Sequence[str] | None,
    config: DatasetConfig = CONFIG,
) -> dict[str, frozenset[str]] | None:
    configured = (
        train_archives,
        validation_archives,
        test_archives,
    )
    if all(archives is None for archives in configured):
        return None
    if any(archives is None for archives in configured):
        raise ValueError(
            "train, validation, and test archives must all be provided"
        )

    available = {record.archive.name for record in records}
    options: dict[str, set[str]] = defaultdict(set)
    for split, archives in zip(config.splits, configured):
        assert archives is not None
        for archive in archives:
            archive_name = Path(archive).name
            if archive_name not in available:
                raise ValueError(
                    f"Archive {archive_name!r} has no valid labelled images"
                )
            options[archive_name].add(split)

    missing = sorted(available - options.keys())
    if missing:
        raise ValueError(
            "Every archive must be assigned explicitly; missing: "
            + ", ".join(missing)
        )
    for archive, splits in options.items():
        if "train" in splits and len(splits) > 1:
            raise ValueError(
                f"Archive {archive!r} cannot mix train and evaluation splits"
            )
    return {
        archive: frozenset(splits) for archive, splits in options.items()
    }


def _assign_source_groups(
    records: list[ImageRecord],
    seed: int,
    source_splits: dict[str, frozenset[str]],
    config: DatasetConfig = CONFIG,
) -> dict[str, str]:
    """Assign groups while constraining each source to configured splits."""
    groups = _patient_groups(records)
    assignments: dict[str, str] = {}
    for label in config.labels:
        label_groups = [
            (patient_key, group)
            for (group_label, patient_key), group in groups.items()
            if group_label == label
        ]
        label_groups.sort(
            key=lambda item: _stable_sort_key(seed, label, item[0])
        )
        flexible: list[tuple[str, list[ImageRecord], frozenset[str]]] = []
        for patient_key, group in label_groups:
            candidates = set(config.splits)
            for record in group:
                candidates &= source_splits[record.archive.name]
            if not candidates:
                raise ValueError(
                    "Patient group "
                    f"{patient_key} has conflicting archive splits"
                )
            if len(candidates) == 1:
                assignments[patient_key] = next(iter(candidates))
            else:
                flexible.append(
                    (patient_key, group, frozenset(candidates))
                )

        flexible_splits = sorted(
            {
                split
                for _, _, candidates in flexible
                for split in candidates
            },
            key=config.splits.index,
        )
        total_flexible = sum(len(group) for _, group, _ in flexible)
        targets = {
            split: total_flexible / len(flexible_splits)
            for split in flexible_splits
        }
        current = {split: 0 for split in flexible_splits}
        for patient_key, group, candidates in flexible:
            split = min(
                candidates,
                key=lambda candidate: (
                    current[candidate] / targets[candidate],
                    config.splits.index(candidate),
                ),
            )
            assignments[patient_key] = split
            current[split] += len(group)

    return assignments


def _prepare_output(
    output_dir: Path, config: DatasetConfig = CONFIG
) -> None:
    for split in config.splits:
        for label in config.labels:
            label_dir = output_dir / split / label
            label_dir.mkdir(parents=True, exist_ok=True)
            for image_path in label_dir.glob(f"*{config.output_suffix}"):
                image_path.unlink()
    manifest = output_dir / config.manifest_filename
    if manifest.exists():
        manifest.unlink()


def build_dataset(
    raw_dir: Path,
    output_dir: Path,
    seed: int | None = None,
    train_archives: Sequence[str] | None = None,
    validation_archives: Sequence[str] | None = None,
    test_archives: Sequence[str] | None = None,
    *,
    config: DatasetConfig = CONFIG,
) -> dict[str, dict[str, int]]:
    """Convert unique raw images into the requested folder structure."""
    if all(
        archives is None
        for archives in (
            train_archives,
            validation_archives,
            test_archives,
        )
    ):
        train_archives = config.train_archives
        validation_archives = config.validation_archives
        test_archives = config.test_archives
    seed = config.default_seed if seed is None else seed
    archive_names = _selected_archive_names(
        train_archives,
        validation_archives,
        test_archives,
    )
    records = _source_records(raw_dir, archive_names, config)
    source_splits = _archive_split_options(
        records,
        train_archives,
        validation_archives,
        test_archives,
        config,
    )
    assignments = (
        _assign_groups(records, seed, config)
        if source_splits is None
        else _assign_source_groups(records, seed, source_splits, config)
    )
    _prepare_output(output_dir, config)

    counts = {
        split: {label: 0 for label in config.labels}
        for split in config.splits
    }
    manifest_rows: list[dict[str, str]] = []
    records_by_archive: dict[Path, list[ImageRecord]] = defaultdict(list)
    for record in records:
        records_by_archive[record.archive].append(record)

    processed_count = 0
    for archive in sorted(
        records_by_archive, key=lambda path: path.name.casefold()
    ):
        archive_records = records_by_archive[archive]
        LOGGER.info(
            "Processing archive %s (%d images)",
            archive.name,
            len(archive_records),
        )
        with ZipFile(archive) as zip_file:
            for record in archive_records:
                split = assignments[record.patient_key]
                output_path = (
                    output_dir / split / record.label / record.filename
                )
                image = _decode_image(zip_file.read(record.member_name))
                processed = preprocess_xray(image, config=config)
                if processed is None or not cv2.imwrite(
                    str(output_path), processed
                ):
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
                processed_count += 1
                if (
                    (
                        config.process_log_interval
                        and processed_count % config.process_log_interval == 0
                    )
                    or processed_count == len(records)
                ):
                    LOGGER.info(
                        "Processed %d/%d images",
                        processed_count,
                        len(records),
                    )
    with (
        output_dir / config.manifest_filename
    ).open("w", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=config.manifest_fields,
        )
        writer.writeheader()
        writer.writerows(
            sorted(manifest_rows, key=lambda row: row["filename"])
        )

    LOGGER.info("Wrote %d unique images to %s", len(records), output_dir)
    for split in config.splits:
        LOGGER.info("%s: %s", split, counts[split])
    return counts


def parse_args(
    config: DatasetConfig = CONFIG,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_filepath",
        nargs="?",
        type=Path,
        default=config.raw_dir,
        help=(
            "Directory containing raw ZIP archives "
            f"(default: {config.raw_dir})"
        ),
    )
    parser.add_argument(
        "output_filepath",
        nargs="?",
        type=Path,
        default=config.processed_dir,
        help=(
            "Processed dataset directory "
            f"(default: {config.processed_dir})"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Deterministic split seed "
            f"(default: {config.default_seed})"
        ),
    )
    parser.add_argument(
        "--train-archives",
        nargs="+",
        metavar="ARCHIVE",
        help="Archives whose images must be assigned to train",
    )
    parser.add_argument(
        "--validation-archives",
        nargs="+",
        metavar="ARCHIVE",
        help="Archives eligible for validation",
    )
    parser.add_argument(
        "--test-archives",
        nargs="+",
        metavar="ARCHIVE",
        help="Archives eligible for test",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=CONFIG.log_level,
        format=CONFIG.log_format,
    )
    build_dataset(
        args.input_filepath,
        args.output_filepath,
        seed=args.seed,
        train_archives=args.train_archives,
        validation_archives=args.validation_archives,
        test_archives=args.test_archives,
    )


if __name__ == "__main__":
    main()
