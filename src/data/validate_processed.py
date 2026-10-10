# Data validation for the processed dataset
#
# * Checks that each PNG file has a valid PNG header and non‑zero size.
# * Confirms the manifest contains exactly one entry per file and that the label is either "TB" or "normal".
# * Generates a human‑readable report at ``reports/data_validation.txt``.
#
# Run via ``python src/data/validate_processed.py`` or ``make validate-data``.

import csv
import logging
from pathlib import Path

def is_valid_png(path: Path) -> bool:
    """Return True if *path* points to a non‑empty file with a valid PNG header.
    Mirrors the lightweight validation used in ``src/data/make_dataset.py``.
    """
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        with path.open("rb") as f:
            header = f.read(8)
        return header == b"\x89PNG\r\n\x1a\n"
    except Exception:
        return False
    """Return a dict mapping filename → (label, split)."""
    rows = {}
    with MANIFEST.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["filename"]] = (row["label"], row["split"])
    return rows


def compute_histogram(manifest_rows):
    """Return a dict ``{split: {label: count}}`` from manifest rows.
    ``manifest_rows`` is a list of (filename, label, split) tuples.
    """
    hist = {}
    for _, label, split in manifest_rows:
        hist.setdefault(split, {}).setdefault(label, 0)
        hist[split][label] += 1
    return hist

def find_duplicates(manifest_rows):
    """Return a list of filenames that appear in more than one split.
    ``manifest_rows`` is the list of (filename, label, split) tuples.
    """
    seen = {}
    dup = []
    for fname, _, split in manifest_rows:
        if fname in seen and seen[fname] != split:
            dup.append(fname)
        else:
            seen[fname] = split
    return dup

def write_report(errors: list[str], hist: dict, duplicates: list[str]):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("w") as f:
        if not errors:
            f.write("All processed images passed validation.\n\n")
        else:
            f.write("Validation errors found:\n")
            for e in errors:
                f.write(f"- {e}\n")
            f.write("\n")
        f.write("Label distribution per split:\n")
        for split, lbls in sorted(hist.items()):
            f.write(f"- {split}: ")
            f.write(", ".join(f"{lbl}:{cnt}" for lbl, cnt in sorted(lbls.items())))
            f.write("\n")
        if duplicates:
            f.write("\nDuplicate filenames across splits detected:\n")
            for d in duplicates:
                f.write(f"- {d}\n")
        else:
            f.write("\nNo duplicate filenames across splits.\n")


def validate_images(manifest: dict) -> list[str]:
    errors = []
    for split in ("train", "val", "test"):
        split_dir = PROCESSED_DIR / split
        for img_path in split_dir.glob("*.png"):
            fname = img_path.name
            if fname not in manifest:
                errors.append(f"{split}/{fname} missing from manifest")
                continue
            label, _ = manifest[fname]
            if label not in VALID_LABELS:
                errors.append(f"{split}/{fname} has invalid label '{label}'")
            if not is_valid_png(img_path):
                errors.append(f"{split}/{fname} is not a valid PNG file")
    return errors



def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Loading manifest …")
    manifest = load_manifest()

    logger.info("Validating %d images …", len(manifest))
    errors = validate_images(manifest)

    # Enrich report with distribution and duplicate detection
    manifest_rows = [(fname, lbl, split) for fname, (lbl, split) in manifest.items()]
    hist = compute_histogram(manifest_rows)
    duplicates = find_duplicates(manifest_rows)

    logger.info("Writing report …")
    write_report(errors, hist, duplicates)

    if errors:
        logger.error("%d validation errors detected", len(errors))
    else:
        logger.info("Validation successful – no errors.")

if __name__ == "__main__":
    main()
