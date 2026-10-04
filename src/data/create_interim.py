from pathlib import Path
from zipfile import ZipFile
import random
import shutil
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

SAMPLES_PER_DATASET = 100


def create_sample(zip_path: Path, sample_size: int):
    dataset_name = zip_path.stem
    output_dir = INTERIM_DIR / dataset_name

    output_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(zip_path, "r") as z:
        files = [
            info
            for info in z.infolist()
            if not info.is_dir()
        ]

        if not files:
            print(f"Skipping {dataset_name}: no files found")
            return

        sample_size = min(sample_size, len(files))
        samples = random.sample(files, sample_size)

        print(
            f"{dataset_name}: "
            f"sampling {sample_size}/{len(files)} files"
        )

        for info in samples:
            destination = output_dir / info.filename

            # Prevent path traversal from malicious ZIP entries
            if not destination.resolve().is_relative_to(
                output_dir.resolve()
            ):
                continue

            destination.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with z.open(info) as source, open(destination, "wb") as target:
                shutil.copyfileobj(source, target)

    print(f"Created: {output_dir}")


def create_interim_samples():
    zip_files = list(RAW_DIR.glob("*.zip"))

    if not zip_files:
        print("No ZIP files found in data/raw/")
        return

    for zip_path in zip_files:
        create_sample(zip_path, SAMPLES_PER_DATASET)


if __name__ == "__main__":
    create_interim_samples()