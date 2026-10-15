from pathlib import Path
from zipfile import ZipFile
import random


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

SAMPLES_PER_DATASET = 100


def create_sample(
    zip_path: Path,
    sample_size: int,
    interim_dir: Path = INTERIM_DIR,
    rng: random.Random | None = None,
) -> None:
    dataset_name = zip_path.stem
    output_dir = interim_dir / dataset_name

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
        sampler = random.sample if rng is None else rng.sample
        samples = sampler(files, sample_size)

        print(
            f"{dataset_name}: "
            f"sampling {sample_size}/{len(files)} files"
        )

        for info in samples:
            destination = output_dir / info.filename

            if not destination.resolve().is_relative_to(
                output_dir.resolve()
            ):
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)

            with z.open(info) as source, destination.open("wb") as target:
                target.write(source.read())

    print(f"Created: {output_dir}")


def create_interim_samples(
    raw_dir: Path = RAW_DIR,
    interim_dir: Path = INTERIM_DIR,
    sample_size: int = SAMPLES_PER_DATASET,
    seed: int | None = None,
) -> None:
    zip_files = sorted(raw_dir.glob("*.zip"))

    if not zip_files:
        print(f"No ZIP files found in {raw_dir}")
        return

    rng = random.Random(seed) if seed is not None else None
    for zip_path in zip_files:
        create_sample(zip_path, sample_size, interim_dir, rng)


if __name__ == "__main__":
    create_interim_samples()
