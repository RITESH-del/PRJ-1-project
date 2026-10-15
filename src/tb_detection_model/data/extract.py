from pathlib import Path
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"


def extract_all_datasets(
    raw_dir: Path = RAW_DIR,
    interim_dir: Path = INTERIM_DIR,
) -> None:
    zip_files = sorted(raw_dir.glob("*.zip"))

    if not zip_files:
        print(f"No ZIP files found in {raw_dir}")
        return

    for zip_path in zip_files:
        dataset_name = zip_path.stem
        output_dir = interim_dir / dataset_name

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nExtracting: {zip_path.name}")
        print(f"Destination: {output_dir}")

        with ZipFile(zip_path, "r") as zip_file:
            for info in zip_file.infolist():
                destination = output_dir / info.filename
                if not destination.resolve().is_relative_to(
                    output_dir.resolve()
                ):
                    continue
                if info.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zip_file.open(info) as source:
                    with destination.open("wb") as target:
                        target.write(source.read())

        print(f"Finished: {dataset_name}")

    print("\nAll datasets extracted.")


if __name__ == "__main__":
    extract_all_datasets()
