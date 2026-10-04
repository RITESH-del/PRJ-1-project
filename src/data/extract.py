from pathlib import Path
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"


def extract_all_datasets():
    zip_files = list(RAW_DIR.glob("*.zip"))

    if not zip_files:
        print("No ZIP files found in data/raw/")
        return

    for zip_path in zip_files:
        dataset_name = zip_path.stem
        output_dir = INTERIM_DIR / dataset_name

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nExtracting: {zip_path.name}")
        print(f"Destination: {output_dir}")

        with ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(output_dir)

        print(f"Finished: {dataset_name}")

    print("\nAll datasets extracted.")


if __name__ == "__main__":
    extract_all_datasets()