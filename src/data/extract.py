"""Compatibility wrapper for the packaged extraction workflow."""

from tb_detection_model.data.extract import *  # noqa: F401,F403
from tb_detection_model.data.extract import extract_all_datasets


if __name__ == "__main__":
    extract_all_datasets()
