"""Compatibility wrapper for the packaged interim workflow."""

from tb_detection_model.data.create_interim import *  # noqa: F401,F403
from tb_detection_model.data.create_interim import create_interim_samples


if __name__ == "__main__":
    create_interim_samples()
