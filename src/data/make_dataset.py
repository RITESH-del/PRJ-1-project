"""Compatibility wrapper for the packaged dataset builder."""

from tb_detection_model.data.make_dataset import *  # noqa: F401,F403
from tb_detection_model.data.make_dataset import main


if __name__ == "__main__":
    main()
