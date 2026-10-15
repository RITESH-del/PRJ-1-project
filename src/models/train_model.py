"""Script and compatibility wrapper for model training."""

from tb_detection_model.models.train_model import *  # noqa: F401,F403
from tb_detection_model.models.train_model import main

if __name__ == "__main__":
    main()
