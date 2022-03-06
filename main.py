import logging
from pathlib import Path

import cv2
from tensorflow.keras import models

from train import PostureLabel


def main() -> None:
    model = models.load_model(Path.cwd() / "model")


if __name__ == "__main__":
    main()
