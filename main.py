import logging
from pathlib import Path

import cv2
from tensorflow.keras import models

from train import PostureLabel


def main() -> None:
    model = models.load_model(Path.cwd() / "model")

    image = cv2.imread(str(Path.cwd() / "posture/samples/good/0000.jpg"), cv2.IMREAD_GRAYSCALE)
    # image = cv2.resize(image, (224, 224))
    posture: PostureLabel
    conf: Float[32]
    posture, conf = model.predict(image)
    logging.info(f"{posture}, {conf}")
    # print(help(model.__call__))


if __name__ == "__main__":
    main()
