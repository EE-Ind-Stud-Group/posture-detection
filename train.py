import logging
from enum import IntEnum
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from nptyping import Float, Int, NDArray, UInt8
from sklearn.utils import class_weight
from tensorflow.keras import layers, models
from tensorflow.python.keras.engine.sequential import Sequential

from util.image_type import ColorImage, GrayImage


logging.basicConfig(format='%(message)s', level=logging.INFO)


class PostureLabel(IntEnum):
    """Posture can be good or slump."""
    # The values should start from 0 and be consecutive
    # since they're also used to represent the result of prediction.
    GOOD:  int = 0
    SLUMP: int = 1


SAMPLE_DIR: str = Path.cwd() / "posture/samples"
IMAGE_DIMENSIONS: Tuple[int, int] = (640, 480)

def main() -> None:
    train_images: List[GrayImage] = []
    train_labels: List[int] = []

    # The order(index) of the class is the order of the ints that PostureLabels represent.
    # i.e., GOOD.value = 0, SLUMP.value = 1
    # So is clear when the result of the prediction is 1, we know it's slump.
    for label in PostureLabel:
        logging.info(f"Training with label {label.name.lower()}...")

        image_folder = Path.cwd() / f"{SAMPLE_DIR}/{label.name.lower()}"
        for image_path in image_folder.iterdir():
            logging.info(image_path)
            image: GrayImage = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, IMAGE_DIMENSIONS)
            train_images.append(image)
            train_labels.append(label)

    # numpy array with GrayImages
    images: NDArray[(Any, Any, Any), UInt8] = np.array(train_images)
    labels: NDArray[(Any,), Int[32]] = np.array(train_labels)
    # images = images / 255  # Normalize image
    images = images.reshape(len(images), *IMAGE_DIMENSIONS, 1)
    print(images.shape)

    weights: NDArray[(Any,), Float[64]] = class_weight.compute_sample_weight("balanced", labels)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(*IMAGE_DIMENSIONS, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(len(PostureLabel), activation="softmax"))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(images, labels, epochs=5, sample_weight=weights)
    model.save(Path.cwd() / "model")

    logging.info("Training finished.")


if __name__ == "__main__":
    main()
