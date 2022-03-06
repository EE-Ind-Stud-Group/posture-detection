import logging
from enum import IntEnum
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from nptyping import Float, Int, NDArray, UInt8

from tensorflow import keras
from tensorflow.keras import layers, losses, models, utils

from util.image_type import ColorImage, GrayImage


logging.basicConfig(format="%(message)s", level=logging.INFO)


class PostureLabel(IntEnum):
    """Posture can be good or slump."""
    # The values should start from 0 and be consecutive
    # since they"re also used to represent the result of prediction.
    GOOD:  int = 0
    SLUMP: int = 1


class ModelTrainer:
    SAMPLE_DIR = Path.cwd() / "posture/samples"
    IMAGE_SIZE: Tuple[int, int] = (480, 640)
    BATCH_SIZE = 15

    @property
    def model(self) -> models.Sequential:
        return self._model

    def _load_image_dataset(self) -> None:
        self._train_ds = utils.image_dataset_from_directory(
            self.SAMPLE_DIR,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=self.IMAGE_SIZE,
            batch_size=self.BATCH_SIZE
        )
        self._val_ds = utils.image_dataset_from_directory(
            self.SAMPLE_DIR,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=self.IMAGE_SIZE,
            batch_size=self.BATCH_SIZE
        )

    def _create_model(self) -> None:
        self._create_data_augmentation()
        self._model = models.Sequential([
            self._data_augmentation,
            layers.Resizing(*self.IMAGE_SIZE),
            layers.Rescaling(1./255, input_shape=(*self.IMAGE_SIZE, 3)),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(len(self._train_ds.class_names))
        ])

    def _compile_model(self) -> None:
        self._model.compile(
            optimizer="adam",
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

    def _create_data_augmentation(self) -> None:
        self._data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal", input_shape=(*self.IMAGE_SIZE, 3)),
            layers.RandomZoom(0.1),
        ])

    def train_model(self) -> None:
        self._load_image_dataset()
        self._create_model()
        self._compile_model()

        self._epochs = 8
        self._history = self._model.fit(
            self._train_ds,
            validation_data=self._val_ds,
            epochs=self._epochs
        )

    def save_model(self) -> None:
        self._model.save(Path.cwd() / "model")

    def visualize_training_results(self) -> None:
        acc = self._history.history["accuracy"]
        val_acc = self._history.history["val_accuracy"]

        loss = self._history.history["loss"]
        val_loss = self._history.history["val_loss"]

        epochs_range = range(self._epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label="Training Accuracy")
        plt.plot(epochs_range, val_acc, label="Validation Accuracy")
        plt.legend(loc="lower right")
        plt.title("Training and Validation Accuracy")

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label="Training Loss")
        plt.plot(epochs_range, val_loss, label="Validation Loss")
        plt.legend(loc="upper right")
        plt.title("Training and Validation Loss")
        plt.show()


def main() -> None:
    trainer = ModelTrainer()
    trainer.train_model()
    trainer.save_model()
    trainer.visualize_training_results()
    trainer.model.summary()


if __name__ == "__main__":
    main()
