import logging
from enum import IntEnum
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.keras import layers, losses, models, utils


logging.basicConfig(format="%(message)s", level=logging.INFO)


class PostureLabel(IntEnum):
    """Posture can be good or slump."""
    # The values should start from 0 and be consecutive
    # since they"re also used to represent the result of prediction.
    GOOD:  int = 0
    SLUMP: int = 1


class ModelTrainer:
    SAMPLE_DIR = Path.cwd() / "posture/samples"
    IMAGE_SIZE: Tuple[int, int] = (240, 320)
    BATCH_SIZE = 16
    CH_NUM = 1
    EPOCHS = 3  # critical to under/overfitting

    @property
    def model(self) -> models.Sequential:
        return self._model

    def _load_image_dataset(self) -> None:
        self._train_ds = utils.image_dataset_from_directory(
            self.SAMPLE_DIR,
            # validation_split=0.2,
            # subset="training",
            seed=123,
            image_size=self.IMAGE_SIZE,
            batch_size=self.BATCH_SIZE,
            color_mode="grayscale"
        )
        # self._val_ds = utils.image_dataset_from_directory(
        #     self.SAMPLE_DIR,
        #     # validation_split=0.2,
        #     # subset="validation",
        #     seed=123,
        #     image_size=self.IMAGE_SIZE,
        #     batch_size=self.BATCH_SIZE,
        #     color_mode="grayscale"
        # )

    def k_fold_cross_validation(self, k: int) -> None:
        """See how accurate the model is."""
        # refer to
        # https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-keras.md
        image_set = []
        label_set = []
        for label in self.SAMPLE_DIR.iterdir():
            for image in label.iterdir():
                image_set.append(
                    cv2.resize(
                        cv2.imread(str(image), 0),
                        # the order of cv2 is (w, h)
                        (self.IMAGE_SIZE[1], self.IMAGE_SIZE[0])
                    )
                )
                label_set.append(PostureLabel[label.stem.upper()])
        image_set = np.array(image_set)
        label_set = np.array(label_set)

        acc_per_fold = []
        loss_per_fold = []
        kf = KFold(n_splits=k, shuffle=True)
        fold_no = 1
        for train, val in kf.split(image_set, label_set):
            # re-create every loop, otherwise it keeps training the same model
            self._create_model()
            self._compile_model()
            logging.info('------------------------------------------------------------------------')
            logging.info(f"Training for fold {fold_no} ...")
            # GPU may ran out of memory
            with tf.device("/cpu:0"):
                history = self._model.fit(
                    image_set[train],
                    label_set[train],
                    epochs=self.EPOCHS,
                )
            scores = self._model.evaluate(image_set[val], label_set[val])
            logging.info(f"Score for fold {fold_no}: {self._model.metrics_names[0]} of {scores[0]}; "
                         f"{self._model.metrics_names[1]} of {scores[1]:.2%}")
            acc_per_fold.append(scores[1])
            loss_per_fold.append(scores[0])
            fold_no += 1

        logging.info("------------------------------------------------------------------------")
        logging.info("Score per fold")
        for i in range(len(acc_per_fold)):
            logging.info("------------------------------------------------------------------------")
            logging.info(f"> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]:.2%}")
        logging.info("------------------------------------------------------------------------")
        logging.info("Average scores for all folds:")
        logging.info(f"> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})")
        logging.info(f"> Loss: {np.mean(loss_per_fold)}")
        logging.info("------------------------------------------------------------------------")

    def _create_model(self) -> None:
        self._create_data_augmentation()
        self._model = models.Sequential([
            self._data_augmentation,
            layers.Resizing(*self.IMAGE_SIZE),
            layers.Rescaling(1./255, input_shape=(*self.IMAGE_SIZE, self.CH_NUM)),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(len(PostureLabel))
        ])

    def _compile_model(self) -> None:
        self._model.compile(
            optimizer="adam",
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

    def _create_data_augmentation(self) -> None:
        """Data can be flipped horizontally and zoomed slightly."""
        self._data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal", input_shape=(*self.IMAGE_SIZE, self.CH_NUM)),
            layers.RandomZoom(0.1),
        ])

    def train_model(self) -> None:
        self._load_image_dataset()
        self._create_model()
        self._compile_model()

        # tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs")
        with tf.device("/device:GPU:0"):
            self._history = self._model.fit(
                self._train_ds,
                # validation_data=self._val_ds,
                epochs=self.EPOCHS,
                # callbacks=[tensorboard_callback]
            )

    def save_model(self) -> None:
        self._model.save(Path.cwd() / "model")

    def visualize_training_results(self) -> None:
        acc = self._history.history["accuracy"]
        # val_acc = self._history.history["val_accuracy"]

        loss = self._history.history["loss"]
        # val_loss = self._history.history["val_loss"]

        epochs_range = range(self.EPOCHS)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label="Training Accuracy")
        # plt.plot(epochs_range, val_acc, label="Validation Accuracy")
        plt.legend(loc="lower right")
        plt.title("Training and Validation Accuracy")

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label="Training Loss")
        # plt.plot(epochs_range, val_loss, label="Validation Loss")
        plt.legend(loc="upper right")
        plt.title("Training and Validation Loss")
        plt.show()


def main() -> None:
    trainer = ModelTrainer()
    trainer.train_model()
    trainer.save_model()
    trainer.visualize_training_results()
    # trainer.k_fold_cross_validation(5)


if __name__ == "__main__":
    main()
