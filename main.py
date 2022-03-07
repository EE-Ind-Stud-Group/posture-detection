import logging
from typing import Union
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import models, utils

from train import PostureLabel


model = models.load_model(Path.cwd() / "model")


def predict_single_image(imagepath: Union[Path, str]) -> None:
    image = utils.load_img(Path.cwd() / imagepath)
    image_array = utils.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)

    predictions = model(image_array)
    score = tf.nn.softmax(predictions[0])
    logging.info(
        f"This image most likely belongs to {PostureLabel(np.argmax(score)).name} "
        f"with a {np.max(score):.2%} percent confidence."
    )


def predict_images(folderpath: Union[Path, str]) -> None:
    test_ds = utils.image_dataset_from_directory(
        Path.cwd() / folderpath,
        shuffle=False,
        image_size=(480, 640),
        batch_size=15
    )
    predictions = model.predict(test_ds)
    scores = tf.nn.softmax(predictions)
    for score in scores:
        print(PostureLabel(np.argmax(score)).name)


def evaluate_images(folderpath: Union[Path, str]) -> None:
    eval_ds = utils.image_dataset_from_directory(
        Path.cwd() / folderpath,
        seed=123,
        image_size=(480, 640),
        batch_size=15
    )
    model.evaluate(eval_ds)


def main() -> None:
    # predict_single_image("posture/samples/good/0001.jpg")
    evaluate_images("posture/test")


if __name__ == "__main__":
    main()
