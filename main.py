import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import models, utils

from train import PostureLabel


def main() -> None:
    model = models.load_model(Path.cwd() / "model")
    image = utils.load_img(
        Path.cwd() / "posture/samples/good/0001.jpg"
    )
    image_array = utils.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)

    predictions = model(image_array)
    score = tf.nn.softmax(predictions[0])
    logging.info(
        f"This image most likely belongs to {PostureLabel(np.argmax(score)).name} "
        f"with a {np.max(score):.2%} percent confidence."
    )


if __name__ == "__main__":
    main()
