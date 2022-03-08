import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, utils

from train import ModelTrainer, PostureLabel


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


def predict_video_stream(videopath: Optional[str] = None) -> None:
    """
    Arguments:
        videopath: camera in default.
    """
    if videopath is None:
        cam = cv2.VideoCapture(0)
    else:
        cam = cv2.VideoCapture(videopath)

    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")

        frame_exp = tf.expand_dims(frame, 0)

        predictions = model(frame_exp)
        score = tf.nn.softmax(predictions[0])

        label = PostureLabel(np.argmax(score))
        confidence = np.max(score)

        cv2.putText(
            frame,
            f"{label.name}, {confidence:.2%}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 0), 2
        )
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

def predict_images(folderpath: Union[Path, str]) -> None:
    test_ds = utils.image_dataset_from_directory(
        Path.cwd() / folderpath,
        shuffle=False,
        image_size=ModelTrainer.IMAGE_SIZE,
        batch_size=64,
        # color_mode="grayscale"
    )
    predictions = model.predict(test_ds)
    scores = tf.nn.softmax(predictions)
    for score in scores:
        logging.info(PostureLabel(np.argmax(score)).name)


def evaluate_images(folderpath: Union[Path, str]) -> None:
    eval_ds = utils.image_dataset_from_directory(
        Path.cwd() / folderpath,
        seed=123,
        image_size=ModelTrainer.IMAGE_SIZE,
        batch_size=64,
        # color_mode="grayscale"
    )
    model.evaluate(eval_ds)


def main() -> None:
    # evaluate_images("posture/test")
    if len(sys.argv) > 1:
        predict_video_stream(sys.argv[1])


if __name__ == "__main__":
    main()
