"""This file uses only self-trained model."""

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, utils

from train_model import ModelTrainer, PostureLabel


model = models.load_model(Path(__file__).parent / "model")


def predict_single_image(imagepath: Union[Path, str]) -> None:
    image = utils.load_img(Path(__file__).parent / imagepath)
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
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_exp = tf.expand_dims(frame, 0)

        predictions = model(frame_exp)
        score = tf.nn.softmax(predictions[0])

        label = PostureLabel(np.argmax(score))
        confidence = np.max(score)

        cv2.putText(
            frame,
            f"{label.name}, {confidence:.2%}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 0, 0), 2
        )
        cv2.imshow("Frame", frame)
        # otherwise the video stream is fast forward
        key = cv2.waitKey(50) & 0xFF

        if key == ord("q"):
            break


def predict_images(folderpath: Union[Path, str]) -> None:
    test_ds = utils.image_dataset_from_directory(
        Path(__file__).parent / folderpath,
        shuffle=False,
        image_size=ModelTrainer.IMAGE_SIZE,
        batch_size=ModelTrainer.BATCH_SIZE,
        color_mode="grayscale"
    )
    predictions = model.predict(test_ds)
    scores = tf.nn.softmax(predictions)
    for score in scores:
        logging.info(PostureLabel(np.argmax(score)).name)


def evaluate_images(folderpath: Union[Path, str]) -> None:
    eval_ds = utils.image_dataset_from_directory(
        Path(__file__).parent / folderpath,
        seed=123,
        image_size=ModelTrainer.IMAGE_SIZE,
        batch_size=ModelTrainer.BATCH_SIZE,
        color_mode="grayscale"
    )
    model.evaluate(eval_ds)


def main() -> None:
    # evaluate_images("posture/tests")
    with tf.device("/gpu:0"):
        if len(sys.argv) > 1:
            predict_video_stream(sys.argv[1])
        else:
            predict_video_stream()


if __name__ == "__main__":
    main()
