"""A three-layer posture detection with HOG (dlib), MTCNN and self-trained model."""

import time
from pathlib import Path

import cv2
import dlib
import numpy as np
import tensorflow as tf
from imutils import face_utils
from mtcnn import MTCNN
from tensorflow.keras import models

import angle
from train_model import PostureLabel


# dlib hog
hog_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(
    str(Path(__file__).parent
    / "dlib_model/shape_predictor_68_face_landmarks.dat")
)
# mtcnn
mtcnn_detector = MTCNN()
# self-trained model
model = models.load_model(Path(__file__).parent / "model")


ANGLE_THRESHOLD = 15
def is_good(ang) -> bool:
    return abs(ang) <= ANGLE_THRESHOLD


def main() -> None:
    cam = cv2.VideoCapture(0)
    while cam.isOpened():
        ret, frame = cam.read()

        # to see the difference of time consumption
        start = time.perf_counter()
        faces = hog_detector(frame)
        if faces:
            # layer 1: hog
            landmarks = face_utils.shape_to_np(shape_predictor(frame, faces[0]))
            res = f"hog: {is_good(angle.get_hog_angle(landmarks))}"
        else:
            start = time.perf_counter()
            # layer 2: mtcnn
            faces = mtcnn_detector.detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if faces:
                res = f"mtcnn: {is_good(angle.get_mtcnn_angle(faces[0]))}"
            else:
                start = time.perf_counter()
                # layer 3: self-trained model
                frame_exp = tf.expand_dims(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 0)
                predictions = model(frame_exp)
                score = tf.nn.softmax(predictions[0])
                res = f"model: {PostureLabel(np.argmax(score)) is PostureLabel.GOOD}"
        end = time.perf_counter()
        cv2.putText(frame, res, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 0, 0), 2)
        cv2.putText(frame, f"Elapsed: {end - start:.04f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


if __name__ == "__main__":
    main()
