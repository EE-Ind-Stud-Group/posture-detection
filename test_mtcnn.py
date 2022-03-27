"""This file uses only MTCNN."""

import cv2
from mtcnn import MTCNN


GREEN = (0, 255, 0)

detector = MTCNN()
cam = cv2.VideoCapture(0)
while cam.isOpened():
    _, img = cam.read()

    faces = detector.detect_faces(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    )
    for face in faces:
        cv2.rectangle(
            img,
            face["box"][:2],
            (face["box"][0] + face["box"][2], face["box"][1] + face["box"][3]),
            GREEN, 2
        )

        for point in face["keypoints"].values():
            cv2.circle(img, point, 0, GREEN, 2)

    cv2.imshow("img", img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
