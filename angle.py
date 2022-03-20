import math
import warnings
from typing import List



def angle_between(p1, p2) -> float:
    x1, y1 = p1
    x2, y2 = p2
    with warnings.catch_warnings():
        # RuntimeWarning: divide by zero encountered in long_scalars
        # Ignore possible warning when 90.0
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return math.atan((y2 - y1) / (x2 - x1)) * 180 / math.pi


def get_mtcnn_angle(face) -> float:
    horizontal: float = (
        angle_between(face["keypoints"]["right_eye"], face["keypoints"]["left_eye"])
        + angle_between(face["keypoints"]["mouth_right"], face["keypoints"]["mouth_left"])
    ) / 2

    return horizontal


NOSE_BRIDGE_IDXS:   List[int] = [27, 30]
LEFT_EYESIDE_IDXS:  List[int] = [36, 39]
RIGHT_EYESIDE_IDXS: List[int] = [42, 45]
MOUTHSIDE_IDXS:     List[int] = [48, 54]
def get_hog_angle(landmarks) -> float:
    # average the eyes and mouth, they're all horizontal parts
    horizontal: float = (
        (angle_between(landmarks[RIGHT_EYESIDE_IDXS[0]], landmarks[RIGHT_EYESIDE_IDXS[1]])
         + angle_between(landmarks[LEFT_EYESIDE_IDXS[0]], landmarks[LEFT_EYESIDE_IDXS[1]])
        ) / 2
        + angle_between(landmarks[MOUTHSIDE_IDXS[0]], landmarks[MOUTHSIDE_IDXS[1]])
    ) / 2
    vertical: float = angle_between(landmarks[NOSE_BRIDGE_IDXS[0]], landmarks[NOSE_BRIDGE_IDXS[1]])
    # Under normal situations (not so close to the middle line):
    # If skews right, horizontal is positive, vertical is negative, e.g., 15 and -75;
    # if skews left, horizontal is negative, vertical is positive, e.g., -15 and 75.
    # Sum of their absolute value is approximately 90 (usually a bit larger).

    # When close to the middle, it's possible that both values are of the
    # same sign.

    # Horizontal value is already what we want, vertical value needs to be adjusted.
    # And again, we take their average.
    angle: float = (
        (horizontal + (vertical + 90.0)) / 2
        if vertical < 0 else
        (horizontal + (vertical - 90.0)) / 2
    )
    return angle
