from __future__ import annotations

import cv2
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List

import imutils

from train import PostureLabel
from util.image_type import ColorImage


class PostureLabelAnnotator:
    ANNOTATED_IMG_DIR = Path.cwd() / "posture/test"

    def __init__(self, filename: str) -> None:
        self._video_file: Path = (Path.cwd() / filename).resolve()
        self._video = cv2.VideoCapture(
            str(self._video_file)
        )
        self._labels: Dict[PostureLabel, List[ColorImage]] = {}
        for label in PostureLabel:
            self._labels[label] = []

    def start_annotating(self) -> None:
        print(f"Annotating blinks on {self._video_file.name}, which has "
              f"{self._video.get(cv2.CAP_PROP_FRAME_COUNT)} frames...")

        for frame_no, frame in enumerate(self._frames_from_video()):
            cv2.destroyAllWindows()
            self._show_frame_in_middle_of_screen(frame_no, frame)
            self._annotate_frame_by_key(frame)

    def write_annotations(self) -> None:
        print(f"Writing the annotations into {self.ANNOTATED_IMG_DIR}...")

        for label, images in self._labels.items():
            for i, image in enumerate(images):
                cv2.imwrite(str(self.ANNOTATED_IMG_DIR / label.name.lower() / f"{i}.jpg"), image)

    def __enter__(self) -> PostureLabelAnnotator:
        return self

    def __exit__(self, *exc_info) -> None:
        self.write_annotations()

    @staticmethod
    def _show_frame_in_middle_of_screen(frame_no: int, frame: ColorImage) -> None:
        win_name = f"no. {frame_no}"
        cv2.namedWindow(win_name)
        cv2.moveWindow(win_name, 250, 80)
        cv2.imshow(win_name, imutils.resize(frame, width=900))

    def _annotate_frame_by_key(self, frame: ColorImage) -> None:
        while True:
            self._read_key()
            if self._is_valid_key():
                self._labels[PostureLabel(int(self._key))].append(frame)
                break

    def _read_key(self) -> None:
        self._key = chr(cv2.waitKey() & 0xFF)

    def _is_valid_key(self) -> bool:
        try:
            int(self._key)
        except ValueError:
            return False
        return int(self._key) in PostureLabel.__members__.values()

    def _frames_from_video(self) -> Iterator[ColorImage]:
        while self._video.isOpened():
            ret, frame = self._video.read()
            if not ret:
                break
            yield frame


def main(video_to_annotate: str) -> None:
    with PostureLabelAnnotator(video_to_annotate) as annotator:
        annotator.start_annotating()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError(f"\n\t usage: python {__file__} $(video_to_annotate)")

    main(sys.argv[1])
