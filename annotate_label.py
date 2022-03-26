"""Annotates videos with good/slump."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, List

import cv2
import imutils

from train import PostureLabel
from util.image_type import ColorImage


class PostureLabelAnnotator:
    ANNOTATED_IMG_DIR = Path(__file__).parent / "posture/test"

    def __init__(self, filename: str) -> None:
        self._video_file: Path = (Path(__file__).parent / filename).resolve()
        self._video = cv2.VideoCapture(
            str(self._video_file)
        )

    def start_annotating(self) -> None:
        print(f"Annotating blinks on {self._video_file.name}, which has "
              f"{self._video.get(cv2.CAP_PROP_FRAME_COUNT)} frames...")

        for frame_no, frame in enumerate(self._frames_from_video()):
            self._frame_no = frame_no
            self._frame = frame
            cv2.destroyAllWindows()
            self._show_frame_in_middle_of_screen()
            self._annotate_frame_by_key()

    def _show_frame_in_middle_of_screen(self) -> None:
        win_name = f"no. {self._frame_no}"
        cv2.namedWindow(win_name)
        cv2.moveWindow(win_name, 250, 80)
        cv2.imshow(win_name, imutils.resize(self._frame, width=900))

    def _annotate_frame_by_key(self) -> None:
        while True:
            self._read_key()
            if self._is_discard_key():
                # discard the image by skipping
                break
            if self._is_valid_key():
                cv2.imwrite(
                    str(self.ANNOTATED_IMG_DIR
                        / PostureLabel(int(self._key)).name.lower()
                        / f"{self._frame_no}.jpg"),
                    self._frame
                )
                break

    def _read_key(self) -> None:
        self._key = chr(cv2.waitKey() & 0xFF)

    def _is_discard_key(self) -> bool:
        return self._key == "x"

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
    PostureLabelAnnotator(video_to_annotate).start_annotating()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError(f"\n\t usage: python {__file__} $(video_to_annotate)")

    main(sys.argv[1])
