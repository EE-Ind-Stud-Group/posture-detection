"""
Specify the target image directories and the destination.
All images from the target will be dumped into a list,
renamed with consecutive numbers and stored back into the destination.
"""

from pathlib import Path
from typing import List

import cv2


def main(targets: List[str], destination: str) -> None:
    # dump
    target_images = []
    for target in map(Path, targets):
        if not target.exists():
            print(f"path: {target} does not exist. Skipped.")
            continue

        for image in target.iterdir():
            target_images.append(cv2.imread(str(image)))

    # rename and store
    if not Path(destination).exists():
        print(f"path: {destination} does not exist. Created.")
        Path(destination).mkdir(parents=True)

    for i, image in enumerate(target_images):
        cv2.imwrite(f"{destination}/{i}.jpg", image)


if __name__ == "__main__":
    main(targets=["./posture/samples/slump", "./posture/test/slump"],
         destination="./posture/samples_/slump")
