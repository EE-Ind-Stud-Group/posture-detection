"""Plots the angles from files: good_angle.log and slump_angle.log."""

from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    good_angles = list(map(float, (Path(__file__).parent / "good_angle.log").read_text().split()))
    slump_angles = list(map(float, (Path(__file__).parent / "slump_angle.log").read_text().split()))

    fig, ax = plt.subplots(1)
    ax.scatter(range(len(good_angles)), good_angles, color="r", alpha=0.7, label="good angle")
    ax.scatter(range(len(slump_angles)), slump_angles, color="b", alpha=0.7, label="slump angle")
    ax.set(ylim=(-90, 90))
    ax.set_title('MTCNN detected angles')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
