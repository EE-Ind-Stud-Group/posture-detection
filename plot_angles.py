"""Plots the angles from files: good_angle.log and slump_angle.log."""

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    # load angles
    good_angles = list(map(float, (Path(__file__).parent / "good_angle.log").read_text().split()))
    slump_angles = list(map(float, (Path(__file__).parent / "slump_angle.log").read_text().split()))

    _, ax = plt.subplots(1)
    ax.set_title("Detected Angles")
    # angles
    ax.scatter(range(len(good_angles)), good_angles, color="r", alpha=0.7, s=12, label="good angle")
    ax.scatter(range(len(slump_angles)), slump_angles, color="b", alpha=0.7, s=12, label="slump angle")
    ax.set_xlim(0, len(good_angles))

    def plot_portions(portions_of_good: Iterable[float], line_styles: Iterable[str]) -> None:
        for portion, line_style in zip(portions_of_good, line_styles):
            bound = sorted(good_angles, key=abs)[int(len(good_angles) * portion)]
            ax.hlines([bound, -bound], 0, len(good_angles), color="black", linestyle=line_style)
            # add specific ticks
            ax.set_yticks(np.concatenate((ax.get_yticks(), [bound, -bound])))
            # calculate how many slumps inside the bound of good
            inside = list(filter(lambda x: abs(x) <= abs(bound), slump_angles))
            ax.text(len(slump_angles) + 50, bound + 2,
                    f"{len(inside) / len(slump_angles):.01%} of slumps inside the {portion:.01%} bound of good")

    plot_portions(portions_of_good=[0.995, 0.95], line_styles=["dashed", "dotted"])
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
