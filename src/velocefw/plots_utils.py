"""Functions that help with formatting plots."""

import numpy as np
from matplotlib.axes import Axes

WIDTH_JOURNAL_COLUMN = 3.46  # inches


def format_xticks(
    axs: Axes,
    x_ticks: np.ndarray,
    x_ticks_minor: np.ndarray,
    label_size: int = 10,
) -> None:
    """Format the x-axis ticks."""
    axs.set_xticks(x_ticks, minor=False)
    axs.set_xticks(x_ticks_minor, minor=True)

    axs.xaxis.set_tick_params(
        which="major",
        direction="in",
        length=4,
        width=1.2,
        top=True,
        bottom=True,
        labeltop=False,
        labelbottom=True,
        labelsize=label_size,
    )
    axs.xaxis.set_tick_params(
        which="minor",
        direction="in",
        length=2,
        width=0.8,
        top=True,
        bottom=True,
    )


def format_yticks(
    axs: Axes,
    y_ticks: np.ndarray,
    y_ticks_minor: np.ndarray,
    label_size: int = 10,
) -> None:
    """Format the y-axis ticks."""
    axs.set_yticks(y_ticks, minor=False)
    axs.set_yticks(y_ticks_minor, minor=True)

    axs.yaxis.set_tick_params(
        which="major",
        direction="in",
        length=4,
        width=1.2,
        left=True,
        right=True,
        labelsize=label_size,
    )
    axs.yaxis.set_tick_params(
        which="minor",
        direction="in",
        length=2,
        width=0.8,
        left=True,
        right=True,
        labelleft=False,
    )
