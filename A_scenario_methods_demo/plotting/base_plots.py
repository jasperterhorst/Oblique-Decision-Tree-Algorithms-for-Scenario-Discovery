"""
Base plotting routines for the scenario methods demo.
"""

import os
from typing import Callable, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from A_scenario_methods_demo.utils import setup_matplotlib
from src.config import (
    SCATTER_COLORS,
    AXIS_LINE_COLOR,
    GRID_COLOR,
    QUADRILATERAL_COLOR
)


def generic_plot(
    title: str,
    xlabel: str,
    ylabel: str,
    note: Optional[str],
    save_path: Optional[str],
    draw_func: Callable[..., None],
    display_figsize: tuple[float, float] = (4, 3),
    save_figsize: tuple[float, float] = (6, 5),
    grid: bool = True
) -> None:
    """
    Create a plot with standardized styling.

    Parameters:
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        note (str | None): Optional note to show under the plot.
        save_path (str | None): File path to save the plot.
        draw_func (Callable[[Axes], None]): Drawing function that receives the axis.
        display_figsize (tuple): Size used when displaying interactively.
        save_figsize (tuple): Size used when saving.
        grid (bool): Whether to include a background grid.
    """
    setup_matplotlib()
    fig_size = save_figsize if save_path else display_figsize
    fig, ax = plt.subplots(figsize=fig_size)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    draw_func(ax)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(AXIS_LINE_COLOR)
    ax.spines["bottom"].set_linewidth(1)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(AXIS_LINE_COLOR)
    ax.spines["left"].set_linewidth(1)

    if grid:
        ax.grid(True, linestyle="--", linewidth=0.5, color=GRID_COLOR)

    ax.tick_params(axis="both", labelsize=14, colors="gray")
    ax.set_title(title, fontsize=22, pad=15)
    ax.set_xlabel(xlabel, fontsize=18, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=18, labelpad=10)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="lower right", fontsize=14, frameon=True)

    if note:
        fig.text(0.5, 0.01, note, va="bottom", ha="center", fontsize=14, color="gray")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()
    plt.close(fig)


def plot_base(
    ax: Axes,
    samples: np.ndarray,
    y: np.ndarray,
    quadrilateral: np.ndarray,
    quadrilateral_label: str = "Sampled Quadrilateral",
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None
) -> None:
    """
    Plot classified sample points and a background quadrilateral.

    Parameters:
        ax (Axes): Target axis.
        samples (np.ndarray): Sample points of shape (n, 2).
        y (np.ndarray): Binary labels (0 or 1).
        quadrilateral (np.ndarray): Coordinates of quadrilateral polygon.
        quadrilateral_label (str): Label for the polygon.
        xlim (tuple | None): x-axis limits.
        ylim (tuple | None): y-axis limits.
    """
    ax.scatter(samples[y == 0, 0], samples[y == 0, 1],
               c=SCATTER_COLORS["no_interest"], s=20, edgecolors="none", zorder=3,
               label="Not of Interest")
    ax.scatter(samples[y == 1, 0], samples[y == 1, 1],
               c=SCATTER_COLORS["interest"], s=20, edgecolors="none", zorder=3,
               label="Of Interest")

    quadri = Polygon(quadrilateral, closed=True, fill=True,
                     facecolor=QUADRILATERAL_COLOR, alpha=0.6,
                     edgecolor="none", zorder=0, label=quadrilateral_label)
    ax.add_patch(quadri)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
