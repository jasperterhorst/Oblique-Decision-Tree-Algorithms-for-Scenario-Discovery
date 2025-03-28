"""
Base plotting routines for the scenario methods demo.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from A_scenario_methods_demo.utils import setup_matplotlib
from src.config.colors import (
    PRIMARY_LIGHT, PRIMARY_DARK,
    AXIS_LINE_COLOR, GRID_COLOR, QUADRILATERAL_COLOR
)


def generic_plot(title, xlabel, ylabel, note, save_path, draw_func,
                 display_figsize=(4, 3), save_figsize=(6, 5), grid=True):
    """
    Create a plot with standardized styling.
    """
    setup_matplotlib()
    fig_size = save_figsize if save_path else display_figsize
    fig, ax = plt.subplots(figsize=fig_size)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    draw_func(ax)

    # Style axes
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


# def generic_plot(title, xlabel, ylabel, note, save_path, draw_func,
#                  display_figsize=(4, 3), save_figsize=(6, 5),
#                  grid=True, show_legend=None):
#     setup_matplotlib()
#     fig_size = save_figsize if save_path else display_figsize
#     fig, ax = plt.subplots(figsize=fig_size)
#     fig.patch.set_facecolor("white")
#     ax.set_facecolor("white")
#     draw_func(ax)
#
#     # Axes styling
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.spines["bottom"].set_visible(True)
#     ax.spines["bottom"].set_color(AXIS_LINE_COLOR)
#     ax.spines["bottom"].set_linewidth(1)
#     ax.spines["left"].set_visible(True)
#     ax.spines["left"].set_color(AXIS_LINE_COLOR)
#     ax.spines["left"].set_linewidth(1)
#
#     if grid:
#         ax.grid(True, linestyle="--", linewidth=0.5, color=GRID_COLOR)
#
#     ax.tick_params(axis="both", labelsize=14, colors="gray")
#     ax.set_title(title, fontsize=22, pad=15)
#     ax.set_xlabel(xlabel, fontsize=18, labelpad=10)
#     ax.set_ylabel(ylabel, fontsize=18, labelpad=10)
#
#     # Determine whether to show legend
#     if show_legend is None:
#         show_legend = bool(save_path)
#
#     if show_legend:
#         handles, labels = ax.get_legend_handles_labels()
#         if handles:
#             ax.legend(handles, labels, loc="lower right", fontsize=14, frameon=True)
#
#     if note:
#         fig.text(0.5, 0.01, note, va="bottom", ha="center", fontsize=14, color="gray")
#
#     plt.tight_layout()
#
#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         fig.savefig(save_path, bbox_inches="tight")
#         print(f"Figure saved to: {save_path}")
#
#     plt.show()
#     plt.close(fig)


def plot_base(ax, samples, y, quadrilateral, quadrilateral_label="Sampled Quadrilateral", xlim=None, ylim=None):
    """
    Plot scatter points and a background polygon.
    """
    ax.scatter(samples[y == 0, 0], samples[y == 0, 1],
               c=PRIMARY_LIGHT, s=20, edgecolors="none", zorder=3,
               label="Not of Interest")
    ax.scatter(samples[y == 1, 0], samples[y == 1, 1],
               c=PRIMARY_DARK, s=20, edgecolors="none", zorder=3,
               label="Of Interest")
    quadri = Polygon(quadrilateral, closed=True, fill=True,
                     facecolor=QUADRILATERAL_COLOR, alpha=0.6,
                     edgecolor="none", zorder=0, label=quadrilateral_label)
    ax.add_patch(quadri)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
