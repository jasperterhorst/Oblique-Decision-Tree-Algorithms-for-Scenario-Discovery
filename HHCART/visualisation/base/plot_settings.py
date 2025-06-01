"""
HHCART Plot Style Settings
---------------------------
Shared matplotlib styling utilities for formatting all HHCART visualisations.
Ensures consistent fonts, axis styles, gridlines, and export settings.
"""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


# === Global Matplotlib Defaults ===
def apply_global_plot_settings() -> None:
    """
    Set global matplotlib parameters for consistent appearance across all plots.
    """
    plt.rcParams.update({
        'text.usetex': False,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.labelsize': 20,
        'axes.titlesize': 21,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 15,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.color': 'grey',
        'grid.linestyle': 'dashed',
        'grid.linewidth': 0.5,
    })


# === Beautification Utilities ===
def beautify_plot(ax: Axes, title: str = None, xlabel: str = None,
                  ylabel: str = None) -> None:
    """
    Apply consistent formatting and optionally save a single-axes plot.

    Args:
        ax (Axes): Axis to format.
        title (str, optional): Plot title.
        xlabel (str, optional): X-axis label.
        ylabel (str, optional): Y-axis label.
    """
    fig = ax.get_figure()

    if title:
        ax.set_title(title, fontsize=24, pad=20, wrap=True)
    if xlabel:
        ax.set_xlabel(xlabel or 'X-axis', fontsize=20, labelpad=10)
    if ylabel:
        ax.set_ylabel(ylabel or 'Y-axis', fontsize=20, labelpad=10)

    _style_axes(ax)

    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_text(text.get_text().replace("_", " "))
            text.set_fontsize(15)
        legend._legend_box.sep = 8
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("gray")


def beautify_subplot(ax: Axes, title: str = None, xlabel: str = None,
                     ylabel: str = None, xlim: tuple = None, ylim: tuple = None) -> None:
    """
    Format individual subplots within a grid.

    Args:
        ax (Axes): Subplot axis.
        title (str, optional): Subplot title.
        xlabel (str, optional): X-axis label.
        ylabel (str, optional): Y-axis label.
        xlim (tuple, optional): X-axis limits.
        ylim (tuple, optional): Y-axis limits.
    """
    if title:
        ax.set_title(title, fontsize=18)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=15)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=15)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    _style_axes(ax)


def _style_axes(ax: Axes) -> None:
    """
    Format spines, ticks, grid, and background for a clean layout.

    Args:
        ax (Axes): Axis to style.
    """
    ax.set_facecolor("white")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color("gray")
    ax.spines['bottom'].set_color("gray")

    ax.tick_params(axis='both', colors="gray", labelsize=13)
    ax.grid(True, linestyle='dashed', color='grey', linewidth=0.5)
