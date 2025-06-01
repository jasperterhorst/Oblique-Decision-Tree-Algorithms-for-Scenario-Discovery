"""
plot_settings.py

Shared styling utilities for plot generation.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from src.config.colors_and_plot_styles import AXIS_LINE_COLOR


def apply_global_plot_settings() -> None:
    """
    Set global matplotlib font and layout parameters.

    This function should be called at the beginning of any script
    that generates figures to ensure consistent font and style settings.
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


def beautify_plot(ax: Axes, title: str = None, xlabel: str = None,
                  ylabel: str = None, save_path: str = None) -> None:
    """
    Apply standardized beautification to a Matplotlib plot.

    Parameters:
        ax (Axes): The plot axis to format.
        title (str, optional): Title of the plot.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        save_path (str, optional): Path to save the figure (PDF).
    """
    fig = ax.get_figure()

    if title:
        ax.set_title(title, fontsize=24, pad=20, wrap=True)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=20, labelpad=10)
    else:
        ax.set_xlabel(ax.get_xlabel() or 'X-axis', fontsize=20, labelpad=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=20, labelpad=10)
    else:
        ax.set_ylabel(ax.get_ylabel() or 'Y-axis', fontsize=20, labelpad=10)

    _style_axes(ax)

    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_text(text.get_text().replace("_", " "))
            text.set_fontsize(15)
        legend._legend_box.sep = 8
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("gray")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved: {save_path}")

    plt.show()
    plt.close(fig)


def beautify_subplot(ax: Axes, title: str = None, xlabel: str = None,
                     ylabel: str = None, xlim: tuple = None, ylim: tuple = None) -> None:
    """
    Apply standardized beautification to an individual subplot.

    Parameters:
        ax (Axes): The subplot axis to beautify.
        title (str, optional): Title of the subplot.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        xlim (tuple, optional): (min, max) for x-axis limits.
        ylim (tuple, optional): (min, max) for y-axis limits.
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
    Apply spine, grid, tick, and face color styling to an axis.

    Parameters:
        ax (Axes): The axis to style.
    """
    ax.set_facecolor("white")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(AXIS_LINE_COLOR)
    ax.spines['bottom'].set_color(AXIS_LINE_COLOR)

    ax.tick_params(axis='both', colors=AXIS_LINE_COLOR, labelsize=13)
    ax.grid(True, linestyle='dashed', color='grey', linewidth=0.5)
