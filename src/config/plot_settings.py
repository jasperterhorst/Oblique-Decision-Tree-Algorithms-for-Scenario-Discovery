import os
import matplotlib.pyplot as plt
from src.config.colors import AXIS_LINE_COLOR


def beautify_plot(ax, title=None, xlabel=None, ylabel=None, save_path=None):
    """
    Apply standardized beautification to a Matplotlib plot.

    Parameters:
    — ax: The matplotlib Axes object.
    — title: str, optional title to display.
    — xlabel: str, optional label for the x-axis.
    — ylabel: str, optional label for the y-axis.
    — save_path: str, optional path to save the figure as PDF.
    """
    # Set global font settings before any text is rendered.
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    fig = ax.get_figure()

    # Set plot title if provided.
    if title:
        ax.set_title(title, fontsize=24, pad=15)

    # Set x-axis label; use provided xlabel or default.
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=20)
    else:
        ax.set_xlabel(ax.get_xlabel() or 'X-axis', fontsize=20)

    # Set y-axis label; use provided ylabel or default.
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=20)
    else:
        ax.set_ylabel(ax.get_ylabel() or 'Y-axis', fontsize=20)

    handles, labels = ax.get_legend_handles_labels()
    new_labels = [label.replace('_', ' ') for label in labels]
    ax.legend(handles, new_labels, loc='upper right', fontsize=15)

    ax.legend(loc='upper right', fontsize=15)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(AXIS_LINE_COLOR)
    ax.spines['bottom'].set_color(AXIS_LINE_COLOR)
    ax.grid(True, linestyle='dashed', color='grey', linewidth=0.5)
    ax.tick_params(axis='both', colors=AXIS_LINE_COLOR)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved: {save_path}")

    plt.show()
    plt.close(fig)
