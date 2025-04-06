import os
import matplotlib.pyplot as plt
from src.config.colors import AXIS_LINE_COLOR


def beautify_plot(ax, title=None, xlabel=None, ylabel=None, save_path=None):
    """
    Apply standardized beautification to a Matplotlib plot.

    - Respects custom legend passed externally.
    - Replaces underscores in legend labels.
    - Adds vertical spacing between legend entries.
    """
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

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

    # === Legend formatting ===
    legend = ax.get_legend()
    if legend:
        # Replace underscores with spaces in labels
        for text in legend.get_texts():
            text.set_text(text.get_text().replace("_", " "))
            text.set_fontsize(15)

        # Optional: add spacing between entries
        legend._legend_box.sep = 8  # vertical spacing between entries
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("gray")

    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.set_facecolor("white")

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


def beautify_subplot(ax, title=None, xlabel=None, ylabel=None):
    """
    Apply beautification to individual subplots.

    Parameters:
        ax (matplotlib.axes.Axes): The subplot axis to beautify.
        title (str): Title of the subplot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    if title:
        ax.set_title(title, fontsize=12)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)

    ax.xaxis.set_tick_params(labelsize=9)
    ax.yaxis.set_tick_params(labelsize=9)
    ax.set_facecolor("white")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(AXIS_LINE_COLOR)
    ax.spines['bottom'].set_color(AXIS_LINE_COLOR)

    ax.grid(True, linestyle='dashed', color='grey', linewidth=0.5)
    ax.tick_params(axis='both', colors=AXIS_LINE_COLOR)

