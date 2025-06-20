"""
Performance Metrics vs. Model Structure (performance_metrics_depthwise.py)
--------------------------------------------------------------------------
Plots key performance metrics against a structural or interpretability axis,
such as tree depth or the number of subspaces labelled as class 1.

This visualisation enables comparative evaluation of trade-offs between performance
and structural complexity.

Designed for integration with the HHCartD object.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from .base.plot_settings import apply_global_plot_settings, beautify_plot
from .base.save_figure import save_figure


def plot_metrics_vs_structure(
    hh,
    save: bool = False,
    filename: str = None,
    title: str = None,
    x_axis: str = "depth",
    y_axis: list = ["coverage", "density"],
):
    """
    Plot accuracy, coverage, and density against a structural complexity axis.

    Args:
        hh (HHCartD): Trained HHCART_SD object with `.metrics_df` populated.
        save (bool): Whether to save the figure.
        filename (str, optional): PDF output filename.
        title (str, optional): Custom plot title.
        x_axis (str): X-axis variable. Options:
                      - 'depth' (default): Tree depth
                      - 'class1_leaf_count': Number of subspaces labelled as class 1
        y_axis (list): List of metrics to plot against the x-axis. Default is ["coverage", "density"].

    Raises:
        ValueError: If the x_axis or required metric columns are missing.

    Returns:
        (matplotlib.figure.Figure, matplotlib.axes.Axes): Generated figure and axis.
    """
    # Check if hh contains a metrics_df that is not None or empty
    if hh.metrics_df is None or hh.metrics_df.empty:
        raise ValueError("[ERROR] No metrics data found. Did you run `.build_tree()`?")

    # Copy and sort the metrics DataFrame by the x_axis
    df = hh.metrics_df.copy().sort_values(x_axis)

    # Validate x_axis and y_axis columns are in the DataFrame
    required_cols = set(y_axis) | {x_axis}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"[ERROR] metrics_df must include: {required_cols}")

    # Set standard plot settings
    apply_global_plot_settings()
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Plot each y_axis metric against the x_axis
    for metric in y_axis:
        ax.plot(
            df[x_axis],
            df[metric],
            marker="o",
            label=metric.capitalize()
        )

    # Ensure integer-only ticks and start from 0
    ax.set_xlim(left=0, right=df[x_axis].max())
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Set axis labels
    axis_label = "Class 1 Leaf Count" if x_axis == "class1_leaf_count" else "Tree Depth"

    # Beautify the plot with title and labels
    beautify_plot(
        ax,
        title=title or f"Performance Metrics vs. {axis_label}",
        xlabel=axis_label,
        ylabel="Score"
    )

    # Set y-axis limits and legend
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()

    # Save the figure if requested
    save_figure(hh, filename or f"metrics_vs_{x_axis}.pdf", save)

    return fig, ax
