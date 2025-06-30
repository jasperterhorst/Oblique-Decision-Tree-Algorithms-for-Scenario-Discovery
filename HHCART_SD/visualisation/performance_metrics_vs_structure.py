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
import numpy as np

from .base.plot_settings import apply_global_plot_settings, beautify_plot
from .base.save_figure import save_figure


def plot_metrics_vs_structure(
        hh,
        save: bool = False,
        filename: str = None,
        title: str = None,
        x_axis: str = "depth",
        y_axis: list = None,
        figsize: tuple = (5, 4)
):
    """
    Plot accuracy, coverage, and density against a structural complexity axis,
    with annotations showing the corresponding value of the other complexity metric.

    Args:
        hh (HHCartD): Trained HHCART_SD object with `.metrics_df` populated.
        save (bool): Whether to save the figure.
        filename (str, optional): PDF output filename.
        title (str, optional): Custom plot title.
        x_axis (str): X-axis variable. Options:
                      - 'depth' (default): Annotates with class 1 leaf count.
                      - 'class1_leaf_count': Annotates with tree depth.
        y_axis (list, optional): List of metrics to plot against the x-axis.
                                 Defaults to ["coverage", "density"].
        figsize (tuple, optional): The figure size (width, height) in inches. Defaults to (6, 4.5).

    Raises:
        ValueError: If the x_axis or required metric columns are missing.

    Returns:
        (matplotlib.figure.Figure, matplotlib.axes.Axes): Generated figure and axis.
    """
    if y_axis is None:
        y_axis = ["coverage", "density"]

    if hh.metrics_df is None or hh.metrics_df.empty:
        raise ValueError("[ERROR] No metrics data found. Did you run `.build_tree()`?")

    df = hh.metrics_df.copy().sort_values(x_axis)

    required_cols = set(y_axis) | {x_axis}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"[ERROR] metrics_df must include: {required_cols}")

    apply_global_plot_settings()
    fig, ax = plt.subplots(figsize=figsize)

    for metric in y_axis:
        ax.plot(
            df[x_axis],
            df[metric],
            marker="o",
            label=metric.capitalize()
        )

    ax.set_xlim(left=0, right=df[x_axis].max() * 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    axis_label = "Class 1 Leaf Count" if x_axis == "class1_leaf_count" else "Tree Depth"

    beautify_plot(
        ax,
        title=title or f"Performance Metrics vs. {axis_label}",
        xlabel=axis_label,
        ylabel="Score"
    )

    # ANNOTATION LOGIC
    annotation_col = None
    if x_axis == "depth":
        annotation_col = "class1_leaf_count"
    elif x_axis == "class1_leaf_count":
        annotation_col = "depth"

    if annotation_col and annotation_col in df.columns:
        df['annotation_y'] = df[y_axis].max(axis=1)
        x_coords = df[x_axis]
        y_coords = df['annotation_y']
        labels = df[annotation_col]

        x_range = x_coords.max() - x_coords.min()
        merge_tol_x = 0.04 * x_range if x_range > 0 else 0.04
        merge_tol_y = 0.04

        groups = []

        for x, y, label in zip(x_coords, y_coords, labels):
            assigned = False
            for grp in groups:
                xs, ys = zip(*grp["coords"])
                centroid = (np.mean(xs), np.mean(ys))
                dist = np.hypot((x - centroid[0]) / merge_tol_x, (y - centroid[1]) / merge_tol_y)

                if dist <= 1:
                    grp["coords"].append((x, y))
                    grp["labels"].append(label)
                    assigned = True
                    break

            if not assigned:
                groups.append({"coords": [(x, y)], "labels": [label]})

        # Annotate each group
        for grp in groups:
            xs, ys = zip(*grp["coords"])
            centroid = (np.mean(xs), np.mean(ys))
            sorted_labels = sorted(list(set(grp["labels"])))
            text_str = "/".join(map(str, sorted_labels))

            horizontal_offset = 0.02 * x_range if x_range > 0 else 0.02

            ax.text(
                centroid[0] + horizontal_offset,
                centroid[1] + 0.02,
                text_str,
                fontsize=10,  # Font size increased
                ha="left",
                va="bottom",
                color="dimgray"
            )

    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')
    plt.tight_layout()

    save_figure(hh, filename or f"metrics_vs_{x_axis}.pdf", save)

    return fig, ax
