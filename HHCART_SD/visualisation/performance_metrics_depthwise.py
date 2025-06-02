"""
Performance Metrics vs. Tree Depth Visualization (performance_metrics_depthwise.py)
--------------------------------------------------
Generates a line plot showing how accuracy, coverage,
density, and F-score change across tree depths.

Designed for integration with the HHCartD object.
"""

import matplotlib.pyplot as plt

from .base.plot_settings import apply_global_plot_settings, beautify_plot
from .base.save_figure import save_figure


def plot_metrics_over_depth(hh, save: bool = False, filename: str = None, title: str = None, y_metric: str = "accuracy"):
    """
    Plot a selected performance metric over tree depth.

    Args:
        hh (HHCartD): Trained HHCART wrapper with .metrics_df populated.
        save (bool): Whether to save the figure.
        filename (str): PDF output filename.
        title (str, optional): Plot title.
        y_metric (str): Which metric to plot. Can be 'accuracy', 'coverage', 'density', or 'subspaces_label1'.
    """
    if hh.metrics_df is None or hh.metrics_df.empty:
        raise ValueError("[❌] No metrics data found. Did you run .build_tree()?")

    df = hh.metrics_df.copy().sort_values("depth")

    if y_metric not in df.columns:
        raise ValueError(f"[❌] Metric '{y_metric}' not found in metrics_df. Available: {list(df.columns)}")

    apply_global_plot_settings()
    fig, ax = plt.subplots(figsize=(5.5, 4))

    ax.plot(df["depth"], df[y_metric], marker="o", label=y_metric.replace("_", " ").capitalize())

    beautify_plot(
        ax,
        title=title or f"{y_metric.replace('_', ' ').capitalize()} vs. Tree Depth",
        xlabel="Tree Depth",
        ylabel=y_metric.replace("_", " ").capitalize()
    )

    ax.set_xlim(0, df["depth"].max())
    if y_metric != "subspaces_label1":
        ax.set_ylim(0, 1)

    ax.legend()
    plt.tight_layout()
    save_figure(hh, filename or f"{y_metric}_vs_depth.pdf", save)

    return fig, ax


# """
# Performance Metrics vs. Tree Depth Visualization (performance_metrics_depthwise.py)
# --------------------------------------------------
# Generates a line plot showing how accuracy, coverage,
# density, and F-score change across tree depths.
#
# Designed for integration with the HHCartD object.
# """
#
# import matplotlib.pyplot as plt
#
# from .base.plot_settings import apply_global_plot_settings, beautify_plot
# from .base.save_figure import save_figure
#
#
# def plot_metrics_over_depth(hh, save: bool = False, filename: str = None, title: str = None):
#     """
#     Plot key performance metrics over tree depth.
#
#     Args:
#         hh (HHCartD): Trained HHCART_SD-D wrapper with .metrics_df populated.
#         save (bool): Whether to save the figure.
#         filename (str): PDF output filename.
#         title (str, optional): Title to display on the plot.
#
#     Returns:
#         (fig, ax): Tuple of Matplotlib figure and axis.
#     """
#     if hh.metrics_df is None or hh.metrics_df.empty:
#         raise ValueError("[❌] No metrics data found. Did you run .build_tree()?")
#
#     if filename is None:
#         filename = "performance_metrics_depthwise.pdf"
#
#     df = hh.metrics_df.copy().sort_values("depth")
#
#     apply_global_plot_settings()
#     fig, ax = plt.subplots(figsize=(5.5, 4))
#
#     # Plot all core metrics
#     metrics = ["accuracy", "coverage", "density"]
#     for metric in metrics:
#         if metric in df.columns:
#             ax.plot(df["depth"], df[metric], marker="o", label=metric.capitalize())
#
#     final_title = title or "Performance Metrics vs. Tree Depth"
#     beautify_plot(ax,
#                   title=final_title,
#                   xlabel="Tree Depth",
#                   ylabel="Score")
#
#     ax.set_xlim(0, df["depth"].max())
#     ax.set_ylim(0, 1)
#
#     ax.legend()
#     plt.tight_layout()
#     save_figure(hh, filename, save)
#
#     return fig, ax
