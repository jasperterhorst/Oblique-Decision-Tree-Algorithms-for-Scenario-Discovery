"""
Clipped Region Visualisation from Predictions (oblique_regions.py)
-------------------------------------------------------------
Plot decision regions using predictions for each node's area across all depths of an HHCartD model.

Supports:
- Filled decision regions based on tree predictions
- 2D scatter visualisation for actual input data
- Optional saving using the save_figure utility
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.colors import ListedColormap
from typing import Optional

from .base.save_figure import save_figure
from .base.colors import PRIMARY_LIGHT, PRIMARY_DARK
from .base.plot_settings import beautify_subplot, apply_global_plot_settings


def plot_regions_2d_grid(hh, save: bool = False, filename: Optional[str] = None, title: Optional[str] = None):
    """
    Plot 2D decision regions using predictions for each pruned tree depth in an HHCartD object.

    Args:
        hh (HHCartD): Trained HHCartD wrapper with .trees_by_depth populated.
        save (bool): Whether to save the resulting figure.
        filename (str, optional): Custom filename for saving.
        title (str, optional): Optional title.
    """
    apply_global_plot_settings()

    if hh.X.shape[1] != 2:
        raise ValueError("Decision regions can only be visualised for 2D input.")

    X = hh.X
    y = hh.y
    max_depth = max(hh.available_depths())
    custom_cmap = ListedColormap([PRIMARY_LIGHT, PRIMARY_DARK])

    x_min, x_max = X.iloc[:, 0].min(), X.iloc[:, 0].max()
    y_min, y_max = X.iloc[:, 1].min(), X.iloc[:, 1].max()
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    scatter_colors = np.where(y == 0, PRIMARY_LIGHT, PRIMARY_DARK)
    n_cols = 3
    n_plots = max_depth + 1
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten()

    title = title or "Oblique Decision Regions by Tree Depth"
    fig.suptitle(title, fontsize=24, y=1.02)

    ax = axes[0]
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=scatter_colors, s=5, alpha=0.7)
    beautify_subplot(ax, title="Input Data", xlabel=X.columns[0], ylabel=X.columns[1])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    for depth in range(1, max_depth + 1):
        tree = hh.get_tree_by_depth(depth)
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = np.array([tree.predict(point) for point in grid_points]).reshape(xx.shape)

        ax = axes[depth]
        ax.pcolormesh(xx, yy, Z, cmap=custom_cmap, alpha=0.7, shading='auto')
        # ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=scatter_colors, s=5, alpha=0.7)
        beautify_subplot(ax, title=f"Depth {depth}", xlabel=X.columns[0], ylabel=X.columns[1])

        coverage = hh.coverage_by_depth.get(depth, None)
        density = hh.density_by_depth.get(depth, None)

        if coverage is not None and density is not None:
            ax.text(0.95, 0.05,
                    f"Coverage: {coverage:.2f}\nDensity: {density:.2f}",
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.2'))

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    for j in range(n_plots, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save:
        filename = filename or "oblique_regions_by_depth.pdf"
        save_figure(hh, filename=filename, save=True)

    plt.show()
    plt.close(fig)
