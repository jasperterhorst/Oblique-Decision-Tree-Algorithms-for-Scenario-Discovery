"""
Coverage–Density Trade-off Path (coverage_density_path.py)
-----------------------------------------------------------
Plots a trajectory in coverage–density space for oblique decision trees across depths.
Each point represents a trained tree at a given depth, and is color-coded according to
one of two interpretability proxies, either:

- the depth of the tree, or
- the number of subspaces predicted as class 1.

This visualisation supports comparative analysis of performance vs. interpretability.

Usage:
-------
    hh.plot_tradeoff_path(color_by="class1_leaf_count")
    hh.plot_tradeoff_path(color_by="depth", save=True)
"""

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
import numpy as np

from .base.plot_settings import apply_global_plot_settings, beautify_plot
from .base.save_figure import save_figure
from .base.colors import generate_color_gradient, PRIMARY_MIDDLE


def plot_coverage_density_tradeoff_path(
    hh,
    save: bool = False,
    filename: str = None,
    title: str = None,
    color_by: str = "depth"
):
    """
    Plot a sequential path in coverage–density space, color-coded by either tree depth
    or the number of subspaces predicted as class 1.

    Args:
        hh (HHCartD): Trained HHCART_SD object with `.metrics_df` containing 'coverage', 'density',
        and chosen color column.
        save (bool): Whether to save the figure as a PDF using the model’s save_dir.
        filename (str, optional): Filename for saving. Defaults to 'coverage_density_path.pdf'.
        title (str, optional): Custom plot title. Defaults depend on color mode.
        color_by (str): Variable used to color the points. Options:
                         - "depth" (default): Discrete color map for each tree depth
                         - "class1_leaf_count": Continuous map for number of subspaces labelled 1

    Raises:
        ValueError: If required columns are missing from the model's metrics_df.

    Returns:
        None. Displays and optionally saves a matplotlib figure.
    """
    apply_global_plot_settings()

    df = hh.metrics_df
    if not {"coverage", "density", color_by}.issubset(df.columns):
        raise ValueError(f"`metrics_df` must include columns: 'coverage', 'density', and '{color_by}'.")

    coverage = df["coverage"].to_numpy()
    density = df["density"].to_numpy()
    color_values = df[color_by].to_numpy()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect("equal", adjustable="box")

    # Trajectory line
    ax.plot(coverage, density, linestyle="--", linewidth=1, color="gray", alpha=0.7, zorder=1)

    if color_by == "depth":
        min_val, max_val = int(color_values.min()), int(color_values.max())
        n_levels = max_val - min_val + 1
        cmap = ListedColormap(generate_color_gradient(PRIMARY_MIDDLE, n_levels))
        norm = BoundaryNorm(np.arange(min_val - 0.5, max_val + 1.5), ncolors=n_levels)
    else:
        min_val, max_val = color_values.min(), color_values.max()
        cmap = get_cmap("viridis_r")
        norm = Normalize(vmin=min_val, vmax=max_val)

    # Scatter points
    for cov, den, val in zip(coverage, density, color_values):
        ax.scatter(
            cov, den,
            s=60,
            c=[cmap(norm(val))],
            edgecolors="black",
            zorder=2
        )

    # Axis styling
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    label_map = {
        "depth": "Tree Depth",
        "class1_leaf_count": "Subspaces Predicted as Label 1"
    }
    default_title = f"Coverage vs. Density ({label_map[color_by]})"

    beautify_plot(
        ax,
        title=title or default_title,
        xlabel="Coverage",
        ylabel="Density"
    )

    # Colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.08, shrink=0.7)
    cbar.set_label(label_map[color_by], fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    fig.tight_layout()

    if save:
        filename = filename or f"coverage_density_path_{color_by}.pdf"
        save_figure(hh, filename=filename, save=True)

    plt.show()
    plt.close(fig)
