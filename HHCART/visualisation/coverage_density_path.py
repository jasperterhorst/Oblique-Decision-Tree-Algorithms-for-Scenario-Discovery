"""
Coverage–Density Trade-off Path (coverage_density_path.py)
-----------------------------------------------------------
Plots a line connecting each trained tree depth in coverage–density space.
Each dot is coloured using a gradient from your custom HHCART palette.

- X-axis: Coverage
- Y-axis: Density
- Line: Connects depths 0 → 1 → 2 …
- Colour: Encodes tree depth using PRIMARY_MIDDLE gradient
"""

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

from HHCART.visualisation.base.plot_settings import apply_global_plot_settings, beautify_plot
from HHCART.visualisation.base.save_figure import save_figure
from HHCART.visualisation.base.colors import generate_color_gradient, PRIMARY_MIDDLE


def plot_coverage_density_tradeoff_path(hh, save: bool = False, filename: str = None, title: str = None):
    """
    Plot a sequential path in coverage–density space, colouring points by tree depth.

    Args:
        hh (HHCartD): A trained model with `.metrics_df` containing 'depth', 'coverage', and 'density'.
        save (bool): Whether to save the figure as a PDF using the model's save_dir.
        filename (str, optional): Filename to use when saving. Defaults to 'coverage_density_path.pdf'.
        title (str, optional): Custom plot title. Defaults to 'Coverage–Density Trade-off Path'.
    """
    apply_global_plot_settings()

    # === Validate inputs ===
    if not hasattr(hh, "metrics_df"):
        raise ValueError("Missing `metrics_df`. Ensure metrics were computed after training.")

    df = hh.metrics_df
    required_cols = {"depth", "coverage", "density"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"`metrics_df` must include columns: {required_cols}")

    depths = df["depth"].to_numpy()
    coverage = df["coverage"].to_numpy()
    density = df["density"].to_numpy()

    min_depth = int(depths.min())
    max_depth = int(depths.max())
    n_levels = max_depth - min_depth + 1

    # === Create discrete color map ===
    gradient_colors = generate_color_gradient(PRIMARY_MIDDLE, n_levels)
    cmap = ListedColormap(gradient_colors)
    boundaries = np.arange(min_depth - 0.5, max_depth + 1.5)  # Bin edges
    norm = BoundaryNorm(boundaries, ncolors=n_levels)

    fig, ax = plt.subplots(figsize=(5, 5))

    # === Plot dashed line connecting all points
    ax.plot(coverage, density, linestyle="--", linewidth=1, color="gray", alpha=0.7, zorder=1)

    # === Plot coloured scatter points
    for cov, den, d in zip(coverage, density, depths):
        ax.scatter(
            cov, den,
            s=60,
            c=[cmap(norm(d))],
            edgecolors="black",
            zorder=2
        )

    # === Axis and title styling ===
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    beautify_plot(
        ax,
        title=title or "Coverage – Density Trade-off",
        xlabel="Coverage",
        ylabel="Density"
    )

    # === Add discrete colorbar for depth
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # Required for colorbar to work
    cbar = plt.colorbar(sm, ax=ax, ticks=range(min_depth, max_depth + 1), pad=0.03, shrink=0.85)
    cbar.set_label("Tree Depth", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    fig.tight_layout()

    if save:
        filename = filename or "coverage_density_path.pdf"
        save_figure(hh, filename=filename, save=True)

    plt.show()
    plt.close(fig)


# """
# Coverage–Density Trade-off Path (coverage_density_path.py)
# -----------------------------------------------------------
# Plots a line connecting each trained tree depth in coverage–density space.
# Each dot is coloured using a gradient from your custom HHCART palette.
#
# - X-axis: Coverage
# - Y-axis: Density
# - Line: Connects depths 0 → 1 → 2 …
# - Colour: Encodes tree depth using your PRIMARY_MIDDLE palette
# """
#
# import matplotlib.pyplot as plt
# from matplotlib.cm import ScalarMappable
# from matplotlib.colors import Normalize
#
# from HHCART.visualisation.base.plot_settings import apply_global_plot_settings, beautify_plot
# from HHCART.visualisation.base.save_figure import save_figure
# from HHCART.visualisation.base.colors import generate_color_gradient, PRIMARY_MIDDLE
#
#
# def plot_coverage_density_tradeoff_path(hh, save: bool = False, filename: str = None, title: str = None):
#     """
#     Plot a sequential path in coverage–density space, colouring points by tree depth.
#     Uses a colorbar instead of individual legend entries.
#
#     Args:
#         hh (HHCartD): A trained model with `.metrics_df` containing 'depth', 'coverage', and 'density'.
#         save (bool): Whether to save the figure as a PDF using the model's save_dir.
#         filename (str, optional): Filename to use when saving. Defaults to 'coverage_density_path.pdf'.
#         title (str, optional): Custom plot title. Defaults to 'Coverage–Density Trade-off Path'.
#     """
#     apply_global_plot_settings()
#
#     if not hasattr(hh, "metrics_df"):
#         raise ValueError("Missing `metrics_df`. Ensure metrics were computed after training.")
#
#     df = hh.metrics_df
#     required_cols = {"depth", "coverage", "density"}
#     if not required_cols.issubset(df.columns):
#         raise ValueError(f"`metrics_df` must include columns: {required_cols}")
#
#     depths = df["depth"].to_numpy()
#     coverage = df["coverage"].to_numpy()
#     density = df["density"].to_numpy()
#
#     # Create color gradient and colormap
#     colors = generate_color_gradient(PRIMARY_MIDDLE, len(depths))
#     norm = Normalize(vmin=min(depths), vmax=max(depths))
#     cmap = ScalarMappable(norm=norm, cmap='Greens')
#     cmap.set_array(depths)
#
#     fig, ax = plt.subplots(figsize=(5, 5))
#
#     # Connect all points
#     ax.plot(coverage, density, linestyle="--", linewidth=1, color="gray", alpha=0.7, zorder=1)
#
#     # Scatter points
#     for cov, den, d, color in zip(coverage, density, depths, colors):
#         ax.scatter(cov, den, s=60, c=[color], edgecolors="black", zorder=2)
#
#     # Axis and title styling
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     beautify_plot(
#         ax,
#         title=title or "Coverage-Density Trade-off Path",
#         xlabel="Coverage",
#         ylabel="Density"
#     )
#
#     # Add colorbar
#     cbar = plt.colorbar(cmap, ax=ax, orientation="vertical", shrink=0.85, pad=0.03)
#     cbar.set_label("Tree Depth", fontsize=14)
#     cbar.ax.tick_params(labelsize=12)
#
#     fig.tight_layout()
#
#     if save:
#         filename = filename or "coverage_density_path.pdf"
#         save_figure(hh, filename=filename, save=True)
#
#     plt.show()
#     plt.close(fig)
