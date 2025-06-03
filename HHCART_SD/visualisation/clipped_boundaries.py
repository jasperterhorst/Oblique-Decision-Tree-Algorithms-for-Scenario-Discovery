"""
Clipped Split Visualisation (clipped_boundaries.py)
----------------------------------------------------
Plot recursively clipped decision boundaries for all depths of an HHCartD model.

Supports:
- Polygon clipping of decision boundaries using valid region constraints
- 2D scatter visualisation with labelled color overlays
- Optional saving using the save_figure utility
"""

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box, LineString
from shapely.ops import split as shapely_split
from typing import Optional
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable

from .base.save_figure import save_figure
from .base.colors import PRIMARY_LIGHT, SECONDARY_LIGHT, truncate_colormap
from .base.plot_settings import beautify_plot, beautify_subplot, apply_global_plot_settings
from ..tree import DecisionNode


# === Polygon Utilities ===

def create_initial_polygon(x_min, x_max, y_min, y_max):
    """Return a rectangular region covering the 2D input domain."""
    return box(x_min, y_min, x_max, y_max)


def cut_polygon_with_line(polygon, w, b, side):
    """
    Split a polygon using a line defined by w^T x + b = 0 and return the specified half.

    Args:
        polygon (Polygon): Region to split.
        w (np.ndarray): Weight vector defining the normal of the decision boundary.
        b (float): Bias term.
        side (str): Which side to keep: '<' (negative) or '>=' (positive).

    Returns:
        Polygon or None: The retained region, or None if invalid.
    """
    x_min, y_min, x_max, y_max = polygon.bounds
    try:
        if not np.isclose(w[1], 0.0):
            x_vals = np.array([x_min - 1, x_max + 1])
            y_vals = -(w[0] * x_vals + b) / w[1]
            line = LineString(zip(x_vals, y_vals))
        else:
            if np.isclose(w[0], 0.0):
                return None
            x_split = -b / w[0]
            y_vals = np.array([y_min - 1, y_max + 1])
            line = LineString([(x_split, y_vals[0]), (x_split, y_vals[1])])

        pieces = shapely_split(polygon, line)
        if len(pieces.geoms) != 2:
            return None

        def signed_distance(geom):
            return np.dot(geom.centroid.coords[0], w) + b

        left_piece, right_piece = sorted(pieces.geoms, key=signed_distance)
        return left_piece if side == '<' else right_piece

    except (ValueError, AttributeError, IndexError):
        return None


def construct_region_from_constraints(constraints, initial_region):
    """
    Apply a sequence of linear constraints (split conditions) to an initial region.

    Args:
        constraints (list): List of (w, b, side) defining split inequalities.
        initial_region (Polygon): Starting region to refine.

    Returns:
        Polygon or None: Final clipped region, or None if invalidated.
    """
    region = initial_region
    for w, b, side in constraints:
        region = cut_polygon_with_line(region, w, b, side)
        if region is None or region.is_empty:
            return None
    return region


# === Main Plotting Function ===
def plot_splits_2d_grid(hh, save: bool = True, filename: Optional[str] = None, title: Optional[str] = None):
    """
    Plot decision boundaries clipped to valid polygon regions for all depths of a trained HHCartD model.

    Args:
        hh (HHCartD): A trained HHCartD object with built trees and metrics.
        save (bool): Whether to save the resulting figure (default: False).
        filename (str, optional): Custom filename for saving. Defaults to 'clipped_oblique_splits.pdf'.
        title (str, optional): Custom title for plotting. Defaults to 'clipped_oblique_splits'.

    Example:
        hh.plot_splits_2d_grid()
    """
    apply_global_plot_settings()

    if hh.X.shape[1] != 2:
        raise ValueError("Clipped oblique splits can only be visualised for 2D input.")

    X = hh.X
    y = hh.y
    all_depths = hh.available_depths()
    max_depth = max(all_depths)

    x_min, x_max = X.iloc[:, 0].min(), X.iloc[:, 0].max()
    y_min, y_max = X.iloc[:, 1].min(), X.iloc[:, 1].max()
    bounds = (x_min, x_max, y_min, y_max)

    scatter_colors = np.where(y == 0, SECONDARY_LIGHT, PRIMARY_LIGHT)

    n_cols = 3
    n_rows = int(np.ceil((max_depth + 1) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))

    title = title or "Clipped Oblique Decision Boundaries by Tree Depth"
    fig.suptitle(title, fontsize=24, y=1.02)

    axes = axes.flatten()

    def plot_split_recursive(current_node, constraints, ax):
        if not isinstance(current_node, DecisionNode):
            return

        region = construct_region_from_constraints(constraints, create_initial_polygon(*bounds))
        if region is None or not region.is_valid or region.area < 1e-8:
            return

        w, b = current_node.weights, current_node.bias
        try:
            if not np.isclose(w[1], 0.0):
                x_vals = np.linspace(x_min - 2, x_max + 2, 2)
                y_vals = -(w[0] * x_vals + b) / w[1]
                line_seg = LineString(zip(x_vals, y_vals))
            else:
                if np.isclose(w[0], 0.0):
                    return
                x_split = -b / w[0]
                y_vals = np.linspace(y_min - 2, y_max + 2, 2)
                line_seg = LineString([(x_split, y_vals[0]), (x_split, y_vals[1])])

            clipped = region.intersection(line_seg)
            if not clipped.is_empty and hasattr(clipped, "xy"):
                x_vals, y_vals = clipped.xy
                ax.plot(x_vals, y_vals, 'k--', linewidth=1)

        except (ValueError, AttributeError, TypeError) as e:
            print(f"[⚠️] Failed to plot split line: {e}")
            return

        if len(current_node.children) > 0:
            plot_split_recursive(current_node.children[0], constraints + [(w, b, '<')], ax)
            if len(current_node.children) > 1:
                plot_split_recursive(current_node.children[1], constraints + [(w, b, '>=')], ax)

    for i, depth in enumerate(all_depths):
        tree = hh.get_tree_by_depth(depth)
        ax = axes[i]

        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=scatter_colors, s=5, alpha=0.7)
        plot_split_recursive(tree.root, [], ax)

        beautify_subplot(ax, title=f"Depth {depth}", xlabel=X.columns[0], ylabel=X.columns[1])

        # Annotate coverage and density in top-right corner
        coverage = hh.coverage_by_depth.get(depth, None)
        density = hh.density_by_depth.get(depth, None)

        if coverage is not None and density is not None:
            ax.text(
                0.95, 0.05,
                f"Coverage: {coverage:.2f}\nDensity: {density:.2f}",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.2')
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    for j in range(len(all_depths), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save:
        filename = filename or "clipped_oblique_splits.pdf"
        save_figure(hh, filename=filename, save=True)

    plt.show()
    plt.close(fig)


def plot_splits_2d_overlay(
    hh,
    cmap: str = "YlGnBu",
    save: bool = True,
    filename: Optional[str] = None,
    title: Optional[str] = None
):
    """
    Plot all clipped split lines across all tree depths in a single 2D plot.
    Splits are visualised using a colourmap encoding depth via colour intensity.

    Args:
        hh (HHCartD): A trained HHCartD object with built trees and metrics.
        cmap (str): Matplotlib colormap name (e.g., 'Blues', 'Purples', 'YlGnBu').
        save (bool): Whether to save the figure (default: True).
        filename (str, optional): Custom filename (PDF).
        title (str, optional): Custom plot title.
    """
    apply_global_plot_settings()

    if hh.X.shape[1] != 2:
        raise ValueError("This visualisation only supports 2D input.")

    if cmap not in {"Blues", "Purples", "YlGnBu", "viridis", "cividis"}:
        raise ValueError(f"Unsupported cmap '{cmap}'.")

    X, y = hh.X, hh.y
    scatter_colors = np.where(y == 0, SECONDARY_LIGHT, PRIMARY_LIGHT)
    x_min, x_max = X.iloc[:, 0].min(), X.iloc[:, 0].max()
    y_min, y_max = X.iloc[:, 1].min(), X.iloc[:, 1].max()
    bounds = (x_min, x_max, y_min, y_max)

    max_depth = max(hh.available_depths())

    cmap_continuous = truncate_colormap(cmap, reverse=True, minval=0.0, maxval=0.8, n=512)
    sample_points = np.linspace(0, 1, max_depth + 1)
    cmap_colors = cmap_continuous(sample_points)

    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=scatter_colors, s=5)

    def recurse(node, constraints, depth):
        if not isinstance(node, DecisionNode):
            return

        region = construct_region_from_constraints(constraints, create_initial_polygon(*bounds))
        if region is None or not region.is_valid:
            return

        w, b = node.weights, node.bias
        try:
            if not np.isclose(w[1], 0.0):
                x_vals = np.linspace(x_min - 1, x_max + 1, 2)
                y_vals = -(w[0] * x_vals + b) / w[1]
                line = LineString(zip(x_vals, y_vals))
            else:
                x_split = -b / w[0]
                y_vals = np.linspace(y_min - 1, y_max + 1, 2)
                line = LineString([(x_split, y_vals[0]), (x_split, y_vals[1])])

            clipped = region.intersection(line)
            if not clipped.is_empty and hasattr(clipped, "xy"):
                x_vals, y_vals = clipped.xy
                line_color = cmap_colors[depth]
                ax.plot(x_vals, y_vals, color=line_color, linewidth=1.8, linestyle="-")

        except Exception as e:
            print(f"[⚠️] Split plot failed at depth {depth}: {e}")

        for i, child in enumerate(node.children):
            direction = '<' if i == 0 else '>='
            recurse(child, constraints + [(w, b, direction)], depth + 1)

    final_tree = hh.get_tree_by_depth(max_depth)
    recurse(final_tree.root, [], 0)

    beautify_plot(
        ax,
        title=title or "Split Evolution",
        xlabel=X.columns[0],
        ylabel=X.columns[1]
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    cmap_listed = ListedColormap(cmap_colors)
    bounds = np.arange(-0.5, max_depth + 1.5, 1)
    norm = BoundaryNorm(boundaries=bounds, ncolors=cmap_listed.N)

    sm = ScalarMappable(cmap=cmap_listed, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, boundaries=bounds, ticks=np.arange(max_depth + 1), spacing="proportional",
                        shrink=0.8)
    cbar.set_label("Split Depth", fontsize=12)

    if save:
        filename = filename or "splits_2d_overlay.pdf"
        save_figure(hh, filename=filename, save=True)

    plt.show()
    plt.close(fig)
