"""
CART plotting for the scenario methods demo.
"""

from typing import Optional, List
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle, Polygon, Patch
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from A_scenario_methods_demo.plotting.base_plots import generic_plot
from src.config import (
    SCATTER_COLORS,
    CART_OUTLINE_COLOR,
    QUADRILATERAL_COLOR
)


def plot_cart_spatial_evolution(
    samples: np.ndarray,
    y: np.ndarray,
    quadrilateral: np.ndarray,
    cart_boxes: List[dict],
    cart_classifications: List[int],
    title: str = "CART Boxes on Data",
    note: str = "",
    save_path: Optional[str] = None,
    axis_limits: tuple[float, float, float, float] = (0, 1, 0, 1),
    alpha: float = 0.2,
    tol: float = 1e-6
) -> None:
    """
    Plot the spatial evolution of CART decision boxes and classified samples.

    Parameters:
        samples (np.ndarray): Sample coordinates (n, 2).
        y (np.ndarray): Binary class labels.
        quadrilateral (np.ndarray): Boundary polygon coordinates.
        cart_boxes (list of dict): Each box is a dict with 'x' and 'y' bounds.
        cart_classifications (list of int): Predicted class (0 or 1) for each box.
        title (str): Plot title.
        note (str): Footer note under the figure.
        save_path (str | None): Optional file path to save figure.
        axis_limits (tuple): (xmin, xmax, ymin, ymax) plot bounds.
        alpha (float): Transparency of box fills.
        tol (float): Tolerance for merging edges.
    """

    def draw(ax: Axes) -> None:
        # Draw background quadrilateral
        quadri_patch = Polygon(quadrilateral, closed=True, color=QUADRILATERAL_COLOR, zorder=0)
        ax.add_patch(quadri_patch)

        # Draw CART boxes
        for i, box in enumerate(cart_boxes):
            fill_color = SCATTER_COLORS["interest"] if cart_classifications[i] == 1 else SCATTER_COLORS["no_interest"]
            fill_rgba = to_rgba(fill_color, alpha=alpha)
            rect = Rectangle(
                (box["x"][0], box["y"][0]),
                box["x"][1] - box["x"][0],
                box["y"][1] - box["y"][0],
                facecolor=fill_rgba,
                edgecolor="none",
                linewidth=0,
                zorder=4
            )
            ax.add_patch(rect)

            # Add edges only if not already drawn
            drawn_edges = []
            edges = [
                ((box["x"][0], box["y"][0]), (box["x"][1], box["y"][0])),
                ((box["x"][1], box["y"][0]), (box["x"][1], box["y"][1])),
                ((box["x"][1], box["y"][1]), (box["x"][0], box["y"][1])),
                ((box["x"][0], box["y"][1]), (box["x"][0], box["y"][0]))
            ]
            for edge in edges:
                if not any(
                    (np.allclose(edge[0], e[0], atol=tol) and np.allclose(edge[1], e[1], atol=tol)) or
                    (np.allclose(edge[0], e[1], atol=tol) and np.allclose(edge[1], e[0], atol=tol))
                    for e in drawn_edges
                ):
                    ax.plot(
                        [edge[0][0], edge[1][0]],
                        [edge[0][1], edge[1][1]],
                        color=CART_OUTLINE_COLOR,
                        linewidth=2,
                        zorder=5
                    )
                    drawn_edges.append(edge)

        # Scatter samples
        ax.scatter(samples[y == 0, 0], samples[y == 0, 1],
                   c=SCATTER_COLORS["no_interest"], s=20, edgecolors="none", zorder=3)
        ax.scatter(samples[y == 1, 0], samples[y == 1, 1],
                   c=SCATTER_COLORS["interest"], s=20, edgecolors="none", zorder=3)

        xmin, xmax, ymin, ymax = axis_limits
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # Legend
        legend_handles = [
            Line2D([], [], color=SCATTER_COLORS["no_interest"], marker='o', linestyle='None',
                   markersize=6, label="Not of Interest"),
            Line2D([], [], color=SCATTER_COLORS["interest"], marker='o', linestyle='None',
                   markersize=6, label="Of Interest"),
            Line2D([], [], color=CART_OUTLINE_COLOR, linewidth=2, label="CART Boundaries"),
            Patch(facecolor=QUADRILATERAL_COLOR, edgecolor="none", label="Quadrilateral")
        ]
        ax.legend(handles=legend_handles, loc="lower right")

    generic_plot(title, "X-axis", "Y-axis", note, save_path, draw, grid=False)
