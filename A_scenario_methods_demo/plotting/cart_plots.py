"""
CART plotting for the scenario methods demo.
"""

import numpy as np
from matplotlib.patches import Rectangle, Polygon, Patch
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from A_scenario_methods_demo.plotting.base_plots import generic_plot
from src.config.colors import (
    PRIMARY_LIGHT, PRIMARY_DARK,
    CART_OUTLINE_COLOR, QUADRILATERAL_COLOR
)


def plot_cart_spatial_evolution(samples, y, quadrilateral, cart_boxes, cart_classifications,
                                title="CART: Spatial Evolution", note="", save_path=None,
                                axis_limits=(0, 1, 0, 1), alpha=0.2, tol=1e-6):
    def draw(ax):
        # Draw background quadrilateral
        quadri_patch = Polygon(quadrilateral, closed=True, color=QUADRILATERAL_COLOR, zorder=0)
        ax.add_patch(quadri_patch)

        # Draw CART boxes with primary color fills
        for i, box in enumerate(cart_boxes):
            fill_color = PRIMARY_DARK if cart_classifications[i] == 1 else PRIMARY_LIGHT
            fill_rgba = to_rgba(fill_color, alpha=alpha)
            rect = Rectangle((box["x"][0], box["y"][0]),
                             box["x"][1] - box["x"][0],
                             box["y"][1] - box["y"][0],
                             facecolor=fill_rgba, edgecolor="none", linewidth=0, zorder=4)
            ax.add_patch(rect)

            drawn_edges = []
            edges = [((box["x"][0], box["y"][0]), (box["x"][1], box["y"][0])),
                     ((box["x"][1], box["y"][0]), (box["x"][1], box["y"][1])),
                     ((box["x"][1], box["y"][1]), (box["x"][0], box["y"][1])),
                     ((box["x"][0], box["y"][1]), (box["x"][0], box["y"][0]))]
            for edge in edges:
                exists = False
                for drawn_edge in drawn_edges:
                    if (np.allclose(edge[0], drawn_edge[0], atol=tol) and
                        np.allclose(edge[1], drawn_edge[1], atol=tol)) or \
                            (np.allclose(edge[0], drawn_edge[1], atol=tol) and
                             np.allclose(edge[1], drawn_edge[0], atol=tol)):
                        exists = True
                        break
                if not exists:
                    ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]],
                            color=CART_OUTLINE_COLOR, linewidth=2, zorder=5)
                    drawn_edges.append(edge)

        # Plot sample points
        ax.scatter(samples[y == 0, 0], samples[y == 0, 1],
                   c=PRIMARY_LIGHT, s=20, edgecolors="none", zorder=3)
        ax.scatter(samples[y == 1, 0], samples[y == 1, 1],
                   c=PRIMARY_DARK, s=20, edgecolors="none", zorder=3)

        # Set axis limits
        xmin, xmax, ymin, ymax = axis_limits
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # Legend
        legend_handles = [
            Line2D([], [], color=PRIMARY_LIGHT, marker='o', linestyle='None', markersize=6, label="Not of Interest"),
            Line2D([], [], color=PRIMARY_DARK, marker='o', linestyle='None', markersize=6, label="Of Interest"),
            Line2D([], [], color=CART_OUTLINE_COLOR, linewidth=2, label="CART Boundaries"),
            Patch(facecolor=QUADRILATERAL_COLOR, edgecolor="none", label="Sampled Quadrilateral")
        ]
        ax.legend(handles=legend_handles, loc="lower right", fontsize=10)

    generic_plot(title, "X-axis", "Y-axis", note, save_path, draw, grid=False)
