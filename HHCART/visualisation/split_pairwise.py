"""
Visualisation: Pairwise Clipped Oblique Splits
---------------------------------------------
Given a selected decision tree, this module plots clipped oblique split lines
on all 2D feature combinations (i.e., a feature scatter matrix).

Each split is rendered in the appropriate 2D subplot by projecting the constraints
and recursively applying the clipping logic.

To be called via: hh.plot_pairwise_splits(k=..., depth=...)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from shapely.geometry import box, LineString
from shapely.ops import split as shapely_split

from HHCART.tree import DecisionNode
from src.config.colors_and_plot_styles import PRIMARY_LIGHT, SECONDARY_LIGHT
from src.config.plot_settings import beautify_subplot


def create_initial_polygon(x_min, x_max, y_min, y_max):
    return box(x_min, y_min, x_max, y_max)


def cut_polygon_with_line(polygon, w, b, side):
    x_min, y_min, x_max, y_max = polygon.bounds
    try:
        if not np.isclose(w[1], 0.0):
            x_vals = np.array([x_min - 1, x_max + 1])
            y_vals = -(w[0] * x_vals + b) / w[1]
            line = LineString([(x_vals[0], y_vals[0]), (x_vals[1], y_vals[1])])
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

        left, right = sorted(pieces.geoms, key=signed_distance)
        return left if side == '<' else right

    except Exception:
        return None


def construct_region_from_constraints(constraints, initial_region):
    region = initial_region
    for w, b, side in constraints:
        region = cut_polygon_with_line(region, w, b, side)
        if region is None or region.is_empty:
            return None
    return region


def plot_pairwise_splits(X, y, tree, features, save_path=None):
    n_feats = len(features)
    feat_indices = {f: i for i, f in enumerate(features)}
    fig, axes = plt.subplots(n_feats, n_feats, figsize=(3.5 * n_feats, 3.5 * n_feats))

    for i, f1 in enumerate(features):
        for j, f2 in enumerate(features):
            ax = axes[i, j]

            if i == j:
                ax.hist(X[f1][y == 0], bins=20, color=SECONDARY_LIGHT, alpha=0.5)
                ax.hist(X[f1][y == 1], bins=20, color=PRIMARY_LIGHT, alpha=0.5)
            elif i > j:
                color = np.where(y == 0, SECONDARY_LIGHT, PRIMARY_LIGHT)
                ax.scatter(X[f2], X[f1], c=color, s=5, alpha=0.6)
                x_min, x_max = X[f2].min(), X[f2].max()
                y_min, y_max = X[f1].min(), X[f1].max()
                bounds = (x_min, x_max, y_min, y_max)

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)

                def draw_clipped_split(node, constraints):
                    if not isinstance(node, DecisionNode):
                        return
                    w, b = node.weights, node.bias
                    if w is None or len(w) != len(features):
                        return

                    region = construct_region_from_constraints(constraints, create_initial_polygon(*bounds))
                    if region is None:
                        return

                    wi, wj = w[feat_indices[f2]], w[feat_indices[f1]]
                    if np.allclose([wi, wj], 0.0):
                        return

                    try:
                        if not np.isclose(wj, 0.0):
                            x_vals = np.linspace(bounds[0], bounds[1], 2)
                            y_vals = -(wi * x_vals + b) / wj
                            line = LineString(zip(x_vals, y_vals))
                        else:
                            x_split = -b / wi
                            y_vals = np.linspace(bounds[2], bounds[3], 2)
                            line = LineString([(x_split, y_vals[0]), (x_split, y_vals[1])])

                        clipped = region.intersection(line)
                        if not clipped.is_empty and hasattr(clipped, "xy"):
                            x_clip, y_clip = clipped.xy
                            ax.plot(x_clip, y_clip, 'k--', linewidth=1)
                    except Exception:
                        pass

                    # Recurse
                    draw_clipped_split(node.children[0], constraints + [(w, b, '<')])
                    if len(node.children) > 1:
                        draw_clipped_split(node.children[1], constraints + [(w, b, '>=')])

                draw_clipped_split(tree.root, [])

            if i == n_feats - 1:
                ax.set_xlabel(f2)
            if j == 0:
                ax.set_ylabel(f1)
            if i != j:
                beautify_subplot(ax)
            else:
                ax.set_title(f1)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[\u2713] Saved pairwise split figure: {save_path}")
    else:
        plt.show()
    plt.close(fig)
