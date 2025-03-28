# D_oblique_decision_trees/visualisation.py

import matplotlib.pyplot as plt
import numpy as np
from D_oblique_decision_trees.tree import DecisionNode, LeafNode
from matplotlib import cm
from shapely.geometry import Polygon, LineString


def print_tree_structure(tree):
    tree.traverse(lambda node: print(repr(node)))


def plot_decision_boundaries(tree, X, y, ax=None, show_data=True, xlim=(-1, 1), ylim=(-1, 1)):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    if show_data:
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=20, alpha=0.6)

    x_vals = np.linspace(xlim[0], xlim[1], 500)

    def plot_split(node):
        if isinstance(node, DecisionNode):
            w = node.weights
            b = node.bias

            if len(w) != 2:
                return  # skip non-2D

            if np.isclose(w[1], 0.0):
                x = -b / w[0]
                ax.axvline(x, color='k', linestyle='--', linewidth=1)
            else:
                y_vals = -(w[0] * x_vals + b) / w[1]
                ax.plot(x_vals, y_vals, 'k--', linewidth=1)

            for child in node.children:
                plot_split(child)

    plot_split(tree.root)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title("Oblique Decision Boundaries")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    plt.tight_layout()


# def plot_decision_boundaries(tree, X, y, ax=None, show_data=True, cmap_name='viridis', shade_subspaces=True):
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(6, 6))
#
#     if show_data:  # Plot the data points
#         ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=20, alpha=0.6)
#
#     xlim = (0, 1)
#     ylim = (0, 1)
#     max_depth = tree.max_depth  # Maximum depth of the tree
#     colormap = cm.get_cmap(cmap_name, max_depth + 1)
#
#     def clip_polygon_to_box(polygon, x_bounds, y_bounds):
#         """Clip the polygon to the current subspace box"""
#         box = Polygon([(x_bounds[0], y_bounds[0]),
#                        (x_bounds[1], y_bounds[0]),
#                        (x_bounds[1], y_bounds[1]),
#                        (x_bounds[0], y_bounds[1])])
#         return polygon.intersection(box)
#
#     def plot_split(node, x_bounds, y_bounds, depth=0):
#         if not isinstance(node, DecisionNode):
#             return  # Skip if it's not a decision node
#
#         # Define the color for the current depth
#         color = colormap(depth)
#
#         # Create a polygon for the subspace (rectangle)
#         polygon = Polygon([(x_bounds[0], y_bounds[0]),
#                            (x_bounds[1], y_bounds[0]),
#                            (x_bounds[1], y_bounds[1]),
#                            (x_bounds[0], y_bounds[1])])
#
#         # Clip the polygon to the current bounds (subspace box)
#         clipped_polygon = clip_polygon_to_box(polygon, x_bounds, y_bounds)
#
#         # If the subspace is a leaf node, plot it
#         if node.is_leaf():
#             if shade_subspaces:  # Optionally shade subspaces
#                 x, y = clipped_polygon.exterior.xy
#                 ax.fill(x, y, color=color, alpha=1)
#
#             # Plot the leaf node's prediction (e.g., class label)
#             ax.text(np.mean(x_bounds), np.mean(y_bounds), f"Class: {node.prediction}",
#                     color='black', ha='center', va='center', fontsize=8)
#
#         # Plot the split line within the subspace
#         w = node.weights
#         b = node.bias
#         x_vals = np.linspace(x_bounds[0], x_bounds[1], 1000)
#
#         if np.isclose(w[1], 0.0):  # Vertical split
#             x_split = -b / w[0]
#             if x_bounds[0] <= x_split <= x_bounds[1]:
#                 ax.vlines(x_split, y_bounds[0], y_bounds[1], color=color, linewidth=2.5)
#                 left_box = ((x_bounds[0], x_split), y_bounds)
#                 right_box = ((x_split, x_bounds[1]), y_bounds)
#         else:  # Oblique split (non-vertical)
#             y_vals = -(w[0] * x_vals + b) / w[1]
#             line = LineString([(x, y) for x, y in zip(x_vals, y_vals)])
#             clipped_line = line.intersection(clipped_polygon)
#             if isinstance(clipped_line, LineString):
#                 x_plot, y_plot = clipped_line.xy
#                 ax.plot(x_plot, y_plot, color=color, linewidth=2.5)
#
#             left_box = (x_bounds, y_bounds)
#             right_box = (x_bounds, y_bounds)
#
#         # Recurse into children with clipped subspaces
#         if len(node.children) == 2:
#             plot_split(node.children[0], *left_box, depth + 1)
#             plot_split(node.children[1], *right_box, depth + 1)
#
#     plot_split(tree.root, xlim, ylim, depth=0)  # Start plotting from the root node
#     ax.set_xlim(xlim)
#     ax.set_ylim(ylim)
#     ax.set_title("Oblique Decision Boundaries (Leaf Nodes Only)")
#     ax.set_xlabel("x1")
#     ax.set_ylabel("x2")
#     plt.tight_layout()
