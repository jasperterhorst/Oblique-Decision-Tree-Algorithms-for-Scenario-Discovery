"""
Module for plotting decision boundaries and printing tree structures.

Provides functions to visualize the structure of decision trees.
"""

import matplotlib.pyplot as plt
import numpy as np
from D_restructured.core.tree import DecisionNode, LeafNode


def print_tree_structure(tree):
    """Traverse and print the tree structure."""
    tree.traverse(lambda node: print(repr(node)))


def plot_decision_boundaries(tree, X, y, ax=None, show_data=True, xlim=(-1, 1), ylim=(-1, 1)):
    """
    Plot decision boundaries of a decision tree on 2D data.

    Parameters:
        tree: A standardized DecisionTree.
        X (np.ndarray): Input features.
        y (np.ndarray): True labels.
        ax (matplotlib.axes.Axes, optional): Axes to plot on.
        show_data (bool): Whether to overlay data points.
        xlim (tuple): x-axis limits.
        ylim (tuple): y-axis limits.
    """
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
                return  # Skip non-2D decision nodes
            if np.isclose(w[1], 0.0):
                x_line = -b / w[0]
                ax.axvline(x_line, color='k', linestyle='--', linewidth=1)
            else:
                y_vals = -(w[0] * x_vals + b) / w[1]
                ax.plot(x_vals, y_vals, 'k--', linewidth=1)
            for child in node.children:
                plot_split(child)
    plot_split(tree.root)
    ax.set_title("Oblique Decision Boundaries")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    plt.tight_layout()

    return ax
