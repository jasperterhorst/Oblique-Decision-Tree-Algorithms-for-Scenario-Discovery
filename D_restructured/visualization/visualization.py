import matplotlib.pyplot as plt
import numpy as np
from D_oblique_decision_trees.tree import DecisionNode, LeafNode
from matplotlib import cm
from shapely.geometry import Polygon, LineString
from matplotlib.collections import LineCollection


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


def plot_tradeoff_for_oblique_tree(df, algorithm, dataset,
                                   color_metric="depth", cmap="viridis"):
    """
    Plots a trade-off scatter plot of coverage vs. density for a given algorithm and dataset.
    The scatter points and the connecting gradient line are colored based on the selected interpretability metric.
    Accepted values for 'color_metric' include:
        - "depth"
        - "avg_active_feature_count"
        - "feature_utilisation_ratio"
        - "tree_level_sparsity_index"

    If any of the metrics (coverage, density, or the chosen color_metric) are np.nan, they are replaced with 0.

    Parameters:
        df (pd.DataFrame): DataFrame with evaluation metrics.
        algorithm (str): The algorithm name to filter the data.
        dataset (str): The dataset name to filter the data.
        color_metric (str): The interpretability metric for coloring (default "depth").
        cmap (str): Colormap to use (default "viridis").
    """
    # Filter the DataFrame by algorithm and dataset.
    subset = df[(df["algorithm"] == algorithm) & (df["dataset"] == dataset)].copy()

    # Replace np.nan in coverage, density, and the color_metric with 0.
    subset["coverage"] = subset["coverage"].fillna(0)
    subset["density"] = subset["density"].fillna(0)
    subset[color_metric] = subset[color_metric].fillna(0)

    # Set up normalization based on the chosen color metric.
    metric_values = subset[color_metric].to_numpy()
    min_metric = metric_values.min() if len(metric_values) > 0 else 0
    max_metric = metric_values.max() if len(metric_values) > 0 else 1
    norm = plt.Normalize(min_metric, max_metric)

    plt.figure(figsize=(8, 6))

    # Scatter plot: points colored by the chosen interpretability metric.
    scatter = plt.scatter(subset["coverage"], subset["density"],
                          c=subset[color_metric], cmap=cmap, norm=norm,
                          s=50, edgecolor='k', alpha=0.8)
    cbar = plt.colorbar(scatter)
    cbar.set_label(color_metric.replace("_", " ").title())

    # Sort the subset by the chosen metric for drawing the line progression.
    sorted_subset = subset.sort_values(by=color_metric)
    x = sorted_subset["coverage"].to_numpy()
    y = sorted_subset["density"].to_numpy()
    metric_sorted = sorted_subset[color_metric].to_numpy()

    # Create segments for a gradient line.
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Compute the average metric value for each segment.
    avg_metric = (metric_sorted[:-1] + metric_sorted[1:]) / 2

    # Create a LineCollection with the segments.
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(avg_metric)
    lc.set_linewidth(2)
    plt.gca().add_collection(lc)

    plt.xlabel("Coverage")
    plt.ylabel("Density")
    plt.title(
        f"Trade-off Plot for {algorithm} on {dataset.replace('_', ' ').title()}\nColored by {color_metric.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.show()
