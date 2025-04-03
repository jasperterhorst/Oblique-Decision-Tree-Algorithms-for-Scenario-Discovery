"""
Module for generating trade-off plots.

Contains functions to create plots that visualize trade-offs between evaluation metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def plot_tradeoff_for_oblique_tree(df, algorithm, dataset, color_metric="depth", cmap="viridis"):
    """
    Plot a trade-off scatter plot of coverage vs. density for a given algorithm and dataset.

    Parameters:
        df (pd.DataFrame): DataFrame with evaluation metrics.
        algorithm (str): The algorithm to filter by.
        dataset (str): The dataset to filter by.
        color_metric (str): The interpretability metric for coloring.
        cmap (str): Colormap to use.
    """
    subset = df[(df["algorithm"] == algorithm) & (df["dataset"] == dataset)].copy()
    subset["coverage"] = subset["coverage"].fillna(0)
    subset["density"] = subset["density"].fillna(0)
    subset[color_metric] = subset[color_metric].fillna(0)

    metric_values = subset[color_metric].to_numpy()
    min_metric = metric_values.min() if len(metric_values) > 0 else 0
    max_metric = metric_values.max() if len(metric_values) > 0 else 1
    norm = plt.Normalize(min_metric, max_metric)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(subset["coverage"], subset["density"],
                         c=subset[color_metric], cmap=cmap, norm=norm,
                         s=50, edgecolor='k', alpha=0.8)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(color_metric.replace("_", " ").title())

    sorted_subset = subset.sort_values(by=color_metric)
    x = sorted_subset["coverage"].to_numpy()
    y = sorted_subset["density"].to_numpy()
    metric_sorted = sorted_subset[color_metric].to_numpy()
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    avg_metric = (metric_sorted[:-1] + metric_sorted[1:]) / 2
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(avg_metric)
    lc.set_linewidth(2)
    ax.add_collection(lc)

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Density")
    ax.set_title(f"Trade-off Plot for {algorithm} on {dataset.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.show()
