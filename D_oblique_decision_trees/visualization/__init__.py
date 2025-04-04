"""
Visualization package for D_oblique_decision_trees.

Exports functions for decision boundaries, trade-off plots, and result plotting.
"""

from .decision_boundaries import print_tree_structure, plot_decision_boundaries
from .tradeoff_plots import plot_tradeoff_for_oblique_tree
from .results_plots import (plot_metric_vs_depth_per_dataset_and_algorithm, plot_metric_vs_depth_per_algorithm,
                            plot_metric_vs_depth_per_shape)

__all__ = [
    "print_tree_structure",
    "plot_decision_boundaries",
    "plot_tradeoff_for_oblique_tree",
    "plot_metric_vs_depth_per_dataset_and_algorithm",
    "plot_metric_vs_depth_per_algorithm",
    "plot_metric_vs_depth_per_shape"
]
