"""
Visualization package for D_oblique_decision_trees.

Exports functions for decision boundaries, trade-off plots, and result plotting.
"""

from .decision_boundaries import print_tree_structure, plot_decision_boundaries
from .tradeoff_plots import plot_tradeoff_for_oblique_tree
from .batch_results_plots import (plot_metric_vs_depth_per_dataset_and_algorithm, plot_metric_vs_depth_per_algorithm,
                                  plot_metric_vs_depth_per_shape, plot_seed_std_vs_depth_per_algorithm,
                                  plot_coverage_density_all_shapes_for_algorithm)

__all__ = [
    "print_tree_structure",
    "plot_decision_boundaries",
    "plot_tradeoff_for_oblique_tree",
    "plot_metric_vs_depth_per_dataset_and_algorithm",
    "plot_metric_vs_depth_per_algorithm",
    "plot_metric_vs_depth_per_shape",
    "plot_seed_std_vs_depth_per_algorithm",
    "plot_coverage_density_all_shapes_for_algorithm"
]
