"""
Visualization package for C_oblique_decision_trees.

Exports functions for decision boundaries, trade-off plots, and result plotting.
"""

from .single_run_plots import (print_tree_structure, plot_decision_boundaries, plot_coverage_density_for_shape,
                               plot_decision_regions_from_dict, plot_oblique_splits_from_dict)
from .batch_results_plots import (plot_metric_vs_depth_per_dataset_and_algorithm, plot_metric_vs_depth_per_algorithm,
                                  plot_metric_vs_depth_per_shape, plot_seed_std_vs_depth_per_algorithm,
                                  plot_coverage_density_all_shapes_for_algorithm)

__all__ = [
    "print_tree_structure",
    "plot_decision_boundaries",
    "plot_decision_regions_from_dict",
    "plot_oblique_splits_from_dict",
    "plot_coverage_density_for_shape",
    "plot_metric_vs_depth_per_dataset_and_algorithm",
    "plot_metric_vs_depth_per_algorithm",
    "plot_metric_vs_depth_per_shape",
    "plot_seed_std_vs_depth_per_algorithm",
    "plot_coverage_density_all_shapes_for_algorithm",
]
