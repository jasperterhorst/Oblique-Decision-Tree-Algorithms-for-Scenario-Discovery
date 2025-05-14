"""
Visualization package for C_oblique_decision_tree_benchmark.

Exports functions for decision boundaries, trade-off plots, and result plotting.
"""

from .single_run_plots import (print_tree_structure, plot_decision_boundaries, plot_coverage_density_for_shape,
                               plot_decision_regions_from_dict, plot_oblique_splits_from_dict)
from .batch_results_plots import (plot_coverage_density_all_shapes_for_algorithm,
                                  plot_metric_over_depth_by_algorithm_and_group,
                                  plot_runtime_over_depth_grouped_by_data_dim_or_samples,
                                  plot_loglog_runtime_scaling_by_dimension_or_sample_count,
                                  plot_multiple_metrics_over_depth_by_dim_or_sample_size,
                                  plot_multiple_metrics_over_depth_by_label_noise)

__all__ = [
    "print_tree_structure",
    "plot_decision_boundaries",
    "plot_decision_regions_from_dict",
    "plot_oblique_splits_from_dict",
    "plot_coverage_density_for_shape",
    "plot_coverage_density_all_shapes_for_algorithm",
    "plot_metric_over_depth_by_algorithm_and_group",
    "plot_runtime_over_depth_grouped_by_data_dim_or_samples",
    "plot_loglog_runtime_scaling_by_dimension_or_sample_count",
    "plot_multiple_metrics_over_depth_by_dim_or_sample_size",
    "plot_multiple_metrics_over_depth_by_label_noise"
]
