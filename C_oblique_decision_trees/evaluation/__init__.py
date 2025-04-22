"""
evaluation

This module provides a framework for evaluating oblique decision trees and traditional scenario discovery methods.
It includes standardized metrics, evaluation routines, and benchmark execution across datasets and algorithms.
"""

from .metrics import (
    compute_accuracy,
    compute_coverage,
    compute_density,
    compute_f_score,
    compute_leafwise_coverage_density,
    gini_coefficient,
    count_total_constrained_dimensions,
    compute_average_active_feature_count
)

from .io_utils import (
    save_trees_dict,
    load_trees_dict,
    save_depth_sweep_df,
    load_depth_sweep_df
)

from .evaluator import evaluate_tree
from .benchmark_runner import DepthSweepRunner
