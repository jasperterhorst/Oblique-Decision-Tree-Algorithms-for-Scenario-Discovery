"""
C_oblique_decision_trees package

This package provides the tools to work with oblique decision trees, including:
  – Core data structures (tree definitions)
  – Converters to standardize various decision tree models
  – Evaluation routines and benchmarking tools
  – Visualization utilities for analyzing tree performance and interpretability
"""

from .core.tree import TreeNode, DecisionNode, LeafNode, DecisionTree
from .converters.dispatcher import convert_tree
from .evaluation import evaluate_tree, DepthSweepRunner
from .visualization import (
    print_tree_structure,
    plot_decision_boundaries,
    plot_metric_vs_depth_per_dataset_and_algorithm,
    plot_metric_vs_depth_per_algorithm,
    plot_metric_vs_depth_per_shape,
    plot_seed_std_vs_depth_per_algorithm,
    plot_coverage_density_all_shapes_for_algorithm
)

__all__ = [
    "TreeNode", "DecisionNode", "LeafNode", "DecisionTree",
    "convert_tree",
    "evaluate_tree", "DepthSweepRunner",
    "print_tree_structure", "plot_decision_boundaries",
    "plot_metric_vs_depth_per_dataset_and_algorithm", "plot_metric_vs_depth_per_algorithm",
    "plot_metric_vs_depth_per_shape", "plot_seed_std_vs_depth_per_algorithm",
    "plot_coverage_density_all_shapes_for_algorithm"
]
