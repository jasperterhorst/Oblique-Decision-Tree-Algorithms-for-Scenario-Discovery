"""
Visualization package for D_oblique_decision_trees.

Exports functions for decision boundaries, trade-off plots, and result plotting.
"""

from .decision_boundaries import print_tree_structure, plot_decision_boundaries
from .tradeoff_plots import plot_tradeoff_for_oblique_tree
from .results_plots import plot_results, plot_aggregated_metric

__all__ = [
    "print_tree_structure",
    "plot_decision_boundaries",
    "plot_tradeoff_for_oblique_tree",
    "plot_results",
    "plot_aggregated_metric"
]
