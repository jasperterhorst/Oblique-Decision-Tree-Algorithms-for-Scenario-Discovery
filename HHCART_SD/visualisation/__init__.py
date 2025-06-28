"""
Visualization Toolkit for Oblique Trees
========================================

This subpackage contains tools to visualize tree structures and evaluate scenario
discovery metrics. Plots include structure, boundaries, regions, and metrics over depth.

Modules:
--------
- clipped_boundaries.py                 -       2D oblique decision regions
- coverage_density_path.py              -       Tradeoff trajectory plot
- node_size_distribution.py             -       Node population plots
- performance_metrics_vs_structure.py   -       Metrics across depths
- regions.py                            -       Polygonal regions per tree
- tree_structure.py                     -       Tree architecture plots
- split_explanation.py                  -       Visualize oblique split construction

Usage Example:
--------------
hh.plot_tree_structure(depth=3)
hh.plot_splits_2d_grid()
hh.plot_tradeoff_path()
"""

import functools

from .tree_structure import plot_tree_structure as _plot_tree_structure
from .regions import plot_regions_2d_grid as _plot_regions_2d_grid
from .clipped_boundaries import (plot_splits_2d_grid as _plot_splits_2d_grid,
                                 plot_splits_2d_overlay as _plot_splits_2d_overlay)
from .coverage_density_path import plot_tradeoff_path as _plot_tradeoff_path
from .performance_metrics_vs_structure import plot_metrics_vs_structure as _plot_metrics_vs_structure
from .node_size_distribution import plot_node_size_distribution as _plot_node_size_distribution
from .split_explanation import plot_split_explanation_2d as _plot_split_explanation_2d


def bind_plotting_methods(obj):
    """
    Dynamically bind visualization methods to an HHCartD object instance.

    This function attaches plotting methods directly to the HHCartD instance so
    users can visualize model structure, metrics, and regions without calling
    individual plotting modules.

    Raises:
        AttributeError: If required attributes are missing on the object.
    """
    # === Structure plots ===
    obj.plot_tree_structure = functools.wraps(_plot_tree_structure)(
        lambda *args, **kwargs: _plot_tree_structure(obj, *args, **kwargs)
    )
    obj.plot_regions_2d_grid = functools.wraps(_plot_regions_2d_grid)(
        lambda *args, **kwargs: _plot_regions_2d_grid(obj, *args, **kwargs)
    )
    obj.plot_splits_2d_grid = functools.wraps(_plot_splits_2d_grid)(
        lambda *args, **kwargs: _plot_splits_2d_grid(obj, *args, **kwargs)
    )
    obj.plot_splits_2d_overlay = functools.wraps(_plot_splits_2d_overlay)(
        lambda *args, **kwargs: _plot_splits_2d_overlay(obj, *args, **kwargs)
    )
    obj.plot_split_explanation = functools.wraps(_plot_split_explanation_2d)(
        lambda *args, **kwargs: _plot_split_explanation_2d(obj, *args, **kwargs)
    )

    # === Metrics plots ===
    obj.plot_tradeoff_path = functools.wraps(_plot_tradeoff_path)(
        lambda *args, **kwargs: _plot_tradeoff_path(obj, *args, **kwargs)
    )
    obj.plot_metrics_vs_structure = functools.wraps(_plot_metrics_vs_structure)(
        lambda *args, **kwargs: _plot_metrics_vs_structure(obj, *args, **kwargs)
    )
    obj.plot_node_size_distribution = functools.wraps(_plot_node_size_distribution)(
        lambda *args, **kwargs: _plot_node_size_distribution(obj, *args, **kwargs)
    )
