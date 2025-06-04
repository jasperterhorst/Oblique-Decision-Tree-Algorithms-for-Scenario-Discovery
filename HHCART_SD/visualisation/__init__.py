"""
Visualization Toolkit for Oblique Trees
========================================

This subpackage contains tools to visualize tree structures and evaluate scenario
discovery metrics. Plots include structure, boundaries, metrics over depth, and
polygonal decision regions.

Modules:
--------
- clipped_boundaries.py         : 2D oblique decision regions
- coverage_density_path.py      : Tradeoff trajectory plot
- node_size_distribution.py     : Node population plots
- performance_metrics_vs_structure.py : Metrics across depths
- regions.py                    : Polygonal regions per tree
- tree_structure.py             : Tree architecture plots
- base/                         : Plot settings, color maps, saving

Usage Example:
--------------
hh.plot_tree_structure(depth=3)
hh.plot_clipped_boundaries()
hh.plot_tradeoff_path()
"""

from .coverage_density_path import plot_tradeoff_path as _plot_tradeoff_path
from .clipped_boundaries import (plot_splits_2d_grid as _plot_splits_2d_grid,
                                 plot_splits_2d_overlay as _plot_splits_2d_overlay)
from .tree_structure import plot_tree_structure as _plot_tree_structure
from .performance_metrics_vs_structure import plot_metrics_vs_structure as _plot_metrics_vs_structure
from .node_size_distribution import plot_node_size_distribution as _plot_node_size_distribution
from .regions import plot_regions_2d_grid as _plot_regions_2d_grid


def bind_plotting_methods(obj):
    """
    Dynamically bind visualization methods to an HHCartD object instance.

    This function attaches plotting methods directly to the HHCartD instance so
    users can visualize model structure, metrics, and regions without calling
    individual plotting modules.

    Raises:
        AttributeError: If required attributes are missing on the object.
    """
    required_attrs = ["trees_by_depth", "X", "y"]
    for attr in required_attrs:
        if not hasattr(obj, attr):
            raise AttributeError(f"The object must have attribute '{attr}'.")

    def plot_tradeoff_path(color_by: str = "depth", save: bool = False, filename: str = None, title: str = None):
        """
        Plot the sequential coverage–density trade-off path across depths.

        Args:
            color_by (str): Variable to color points by ('depth' or 'class1_leaf_count').
            save (bool): Whether to save the figure.
            filename (str, optional): Output filename (PDF).
            title (str, optional): Optional figure title.

        Returns:
            matplotlib.axes.Axes: Axis with the coverage-density trade-off path.

        Side Effects:
            If `save=True`, a PDF is saved to the model’s save directory (`hh.save_dir`) with the specified filename.
        """
        return _plot_tradeoff_path(obj, save=save, filename=filename, title=title, color_by=color_by)

    def plot_splits_2d_grid(save: bool = False, filename: str = None, title: str = None):
        """
        Plot clipped oblique decision boundaries for all trained depths in separate plots.

        Args:
            save (bool): Whether to save the figure.
            filename (str, optional): Output filename (PDF).
            title (str, optional): Optional figure title.

        Returns:
            matplotlib.axes.Axes: Axis with the clipped oblique boundaries plot.

        Side Effects:
            If `save=True`, a PDF is saved to the model’s save directory (`hh.save_dir`) with the specified filename.
        """
        return _plot_splits_2d_grid(obj, save=save, filename=filename, title=title)

    def plot_splits_2d_overlay(depth: int = None, cmap: str = "YlGnBu", save: bool = False, filename: str = None,
                               title: str = None):
        """
        Plot clipped oblique decision boundaries for all trained depths in an overlay.

        Args:
            depth (int, optional): Maximum depth to visualise. If None, all depths are shown.
            cmap (str, optional): Name of the colormap to use for the overlay.
            save (bool): Whether to save the figure.
            filename (str, optional): Output filename (PDF).
            title (str, optional): Optional figure title.

        Returns:
            matplotlib.axes.Axes: Axis with the clipped oblique boundaries overlay plot.

        Side Effects:
            If `save=True`, a PDF is saved to the model’s save directory (`hh.save_dir`) with the specified filename.
        """
        return _plot_splits_2d_overlay(obj, depth=depth, cmap=cmap, save=save, filename=filename, title=title)

    def plot_tree_structure(depth: int = None, coloring: str = "distribution", save: bool = False,
                            filename: str = None, title: str = None):
        """
        Visualize the structure of the oblique tree at a specific depth.

        Args:
            depth (int): Desired depth to visualize.
            coloring (str): Coloring strategy for nodes ('distribution', 'class', 'samples', or 'none').
            save (bool): Whether to save the figure.
            filename (str, optional): Output filename (PDF).
            title (str, optional): Optional figure title.

        Returns:
            graphviz.Digraph: Graphviz object representing the oblique decision tree structure,
            which can be rendered or exported manually.

        Side Effects:
            If `save=True`, a PDF is saved to the model’s save directory (`hh.save_dir`) with the specified filename.
        """
        return _plot_tree_structure(obj, depth=depth, coloring=coloring, save=save, filename=filename,
                                    title=title)

    def plot_metrics_vs_structure(x_axis: str = "depth", save: bool = False, filename: str = None, title: str = None):
        """
        Plot accuracy, coverage, and density across depths.

        Args:
            x_axis (str): X-axis variable. Options: 'depth' (default) or 'class1_leaf_count'.
            save (bool): Whether to save the figure.
            filename (str, optional): Output filename (PDF).
            title (str, optional): Optional figure title.

        Returns:
            matplotlib.axes.Axes: Axis with the metrics plot.

        Side Effects:
            If `save=True`, a PDF is saved to the model’s save directory (`hh.save_dir`) with the specified filename.
        """
        return _plot_metrics_vs_structure(obj, save=save, filename=filename, title=title, x_axis=x_axis)

    def plot_node_size_distribution(save: bool = False, filename: str = None, title: str = None):
        """
        Plot the distribution of node sizes (sample counts) grouped by depth.

        Args:
            save (bool): Whether to save the figure.
            filename (str, optional): Output filename (PDF).
            title (str, optional): Optional plot title.

        Returns:
            matplotlib.axes.Axes: Axis with the node size distribution plot.

        Side Effects:
            If `save=True`, a PDF is saved to the model’s save directory (`hh.save_dir`) with the specified filename.
        """
        return _plot_node_size_distribution(obj, save=save, filename=filename, title=title)

    def plot_regions_2d_grid(save: bool = False, filename: str = None, title: str = None):
        """
        Plot oblique decision boundaries clipped to valid polygon regions for all depths.

        Args:
            save (bool): Whether to save the figure.
            filename (str, optional): Output filename (PDF).
            title (str, optional): Optional figure title.

        Returns:
            matplotlib.axes.Axes: Axis with the oblique regions plot.

        Side Effects:
            If `save=True`, a PDF is saved to the model’s save directory (`hh.save_dir`) with the specified filename.
        """
        return _plot_regions_2d_grid(obj, save=save, filename=filename, title=title)

    obj.plot_tradeoff_path = plot_tradeoff_path
    obj.plot_splits_2d_grid = plot_splits_2d_grid
    obj.plot_splits_2d_overlay = plot_splits_2d_overlay
    obj.plot_tree_structure = plot_tree_structure
    obj.plot_metrics_vs_structure = plot_metrics_vs_structure
    obj.plot_node_size_distribution = plot_node_size_distribution
    obj.plot_regions_2d_grid = plot_regions_2d_grid
