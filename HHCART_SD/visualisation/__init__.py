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
- performance_metrics_depthwise.py : Metrics across depths
- regions.py                    : Polygonal regions per tree
- tree_structure.py             : Tree architecture plots
- base/                         : Plot settings, color maps, saving

Usage Example:
--------------
hh.plot_tree_structure(depth=3)
hh.plot_clipped_boundaries()
hh.plot_tradeoff_path()
"""

from .coverage_density_path import plot_coverage_density_tradeoff_path
from .clipped_boundaries import plot_clipped_oblique_splits as _plot_clipped_boundaries
from .tree_structure import plot_tree_structure as _plot_tree_structure
from .performance_metrics_depthwise import plot_metrics_over_depth as _plot_metrics_over_depth
from .node_size_distribution import plot_node_size_distribution as _plot_node_size_distribution
from .regions import plot_oblique_regions as _plot_oblique_regions


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

    def plot_tradeoff_path(save: bool = False, filename: str = None, title: str = None, color_by: str = "depth"):
        """
        Plot the sequential coverage–density trade-off path across depths.

        Args:
            save (bool): Whether to save the figure.
            filename (str, optional): Output filename (PDF).
            title (str, optional): Optional figure title.
            color_by (str): Variable to color points by ('depth' or 'class1_leaf_count').

        Returns:
            matplotlib.axes.Axes: Axis with the coverage-density trade-off path.

        Side Effects:
            If `save=True`, a PDF is saved to the model’s save directory (`hh.save_dir`) with the specified filename.
        """
        return plot_coverage_density_tradeoff_path(obj, save=save, filename=filename, title=title, color_by=color_by)

    def plot_clipped_boundaries(save: bool = False, filename: str = None, title: str = None):
        """
        Plot clipped oblique decision boundaries for all trained depths.

        Args:
            save (bool): Whether to save the figure.
            filename (str, optional): Output filename (PDF).
            title (str, optional): Optional figure title.

        Returns:
            matplotlib.axes.Axes: Axis with the clipped oblique boundaries plot.

        Side Effects:
            If `save=True`, a PDF is saved to the model’s save directory (`hh.save_dir`) with the specified filename.
        """
        return _plot_clipped_boundaries(obj, save=save, filename=filename, title=title)

    def plot_tree_structure(depth: int = None, coloring="distribution", save=False, filename=None, title=None):
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

    def plot_metrics_over_depth(save: bool = False, filename: str = None, title: str = None):
        """
        Plot accuracy, coverage, and density across depths.

        Args:
            save (bool): Whether to save the figure.
            filename (str, optional): Output filename (PDF).
            title (str, optional): Optional figure title.

        Returns:
            matplotlib.axes.Axes: Axis with the metrics plot.

        Side Effects:
            If `save=True`, a PDF is saved to the model’s save directory (`hh.save_dir`) with the specified filename.
        """
        return _plot_metrics_over_depth(obj, save=save, filename=filename, title=title)

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

    def plot_oblique_regions(save: bool = False, filename: str = None, title: str = None):
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
        return _plot_oblique_regions(obj, save=save, filename=filename, title=title)

    obj.plot_tradeoff_path = plot_tradeoff_path
    obj.plot_clipped_boundaries = plot_clipped_boundaries
    obj.plot_tree_structure = plot_tree_structure
    obj.plot_metrics_over_depth = plot_metrics_over_depth
    obj.plot_node_size_distribution = plot_node_size_distribution
    obj.plot_oblique_regions = plot_oblique_regions
