"""
Package for applying scenario discovery methods to demonstrate how they work on a synthetic dataset.
"""

from .analysis import run_analysis
from .prim_module import manual_prim, compute_box_metrics
from .pca_rotation_module import apply_pca, rotate_data, transform_box_to_original
from .cart_module import run_cart
from .utils import lighten_color, interpolate_color, setup_matplotlib, get_points_in_box
from .notebook_helpers import update_plots, save_prim_plots, save_cart_plots
from . import plotting

__all__ = [
    "run_analysis",
    "manual_prim",
    "compute_box_metrics",
    "apply_pca",
    "rotate_data",
    "transform_box_to_original",
    "run_cart",
    "lighten_color",
    "interpolate_color",
    "setup_matplotlib",
    "get_points_in_box",
    "update_plots",
    "save_prim_plots",
    "save_cart_plots",
    "plotting",
]
