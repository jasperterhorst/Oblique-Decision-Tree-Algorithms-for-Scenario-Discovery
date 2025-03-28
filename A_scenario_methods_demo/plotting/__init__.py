"""
Plotting package for the scenario methods demo.
"""

from .base_plots import generic_plot, plot_base
from .prim_plots import (
    plot_original_data,
    plot_spatial_evolution,
    plot_peeling_trajectory,
    plot_rotated_data,
    plot_rotated_with_boxes,
    plot_original_with_boxes,
    plot_overlayed_peeling_trajectories,
    plot_peeling_trajectory_with_constraint_colors,
)
from .cart_plots import plot_cart_spatial_evolution

__all__ = [
    "generic_plot", "plot_base",
    "plot_original_data", "plot_spatial_evolution", "plot_peeling_trajectory",
    "plot_rotated_data", "plot_rotated_with_boxes", "plot_original_with_boxes",
    "plot_overlayed_peeling_trajectories", "plot_peeling_trajectory_with_constraint_colors",
    "plot_cart_spatial_evolution",
]
