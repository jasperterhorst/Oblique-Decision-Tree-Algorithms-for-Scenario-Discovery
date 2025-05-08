"""
Plotting package for the scenario methods demo.

This package provides core visualisation functions for:
- PRIM and PCA-PRIM box evolution
- Peeling trajectories
- Rotated PCA space visualisation
- CART spatial splits

All functions follow consistent styling and use centralised config for colours and paths.
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
    # Base
    "generic_plot", "plot_base",
    # PRIM & PCA-PRIM
    "plot_original_data",
    "plot_spatial_evolution",
    "plot_peeling_trajectory",
    "plot_rotated_data",
    "plot_rotated_with_boxes",
    "plot_original_with_boxes",
    "plot_overlayed_peeling_trajectories",
    "plot_peeling_trajectory_with_constraint_colors",
    # CART
    "plot_cart_spatial_evolution",
]
