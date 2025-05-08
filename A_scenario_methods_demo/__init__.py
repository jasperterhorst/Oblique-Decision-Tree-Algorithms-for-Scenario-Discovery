"""
Scenario Methods Demo Package.

Provides all core functions and visualisation tools for applying and showcasing
PRIM, PCA–PRIM, and CART on a synthetic dataset.

Main components:
- `analysis`: Runs full scenario discovery pipelines.
- `prim_module`: Manual PRIM logic and box metrics.
- `pca_rotation_module`: PCA rotation and inverse transforms.
- `cart_module`: CART-based decision region extraction.
- `notebook_helpers`: Save + update utilities for interactive demos.
- `utils`: Shared low-level utilities.
- `plotting`: Unified plotting interface (PRIM, PCA–PRIM, CART).
"""

from .analysis import run_analysis
from .prim_module import manual_prim, compute_box_metrics
from .pca_rotation_module import apply_pca, rotate_data, transform_box_to_original
from .cart_module import run_cart
from .utils import (
    lighten_color,
    interpolate_color,
    setup_matplotlib,
    get_points_in_box,
)
from .notebook_helpers import (
    update_plots,
    save_prim_plots,
    save_cart_plots,
)
from . import plotting

__all__ = [
    # Core analysis
    "run_analysis",
    "manual_prim",
    "compute_box_metrics",
    "apply_pca",
    "rotate_data",
    "transform_box_to_original",
    "run_cart",

    # Utilities
    "lighten_color",
    "interpolate_color",
    "setup_matplotlib",
    "get_points_in_box",

    # Notebook helpers
    "update_plots",
    "save_prim_plots",
    "save_cart_plots",

    # Visualisation module
    "plotting",
]
