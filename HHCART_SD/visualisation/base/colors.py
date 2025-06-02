"""
Color and Style Definitions for HHCART_SD Visualisations
------------------------------------------------------
Centralized palette and helper utilities for consistent color use across
HHCART_SD decision tree visualisation tools.
"""

import numpy as np
from matplotlib.colors import to_rgb, to_hex
import matplotlib.colors as mcolors


# === Utility: Compute midpoint color ===
def midpoint_hex(color1: str, color2: str) -> str:
    """Return the midpoint color as a hex string between two color codes."""
    rgb1 = np.array(to_rgb(color1))
    rgb2 = np.array(to_rgb(color2))
    midpoint_rgb = ((rgb1 + rgb2) / 2).tolist()
    return to_hex(midpoint_rgb)


# === Classification Green–Orange Palette ===
PRIMARY_LIGHT = "#A9BA8F"
PRIMARY_DARK = "#2C5D2D"
PRIMARY_MIDDLE = midpoint_hex(PRIMARY_LIGHT, PRIMARY_DARK)

SECONDARY_LIGHT = "#FFCC99"
SECONDARY_DARK = "#7F3300"
SECONDARY_MIDDLE = midpoint_hex(SECONDARY_LIGHT, SECONDARY_DARK)

# === Use in Classification Scatter Plots ===
SCATTER_COLORS = {
    "no_interest": PRIMARY_LIGHT,
    "interest": PRIMARY_DARK,
}

EVOLUTION_COLORS = {
    "start": SECONDARY_LIGHT,
    "end": SECONDARY_DARK,
    "middle": SECONDARY_MIDDLE,
}


# === Gradients ===
def generate_color_gradient(base_color: str, n_levels: int) -> list:
    """
    Generate a perceptually balanced gradient based on a base color.

    Args:
        base_color (str): Base color in hex or matplotlib name.
        n_levels (int): Number of color levels to produce.

    Returns:
        list: List of RGBA color tuples.
    """
    base_rgb = mcolors.to_rgb(base_color)
    light_rgb = tuple(min(1.0, 1 - 0.4 * (1 - c)) for c in base_rgb)
    dark_rgb = tuple(max(0.0, 0.5 * c) for c in base_rgb)

    gradient_rgb = [
        (0.0, light_rgb),
        (0.5, base_rgb),
        (1.0, dark_rgb),
    ]

    cmap = mcolors.LinearSegmentedColormap.from_list("custom_interp", gradient_rgb, N=n_levels)
    return [cmap(i / (n_levels - 1)) for i in range(n_levels)]
