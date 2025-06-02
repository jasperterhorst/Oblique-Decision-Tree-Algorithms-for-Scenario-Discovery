"""
colors_and_plot_styles.py

Centralized color definitions and plotting styles for all modules,
including classification visuals and oblique decision tree visuals.
"""

import numpy as np
from matplotlib.colors import to_rgb, to_hex
import matplotlib.colors as mcolors


# === Color Utility ===
def midpoint_hex(color1: str, color2: str) -> str:
    """Return the midpoint color as a hex string between two hex color codes."""
    rgb1 = np.array(to_rgb(color1))
    rgb2 = np.array(to_rgb(color2))
    midpoint_rgb = ((rgb1 + rgb2) / 2).tolist()
    return to_hex(midpoint_rgb)


# === Classification Color Palette (A_scenario_methods_demo) ===
PRIMARY_LIGHT = "#A9BA8F"   # Light green
PRIMARY_DARK = "#2C5D2D"    # Dark green
PRIMARY_MIDDLE = midpoint_hex(PRIMARY_LIGHT, PRIMARY_DARK)

SECONDARY_LIGHT = "#FFCC99"  # Light orange
SECONDARY_DARK = "#7F3300"   # Dark orange
SECONDARY_MIDDLE = midpoint_hex(SECONDARY_LIGHT, SECONDARY_DARK)

CART_OUTLINE_COLOR = "#808080"
QUADRILATERAL_COLOR = "lightgrey"
AXIS_LINE_COLOR = "#808080"
GRID_COLOR = "#B0B0B0"

SCATTER_COLORS = {
    "no_interest": PRIMARY_LIGHT,
    "interest": PRIMARY_DARK,
}

EVOLUTION_COLORS = {
    "start": SECONDARY_LIGHT,
    "end": SECONDARY_DARK,
    "middle": SECONDARY_MIDDLE,
}

# === Oblique Tree Visualization Colors and Styles ===
ALGORITHM_COLORS = {
    "CART": "#1f77b4",
    "HHCART_SD A": "#ff7f0e",
    "HHCART_SD D": "#2ca02c",
    "MOC1": "#d62728",
    "RANDCART": "#9467bd",
    "CO2": "#8c564b",
    "WODT": "#e377c2",
    "RIDGE CART": "#7f7f7f",
}

SHAPE_TYPE_LINESTYLES = {
    "2D": '',            # solid
    "3D": (5, 5),        # dashed
    "Other": (1, 3),     # point, strip, point, stripe
}

NOISE_MARKERS = {
    "Label Noise 000": "o",
    "Label Noise 003": "s",
    "Label Noise 005": "D",
    "Label Noise 007": "X",
}


def get_algorithm_color(name: str) -> str:
    return ALGORITHM_COLORS.get(name, "#000000")


def get_shape_linestyle(shape_type: str):
    return SHAPE_TYPE_LINESTYLES.get(shape_type, '')


def get_noise_marker(label_noise_label: str):
    return NOISE_MARKERS.get(label_noise_label, 'o')


def generate_color_gradient(base_color: str, n_levels: int) -> list:
    """
    Generate a color gradient from a base color for consistent use across plots.

    Parameters:
        base_color (str): Hex or named matplotlib color string.
        n_levels (int): Number of colors to generate.

    Returns:
        list: List of RGBA tuples forming a perceptually balanced gradient.
    """
    base_rgb = mcolors.to_rgb(base_color)
    light_rgb = tuple(min(1.0, 1 - 0.4 * (1 - c)) for c in base_rgb)
    dark_rgb = tuple(max(0.0, 0.5 * c) for c in base_rgb)

    # Disambiguate: explicitly define position-color pairs
    gradient_rgb = [
        (0.0, light_rgb),
        (0.5, base_rgb),
        (1.0, dark_rgb),
    ]

    cmap = mcolors.LinearSegmentedColormap.from_list("custom_interp", gradient_rgb, N=n_levels)
    return [cmap(i / (n_levels - 1)) for i in range(n_levels)]
