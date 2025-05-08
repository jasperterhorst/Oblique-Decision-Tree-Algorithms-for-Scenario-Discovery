"""
Centralized color definitions for plots and visuals.
"""

from matplotlib.colors import to_rgb, to_hex
import numpy as np


def midpoint_hex(color1: str, color2: str) -> str:
    """Return the midpoint color as a hex string between two hex color codes."""
    rgb1 = np.array(to_rgb(color1))
    rgb2 = np.array(to_rgb(color2))
    midpoint_rgb = ((rgb1 + rgb2) / 2).tolist()  # Convert ndarray to list
    return to_hex(midpoint_rgb)


# === Primary color set (main classification palette) ===
PRIMARY_LIGHT = "#A9BA8F"   # Light green (e.g., Not of Interest)
PRIMARY_DARK = "#2C5D2D"    # Dark green (e.g., Of Interest)
PRIMARY_MIDDLE = midpoint_hex(PRIMARY_LIGHT, PRIMARY_DARK)

# === Secondary color set (e.g., for box evolution, peeling trajectory) ===
SECONDARY_LIGHT = "#FFCC99"  # Light orange (e.g., start of evolution)
SECONDARY_DARK = "#7F3300"   # Dark orange-brown (e.g., end of evolution)
SECONDARY_MIDDLE = midpoint_hex(SECONDARY_LIGHT, SECONDARY_DARK)

# === Structural / layout colors ===
CART_OUTLINE_COLOR = "#808080"      # Grey for CART box outlines
QUADRILATERAL_COLOR = "lightgrey"   # Background polygon
AXIS_LINE_COLOR = "#808080"         # Axes lines
GRID_COLOR = "#B0B0B0"              # Grid lines

# === Optional named roles (if you want semantic access) ===
SCATTER_COLORS = {
    "no_interest": PRIMARY_LIGHT,
    "interest": PRIMARY_DARK,
}

EVOLUTION_COLORS = {
    "start": SECONDARY_LIGHT,
    "end": SECONDARY_DARK,
}
