"""
Utility functions for plotting and geometric operations.
Includes color helpers, box indexing, and matplotlib styling.
"""

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb, to_hex
from typing import Tuple, cast


def lighten_color(color: str, amount: float = 0.5) -> str:
    """
    Lighten a hex color by mixing it with white.

    Args:
        color: A hex string (e.g., '#A9BA8F').
        amount: Blend factor (0 = no change, 1 = white).

    Returns:
        Hex color string.
    """
    rgb: Tuple[float, float, float] = to_rgb(color)
    light = tuple(c + (1.0 - c) * amount for c in rgb)
    light_rgb = cast(Tuple[float, float, float], light)
    return to_hex(light_rgb)


def interpolate_color(color1: str, color2: str, t: float) -> str:
    """
    Interpolate between two hex colors.

    Args:
        color1: Start color (hex).
        color2: End color (hex).
        t: Interpolation factor between 0 and 1.

    Returns:
        Interpolated hex color.
    """
    c1: Tuple[float, float, float] = to_rgb(color1)
    c2: Tuple[float, float, float] = to_rgb(color2)
    blended = tuple(a * (1 - t) + b * t for a, b in zip(c1, c2))
    blended_rgb = cast(Tuple[float, float, float], blended)
    return to_hex(blended_rgb)


def setup_matplotlib() -> None:
    """
    Set standard matplotlib parameters for consistent styling.
    """
    import matplotlib.pyplot as plt
    plt.rcParams["text.usetex"] = False
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]


def get_points_in_box(box: dict, data: pd.DataFrame) -> np.ndarray:
    """
    Get indices of all points within a rectangular box.

    Args:
        box: Dictionary with 'x' and 'y' keys and (min, max) bounds.
        data: DataFrame with 'x' and 'y' columns.

    Returns:
        Numpy array of indices of points inside the box.
    """
    return data.index[
        (data["x"] >= box["x"][0]) & (data["x"] <= box["x"][1]) &
        (data["y"] >= box["y"][0]) & (data["y"] <= box["y"][1])
    ].values
