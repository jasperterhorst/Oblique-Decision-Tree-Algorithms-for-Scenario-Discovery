"""
Utility functions for handling box representations in scenario discovery.

These functions support converting box limits into corner coordinates,
applying inverse PCA rotations to transformed boxes, and assembling
standardized DataFrame rows for selected boxes with their associated
performance metrics.

This module is intended to be used by extraction and visualization
functions for PRIM, CART, and PCA-PRIM scenario discovery algorithms,
to create comparable and consistent box representations.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from itertools import product


# -----------------------------------------------------------
# Standard PRIM / CART box limits to corners (list of tuples)
# -----------------------------------------------------------

def box_limits_to_corners(box_limits: pd.DataFrame) -> List[Tuple[float, ...]]:
    """
    Given box limits in the form of a DataFrame with index ['min', 'max'] and columns
    as feature names, compute the corners of the box as a list of coordinate tuples.

    Works for arbitrary dimensions, but mostly used for 2D and 3D.

    Parameters
    ----------
    box_limits : pd.DataFrame
        DataFrame with two rows indexed by ['min', 'max'] and columns as feature names.

    Returns
    -------
    corners : List of tuples
        List of all corner coordinates of the hyper-rectangle,
        each as a tuple of feature values.
    """
    features = box_limits.columns.tolist()
    mins = box_limits.loc['min'].values
    maxs = box_limits.loc['max'].values

    # Generate all combinations of min and max for each feature (2^d corners)
    ranges = [(mins[i], maxs[i]) for i in range(len(features))]
    corners = list(product(*ranges))

    return corners


# -----------------------------------------------------------
# PCA-PRIM specific: corners as np.ndarray for rotation
# -----------------------------------------------------------

def compute_corners_from_box_limits(box_limits: pd.DataFrame, features_to_include: List[str] = None) -> np.ndarray:
    """
    Given box limits in PCA space, compute the 2^d corners of the box
    as a matrix (array) of shape (num_corners, n_features).

    Parameters
    ----------
    box_limits : pd.DataFrame
        DataFrame with two rows: either ['min','max'] or [0,1] index.
    features_to_include : List[str], optional
        If provided, only these features will be included.

    Returns
    -------
    corners_pca_space : np.ndarray
        Array of shape (2^d, n_features), each row is a corner.
    """
    # Check the index style
    if all(idx in box_limits.index for idx in ['min', 'max']):
        mins = box_limits.loc['min']
        maxs = box_limits.loc['max']
    elif all(idx in box_limits.index for idx in [0, 1]):
        mins = box_limits.loc[0]
        maxs = box_limits.loc[1]
    else:
        raise ValueError(f"Box limits index {box_limits.index} is not recognized. "
                         f"Expected ['min','max'] or [0,1]. "
                         f"This usually happens if a categorical variable is included. "
                         f"Only numeric variables can be used for polygon corners.")

    # Restrict to requested features (usually 2D numeric features for plotting)
    if features_to_include is not None:
        # First check that all requested features are present:
        missing_features = [f for f in features_to_include if f not in mins.index]
        if missing_features:
            raise ValueError(f"The following requested features are not present in the box limits: {missing_features}")

        mins = mins[features_to_include]
        maxs = maxs[features_to_include]

    # Now safe to build ranges
    from itertools import product
    ranges = []
    for feature in mins.index:
        min_val = mins[feature]
        max_val = maxs[feature]

        # Check that values are numeric
        if not np.issubdtype(type(min_val), np.number) or not np.issubdtype(type(max_val), np.number):
            raise ValueError(f"Feature '{feature}' has non-numeric limits: min={min_val}, max={max_val}. "
                             f"Cannot plot this feature as a polygon.")

        ranges.append((min_val, max_val))

    # Build corners
    corners = list(product(*ranges))

    return np.array(corners)


def apply_inverse_pca_rotation(
    corners_pca_space: np.ndarray,
    pca_rotation_matrix: np.ndarray
) -> np.ndarray:
    """
    Apply inverse PCA rotation to the given corners from PCA space back to
    original input space.

    Parameters
    ----------
    corners_pca_space : np.ndarray
        Array of shape (num_corners, n_features) representing corner coordinates in PCA space.
    pca_rotation_matrix : np.ndarray
        PCA rotation matrix of shape (n_features, n_features).

    Returns
    -------
    corners_original_space : np.ndarray
        Array of shape (num_corners, n_features) of corners in original feature space.
    """
    # The PCA rotation matrix is usually orthogonal, so inverse = transpose
    inverse_rotation = pca_rotation_matrix.T
    corners_original_space = np.dot(corners_pca_space, inverse_rotation)
    return corners_original_space


# -----------------------------------------------------------
# Standardized DataFrame row for any box
# -----------------------------------------------------------

def build_box_row(
    box_id: int,
    coverage: float,
    density: float,
    mass: float,
    res_dim: int,
    coords: List[Tuple[float, ...]],
) -> pd.DataFrame:
    """
    Build a one-row DataFrame describing the box, with stats and coordinates.

    Parameters
    ----------
    box_id : int
        Identifier for the box.
    coverage : float
        Coverage metric for the box.
    density : float
        Density metric for the box.
    mass : float
        Mass metric for the box.
    res_dim : int
        Number of restricted dimensions of the box.
    coords : List of tuples
        List of corner coordinates of the box.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row, columns: ['box_id', 'coverage', 'density', 'mass', 'res_dim', 'coords']
    """
    data = {
        'box_id': [box_id],
        'coverage': [coverage],
        'density': [density],
        'mass': [mass],
        'res_dim': [res_dim],
        'coords': [coords]
    }
    return pd.DataFrame(data)


# -----------------------------------------------------------
# For plotting: order 2D polygon corners
# -----------------------------------------------------------

def order_polygon_corners_2d(corners: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Reorder a list of 2D corner points to form a convex polygon (counterclockwise order).

    Parameters
    ----------
    corners : list of (x, y) tuples

    Returns
    -------
    ordered_corners : list of (x, y) tuples in counterclockwise order
    """
    # Compute centroid
    centroid_x = np.mean([p[0] for p in corners])
    centroid_y = np.mean([p[1] for p in corners])

    # Compute angle for each corner
    def angle(p):
        return np.arctan2(p[1] - centroid_y, p[0] - centroid_x)

    # Sort corners by angle
    ordered_corners = sorted(corners, key=angle)

    # Close the polygon (optional but often nice for plotting)
    ordered_corners.append(ordered_corners[0])

    return ordered_corners
