"""
PCA utilities for rotating _data in scenario discovery.
Used in PCA-PRIM to align input space along principal components.
"""

import numpy as np


def apply_pca(x: np.ndarray, y: np.ndarray, vulnerable_value: int = 1):
    """
    Perform PCA on a subset of X where y == vulnerable_value.

    Args:
        x: Input _data, shape (n_samples, n_features).
        y: Binary labels for each sample.
        vulnerable_value: Value indicating policy-relevant cases (default = 1).

    Returns:
        mu: Mean vector of the subset (shape: [n_features]).
        V: Rotation matrix (columns = principal components, ordered by variance).
    """
    mask = (y == vulnerable_value)
    X_vulnerable = x[mask]

    mu = X_vulnerable.mean(axis=0)
    X_centered = X_vulnerable - mu

    cov_matrix = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eig(cov_matrix)

    sorted_idx = np.argsort(eigvals)[::-1]
    V = eigvecs[:, sorted_idx]

    return mu, V


def rotate_data(x: np.ndarray, mu: np.ndarray, v: np.ndarray):
    """
    Rotate full dataset based on PCA rotation.

    Args:
        x: Original _data, shape (n_samples, n_features).
        mu: Mean vector of vulnerable subset (used for centering).
        v: Rotation matrix from PCA.

    Returns:
        X_rotated: Rotated _data (centered and projected).
    """
    return (x - mu) @ v


def transform_box_to_original(box_rot: dict, v: np.ndarray, mu: np.ndarray):
    """
    Transform a rectangular box from rotated space back to original space.

    Args:
        box_rot: Dictionary with axis-aligned bounds in rotated space.
                 Example: { "x": (x0, x1), "y": (y0, y1) }
        v: Rotation matrix (from PCA).
        mu: Mean vector used for centering.

    Returns:
        ndarray: Array of shape (4, 2) for the 4 corners of the transformed box.
    """
    x0, x1 = box_rot["x"]
    y0, y1 = box_rot["y"]

    corners_rot = np.array([
        [x0, y0],
        [x1, y0],
        [x1, y1],
        [x0, y1]
    ])

    corners_original = corners_rot @ v.T + mu
    return corners_original
