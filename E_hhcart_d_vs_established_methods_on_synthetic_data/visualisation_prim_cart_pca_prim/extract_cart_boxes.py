"""
Extract box definitions from CART model leaves predicting class 1.

Provides utilities to convert leaf nodes into polygon coordinates for plotting
and structured box definitions for further analysis.

Author: Jasper ter Horst (2025)
"""

import numpy as np
import pandas as pd
from ema_workbench.analysis import scenario_discovery_util as sdutil


def extract_cart_boxes(cart_model):
    """
    Extract boxes from a fitted CART model corresponding to leaves predicting class 1.

    Parameters
    ----------
    cart_model : CART
        A fitted CART wrapper instance with attribute `clf` as the sklearn tree,
        and `x` as the original DataFrame input features.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per box (leaf), columns include:
        - 'box_id': leaf index
        - 'coverage': fraction of total class 1 points covered by box
        - 'density': purity of box (fraction of points in box that are class 1)
        - 'mass': fraction of total points in box
        - 'corner_coords': np.ndarray of shape (2^d, d) representing all corner points of the box
        - 'box_limits': DataFrame with min and max per feature (same as box limits)

    Notes
    -----
    Assumes binary classification and class label 1 is the positive target class.
    """
    # Validate fitted model
    if not hasattr(cart_model, "clf"):
        raise ValueError("cart_model must be fitted and have 'clf' attribute")

    tree_ = cart_model.clf.tree_
    features = cart_model.feature_names
    box_init = sdutil._make_box(cart_model.x)
    y = cart_model.y

    # Get total class 1 count
    total_class1 = np.sum(y == 1)

    left = tree_.children_left
    right = tree_.children_right
    threshold = tree_.threshold
    value = tree_.value  # shape: (n_nodes, n_classes) for classifier

    leaf_nodes = np.where(left == -1)[0]

    stats = []

    def recurse_path(node_id):
        """Recursively build the path constraints from root to node."""
        path = []
        while node_id != 0:
            parent = np.where((left == node_id) | (right == node_id))[0][0]
            direction = "left" if left[parent] == node_id else "right"
            path.append((parent, direction))
            node_id = parent
        return path[::-1]

    def apply_path_constraints(path):
        box = box_init.copy()
        for parent, direction in path:
            feat = features[tree_.feature[parent]]
            thresh = threshold[parent]

            if feat not in box.columns:
                # nominal variable - skipping (optional: implement later)
                continue

            if direction == "left":
                box.loc[1, feat] = min(box.loc[1, feat], thresh)
            else:
                box.loc[0, feat] = max(box.loc[0, feat], thresh)
        return box

    for leaf in leaf_nodes:
        leaf_value = value[leaf][0]
        if len(leaf_value) == 1:
            # Regression leaf — skip
            continue

        predicted_class = np.argmax(leaf_value)

        if predicted_class != 1:
            continue  # skip leaves not predicting class 1

        # Reconstruct box
        path = recurse_path(leaf)
        box_limits = apply_path_constraints(path)

        # Select points in box
        idx_in_box = sdutil._in_box(cart_model.x, box_limits)
        y_in_box = y[idx_in_box]

        if len(y_in_box) == 0:
            continue  # skip empty box

        # Compute box statistics
        box_coi = np.sum(y_in_box == 1)
        coverage = box_coi / total_class1 if total_class1 > 0 else 0
        density = box_coi / len(y_in_box)
        mass = len(y_in_box) / len(y)

        # Compute box corner coordinates (2^d corners)
        mins = box_limits.loc[0].values
        maxs = box_limits.loc[1].values
        d = len(mins)

        # Generate all binary combinations (corner patterns)
        binary_patterns = np.array(
            [list(np.binary_repr(i, width=d)) for i in range(2 ** d)], dtype=int
        )
        corners = np.where(binary_patterns == 0, mins, maxs)

        stats.append(
            {
                "box_id": leaf,
                "coverage": coverage,
                "density": density,
                "mass": mass,
                "coords": corners,
                "box_limits": box_limits,
            }
        )

    return pd.DataFrame(stats)
