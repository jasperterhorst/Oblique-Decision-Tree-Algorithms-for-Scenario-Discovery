"""
Split Criteria Module (split_criteria.py)
-----------------------------------------
Implements impurity measures used in decision tree splitting:

- Gini impurity (Breiman et al., 1984)
- Twoing criterion (Murthy et al., 1994)

Each impurity function assumes the following signature:
    impurity(left: np.ndarray, right: np.ndarray) -> float

Inputs must be 1D numpy arrays of class labels (e.g., integers 0/1 for binary classification).
Returns a scalar impurity value. Lower values indicate better splits.
"""

import numpy as np
from collections import Counter


def gini(left: np.ndarray, right: np.ndarray) -> float:
    """
    Compute the weighted Gini impurity of a binary classification split.

    Args:
        left (np.ndarray): Class labels for the left child node.
        right (np.ndarray): Class labels for the right child node.

    Returns:
        float: Weighted Gini impurity score. Returns np.inf if the split is invalid.
    """
    n_left, n_right = len(left), len(right)
    n_total = n_left + n_right

    # Abort if the split is degenerate
    if n_total == 0:
        return np.inf

    def gini_side(labels: np.ndarray) -> float:
        """
        Compute Gini impurity for a single node.

        Args:
            labels (np.ndarray): Class labels for a single child node.

        Returns:
            float: Gini impurity for this node.
        """
        if len(labels) == 0:
            return 0.0
        counts = Counter(labels)
        probs = np.array(list(counts.values()), dtype=float) / len(labels)
        return 1.0 - np.sum(probs ** 2)

    weight_left = n_left / n_total
    weight_right = 1.0 - weight_left

    return weight_left * gini_side(left) + weight_right * gini_side(right)


def twoing(left: np.ndarray, right: np.ndarray) -> float:
    """
    Compute the Twoing criterion for binary classification splits.

    Args:
        left (np.ndarray): Class labels for the left child node.
        right (np.ndarray): Class labels for the right child node.

    Returns:
        float: Inverse of the Twoing value. Smaller values are better.
               Returns np.inf if computation is invalid or undefined.
    """
    n_left, n_right = len(left), len(right)
    n_total = n_left + n_right

    if n_total == 0:
        return np.inf

    # Get all unique class labels
    all_labels = np.concatenate([left, right])
    unique_classes = np.unique(all_labels)

    diff_sum = 0.0
    for cls in unique_classes:
        pL = np.sum(left == cls) / n_left if n_left > 0 else 0.0
        pR = np.sum(right == cls) / n_right if n_right > 0 else 0.0
        diff_sum += abs(pL - pR)

    twoing_val = (n_left / n_total) * (n_right / n_total) * (diff_sum ** 2) / 4.0

    return np.inf if twoing_val == 0.0 else 1.0 / twoing_val
