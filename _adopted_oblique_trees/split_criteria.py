# split_criteria.py
import numpy as np
from collections import Counter


def gini(left, right):
    """
    Compute the Gini impurity of a binary split.
    left, right: arrays of class labels in the two child nodes.
    """

    n_left, n_right = len(left), len(right)
    n_total = n_left + n_right
    if n_total == 0:
        return np.inf
    p_left = n_left / n_total

    # Gini for one side: 1 - sum(p_k^2) over classes
    def gini_side(labels):
        if len(labels) == 0:
            return 0.0
        counts = Counter(labels)
        probs = np.array(list(counts.values()), dtype=float) / len(labels)
        return 1.0 - np.sum(probs ** 2)

    return p_left * gini_side(left) + (1 - p_left) * gini_side(right)


def twoing(left, right):
    """
    Compute the Twoing rule impurity (Murthy et al.) for a split.
    Returns 1 / (Twoing value) so that smaller splits yield lower impurity.
    """

    n_left, n_right = len(left), len(right)
    n_total = n_left + n_right
    if n_total == 0:
        return np.inf
    labels = np.concatenate([left, right])
    classes = np.unique(labels)
    # compute sum |p_L(k) - p_R(k)|
    diff_sum = 0.0
    for k in classes:
        pL = np.sum(left == k) / n_left if n_left > 0 else 0.0
        pR = np.sum(right == k) / n_right if n_right > 0 else 0.0
        diff_sum += abs(pL - pR)
    twoing_val = (n_left / n_total) * (n_right / n_total) * (diff_sum ** 2) / 4.0

    return np.inf if twoing_val == 0 else 1.0 / twoing_val


def mse(left, right):
    """
    Mean squared error impurity for regression splits.
    """

    n_left, n_right = len(left), len(right)
    n_total = n_left + n_right
    if n_total == 0:
        return np.inf
    left_var = np.var(left) if n_left > 0 else 0.0
    right_var = np.var(right) if n_right > 0 else 0.0

    return (n_left / n_total) * left_var + (n_right / n_total) * right_var
