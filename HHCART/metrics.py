"""
Tree Evaluation Metrics (metrics.py)
------------------------------------
Provides performance and interpretability metrics for scenario discovery trees.

Performance metrics:
- Accuracy
- Coverage
- Density
- F-score (harmonic mean of coverage and density)

Interpretability metrics:
- Total number of active features across all splits
- Average number of active features per internal node
"""

import numpy as np
from typing import Union, Sequence
from sklearn.metrics import accuracy_score

from HHCART.tree import DecisionNode, DecisionTree


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

def compute_accuracy(
    tree: DecisionTree,
    X: np.ndarray,
    y: Union[np.ndarray, Sequence[int]]
) -> float:
    """
    Compute standard classification accuracy for the decision tree.

    Args:
        tree (DecisionTree): Fitted tree with `.predict(x)` method.
        X (np.ndarray): Feature matrix (samples × features).
        y (np.ndarray or list of int): True binary labels.

    Returns:
        float: Fraction of correctly predicted labels.
    """
    y_pred = np.array([tree.predict(x) for x in X])
    return accuracy_score(y, y_pred)


def compute_coverage(
    tree: DecisionTree,
    X: np.ndarray,
    y: Union[np.ndarray, Sequence[int]]
) -> float:
    """
    Compute the coverage metric as the proportion of relevant (label=1) samples
    that are correctly captured in the selected region (prediction=1).

    Args:
        tree (DecisionTree): Fitted tree with `.predict(x)` method.
        X (np.ndarray): Feature matrix.
        y (np.ndarray or list of int): True binary labels.

    Returns:
        float: Coverage score, or np.nan if no positive labels exist.
    """
    y = np.asarray(y)
    y_pred = np.array([tree.predict(x) for x in X])

    relevant = (y == 1)
    selected = (y_pred == 1)
    total_relevant = np.sum(relevant)

    if total_relevant == 0:
        return 0.0

    return np.sum(relevant & selected) / total_relevant


def compute_density(
    tree: DecisionTree,
    X: np.ndarray,
    y: Union[np.ndarray, Sequence[int]]
) -> float:
    """
    Compute the density metric as the proportion of selected points (prediction=1)
    that are relevant (label=1).

    Args:
        tree (DecisionTree): Fitted tree with `.predict(x)` method.
        X (np.ndarray): Feature matrix.
        y (np.ndarray or list of int): True binary labels.

    Returns:
        float: Density score, or np.nan if no points are selected.
    """
    y = np.asarray(y)
    y_pred = np.array([tree.predict(x) for x in X])

    selected = (y_pred == 1)
    total_selected = np.sum(selected)

    if total_selected == 0:
        return 0.0

    return np.sum((y == 1) & selected) / total_selected


def compute_f_score(
    tree: DecisionTree,
    X: np.ndarray,
    y: Union[np.ndarray, Sequence[int]]
) -> float:
    """
    Compute the harmonic mean of coverage and density (F-score).

    Args:
        tree (DecisionTree): Fitted tree with `.predict(x)` method.
        X (np.ndarray): Feature matrix.
        y (np.ndarray or list of int): True binary labels.

    Returns:
        float: F-score, or np.nan if either component is undefined.
    """
    coverage = compute_coverage(tree, X, y)
    density = compute_density(tree, X, y)

    if np.isnan(coverage) or np.isnan(density):
        return np.nan
    if coverage + density == 0:
        return 0.0

    return 2 * coverage * density / (coverage + density)


# =============================================================================
# INTERPRETABILITY METRICS
# =============================================================================

def count_total_constrained_dimensions(
    tree: DecisionTree
) -> int:
    """
    Count the number of unique input features (non-zero weights)
    used across all internal decision nodes.

    Args:
        tree (DecisionTree): Oblique tree with `.root` node.

    Returns:
        int: Total number of distinct features used in split decisions.
    """
    constrained_dims = set()
    nodes_to_visit = [tree.root]

    while nodes_to_visit:
        node = nodes_to_visit.pop()
        if isinstance(node, DecisionNode) and node.weights.size > 0:
            used_dims = np.flatnonzero(node.weights)
            constrained_dims.update(used_dims)

        nodes_to_visit.extend(node.children)

    return len(constrained_dims)


def compute_average_active_feature_count(
    tree: DecisionTree
) -> float:
    """
    Compute the average number of active features (non-zero weights)
    per internal decision node.

    Args:
        tree (DecisionTree): Oblique tree with `.root` node.

    Returns:
        float: Average number of active features per internal node.
    """
    counts = []

    for node in tree.root.children:
        if hasattr(node, "weights"):
            count = np.count_nonzero(node.weights)
            counts.append(count)

    return np.mean(counts) if counts else 0.0
