"""
Tree Evaluation Metrics (evaluation.py)
---------------------------------------
Computes performance, interpretability, and structural metrics
for a trained decision tree, using standard scenario discovery criteria.

Supported metrics include:
- Accuracy, Coverage, Density, F-score
- Total and average number of active features per split
- Depth, number of splits, number of leaves
"""

from typing import Union, Sequence, Any
import numpy as np

from HHCART.metrics import (
    compute_accuracy,
    compute_coverage,
    compute_density,
    compute_f_score,
    count_total_constrained_dimensions,
    compute_average_active_feature_count
)


def evaluate_tree(
    tree: Any,
    X: np.ndarray,
    y: Union[np.ndarray, Sequence[int]]
) -> dict:
    """
    Evaluate a decision tree using classification, interpretability, and structure metrics.

    This function is designed for scenario discovery and model comparison. It combines
    performance scores (accuracy, coverage, density, F-score), interpretability scores
    (active feature usage), and structural metadata (depth, split count, leaf count).

    Parameters:
        tree (object): Trained decision tree with:
                       - `predict(x)` method for classifying samples
                       - `max_depth`, `num_splits`, and `num_leaves` attributes
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray or list[int]): True binary labels of shape (n_samples,).

    Returns:
        dict: A dictionary with the following keys:
            - "accuracy": float – Classification accuracy
            - "coverage": float – Coverage of relevant points
            - "density": float – Density of selected region
            - "f_score": float – Harmonic mean of coverage and density
            - "total_active_feature_count": int – Unique features used in splits
            - "avg_active_feature_count": float – Mean features per split
            - "depth": int or None – Maximum tree depth
            - "splits": int or None – Number of internal decision nodes
            - "leaves": int or None – Number of terminal leaf nodes

    Raises:
        ValueError: If tree lacks a `predict()` method or if inputs are malformed.
    """
    # -------------------------------------------------------------------------
    # Type safety and interface sanity checks
    # -------------------------------------------------------------------------
    if not callable(getattr(tree, "predict", None)):
        raise ValueError("[❌] Provided tree object lacks a callable `.predict(x)` method.")
    if not hasattr(X, "__len__") or not hasattr(y, "__len__"):
        raise ValueError("[❌] `X` and `y` must be iterable and aligned.")

    # -------------------------------------------------------------------------
    # Performance metrics
    # -------------------------------------------------------------------------
    accuracy = compute_accuracy(tree, X, y)
    coverage = compute_coverage(tree, X, y)
    density = compute_density(tree, X, y)
    f_score = compute_f_score(tree, X, y)

    # -------------------------------------------------------------------------
    # Interpretability metrics
    # -------------------------------------------------------------------------
    total_active_features = count_total_constrained_dimensions(tree)
    avg_active_features = compute_average_active_feature_count(tree)

    # -------------------------------------------------------------------------
    # Structural attributes (if present on the tree object)
    # -------------------------------------------------------------------------
    max_depth = getattr(tree, "max_depth", None)
    num_splits = getattr(tree, "num_splits", None)
    num_leaves = getattr(tree, "num_leaves", None)

    # -------------------------------------------------------------------------
    # Compile final results
    # -------------------------------------------------------------------------
    return {
        # Performance
        "accuracy": accuracy,
        "coverage": coverage,
        "density": density,
        "f_score": f_score,

        # Interpretability
        "total_active_feature_count": total_active_features,
        "avg_active_feature_count": avg_active_features,

        # Structural metadata
        "depth": max_depth,
        "splits": num_splits,
        "leaves": num_leaves,
    }
