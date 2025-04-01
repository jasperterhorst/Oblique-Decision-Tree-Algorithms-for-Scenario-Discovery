"""
Evaluator module for decision trees.

Computes various performance and interpretability metrics for a given decision tree.
"""

from .metrics import (
    compute_accuracy,
    compute_coverage,
    compute_density,
    compute_f_score,
    compute_average_active_feature_count,
    compute_feature_utilisation_ratio,
    compute_tree_level_sparsity_index,
    composite_interpretability_score
)


def evaluate_tree(tree, X, y, training_time=None):
    """
    Evaluate a decision tree model by computing various performance and interpretability metrics,
    along with tree meta-data.

    Parameters:
        tree: A decision tree model instance. Expected to have attributes:
              'max_depth', 'num_splits', and 'num_leaves'.
        X (iterable): Input feature vectors.
        y (iterable): True labels corresponding to X.
        training_time (float, optional): Training duration in seconds.

    Returns:
        dict: A dictionary containing evaluation metrics:
            — “accuracy”: Model accuracy.
            — “coverage”: Fraction of relevant instances captured.
            — “density”: Purity of the selected region.
            — “f_score”: Harmonic mean of coverage and density.
            — “depth”: Maximum depth of the tree.
            — “splits”: Total number of splits in the tree.
            — “leaves”: Total number of leaves in the tree.
            — “avg_active_feature_count”: Average number of active features per node.
            — “feature_utilisation_ratio”: Average ratio of nonzero weights to total features per node.
            — “tree_level_sparsity_index”: Overall sparsity index of the tree.
            — “composite_interpretability_score”: A composite interpretability score.
            — “training_time”: Training duration in seconds.
    """
    metrics = {
        "accuracy": compute_accuracy(tree, X, y),
        "coverage": compute_coverage(tree, X, y),
        "density": compute_density(tree, X, y),
        "f_score": compute_f_score(tree, X, y),
        "depth": getattr(tree, "max_depth", None),
        "splits": getattr(tree, "num_splits", None),
        "leaves": getattr(tree, "num_leaves", None),
        "avg_active_feature_count": compute_average_active_feature_count(tree),
        "feature_utilisation_ratio": compute_feature_utilisation_ratio(tree),
        "tree_level_sparsity_index": compute_tree_level_sparsity_index(tree),
        "composite_interpretability_score": composite_interpretability_score(tree),
        "training_time": training_time
    }
    return metrics
