from C_oblique_decision_trees.evaluation.metrics import (
    compute_accuracy,
    compute_coverage,
    compute_density,
    compute_f_score,
    compute_leafwise_coverage_density,
    gini_coefficient,
    compute_average_active_feature_count,
    compute_feature_utilisation_ratio,
    compute_tree_level_sparsity_index,
    composite_interpretability_score
)


def evaluate_tree(tree, X, y):
    """
    Evaluate a decision tree model by computing performance, interpretability, and distribution metrics.

    Parameters:
        tree: A decision tree with attributes:
              'max_depth', 'num_splits', 'num_leaves', and a `predict` method.
        X (iterable): Input feature vectors.
        y (iterable): Ground-truth binary labels.

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    # Core performance metrics
    coverage = compute_coverage(tree, X, y)
    density = compute_density(tree, X, y)
    f_score = compute_f_score(tree, X, y)

    # Gini over all leaves
    cov_all, den_all = compute_leafwise_coverage_density(tree, X, y, filter_class=None)
    gini_cov_all = gini_coefficient(cov_all)
    gini_den_all = gini_coefficient(den_all)

    # Gini over class-1 predicting leaves only
    cov_1, den_1 = compute_leafwise_coverage_density(tree, X, y, filter_class=1)
    gini_cov_1 = gini_coefficient(cov_1)
    gini_den_1 = gini_coefficient(den_1)

    # Interpretability metrics
    avg_active = compute_average_active_feature_count(tree)
    util_ratio = compute_feature_utilisation_ratio(tree)
    sparsity = compute_tree_level_sparsity_index(tree)
    interpret_score = composite_interpretability_score(tree)

    return {
        "accuracy": compute_accuracy(tree, X, y),
        "coverage": coverage,
        "density": density,
        "f_score": f_score,
        "gini_coverage_all_leaves": gini_cov_all,
        "gini_density_all_leaves": gini_den_all,
        "gini_coverage_class_1_leaves": gini_cov_1,
        "gini_density_class_1_leaves": gini_den_1,
        "depth": getattr(tree, "max_depth", None),
        "splits": getattr(tree, "num_splits", None),
        "leaves": getattr(tree, "num_leaves", None),
        "avg_active_feature_count": avg_active,
        "feature_utilisation_ratio": util_ratio,
        "tree_level_sparsity_index": sparsity,
        "composite_interpretability_score": interpret_score
    }
