from C_oblique_decision_trees.evaluation.metrics import (
    compute_accuracy,
    compute_coverage,
    compute_density,
    compute_f_score,
    compute_leafwise_coverage_density,
    gini_coefficient,
    count_total_constrained_dimensions,
    compute_average_active_feature_count
)


def evaluate_tree(tree, X, y):
    """
    Evaluate a decision tree model by computing performance, interpretability, and
    scenario quality metrics relevant to scenario discovery.

    Parameters:
        tree: A decision tree with attributes:
              'max_depth', 'num_splits', 'num_leaves', and a `predict` method.
        X (iterable): Input feature vectors.
        y (iterable): Ground-truth binary labels.

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    # =========================================================================
    # CORE PERFORMANCE METRICS
    # =========================================================================
    accuracy = compute_accuracy(tree, X, y)
    coverage = compute_coverage(tree, X, y)
    density = compute_density(tree, X, y)
    f_score = compute_f_score(tree, X, y)

    # =========================================================================
    # SCENARIO QUALITY & PARSIMONY METRICS
    # =========================================================================
    # Gini metrics over coverage and density of all leaves predicting leaves
    cov_all, den_all = compute_leafwise_coverage_density(tree, X, y)

    gini_cov_all = gini_coefficient(cov_all)
    gini_den_all = gini_coefficient(den_all)

    # =========================================================================
    # INTERPRETABILITY & COMPLEXITY METRICS
    # =========================================================================
    total_active = count_total_constrained_dimensions(tree)
    avg_active = compute_average_active_feature_count(tree)

    # =========================================================================
    # STRUCTURAL METADATA
    # =========================================================================
    depth = getattr(tree, "max_depth", None)
    splits = getattr(tree, "num_splits", None)
    leaves = getattr(tree, "num_leaves", None)

    # =========================================================================
    # RETURN EVALUATION DICTIONARY
    # =========================================================================
    return {
        # Performance
        "accuracy": accuracy,
        "coverage": coverage,
        "density": density,
        "f_score": f_score,

        # Scenario quality & parsimony
        "gini_coverage_all_leaves": gini_cov_all,
        "gini_density_all_leaves": gini_den_all,

        # Interpretability
        "total_active_feature_count": total_active,
        "avg_active_feature_count": avg_active,

        # Metadata
        "depth": depth,
        "splits": splits,
        "leaves": leaves,
    }
