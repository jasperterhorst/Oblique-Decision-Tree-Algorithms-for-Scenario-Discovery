import numpy as np
from sklearn.metrics import accuracy_score


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================
def compute_accuracy(tree, X, y):
    """
    Compute the accuracy of a model on a given dataset.

    Parameters:
        tree: A trained model with a `predict` method.
        X (iterable): An iterable of input feature vectors.
        y (iterable): An iterable of true labels.

    Returns:
        float: The accuracy score as a fraction.
    """
    y_pred = [tree.predict(x) for x in X]
    return accuracy_score(y, y_pred)


def compute_coverage(tree, X, y):
    """
    Compute the coverage metric. If the tree has depth 0, return 1.0.
    Otherwise, compute the proportion of relevant (y == 1) instances captured.

    Parameters:
        tree: A trained decision tree with `predict` and `depth` attributes.
        X (iterable): Input features.
        y (iterable): True binary labels.

    Returns:
        float: Coverage score.
    """
    y = np.array(y)
    if tree.max_depth == 0:
        return 1.0

    y_pred = np.array([tree.predict(x) for x in X])
    relevant = (y == 1)
    selected = (y_pred == 1)

    total_relevant = np.sum(relevant)
    if total_relevant == 0:
        return np.nan

    return np.sum(relevant & selected) / total_relevant


def compute_density(tree, X, y):
    """
    Compute the density metric. If tree has depth 0, return the average of y.
    Otherwise, compute the proportion of selected instances that are relevant.

    Parameters:
        tree: A trained decision tree with `predict` and `depth` attributes.
        X (iterable): Input features.
        y (iterable): True binary labels.

    Returns:
        float: Density score.
    """
    y = np.array(y)
    if tree.max_depth == 0:
        return np.mean(y)

    y_pred = np.array([tree.predict(x) for x in X])
    selected = (y_pred == 1)

    total_selected = np.sum(selected)
    if total_selected == 0:
        return 0.0

    return np.sum((y == 1) & selected) / total_selected


# def compute_coverage(tree, X, y):
#     """
#     Compute the coverage metric as the proportion of relevant instances (where y == 1)
#     that are correctly captured by the model.
#
#     Parameters:
#         tree: A trained model with a `predict` method.
#         X (iterable): An iterable of input feature vectors.
#         y (iterable): An iterable of true labels.
#
#     Returns:
#         float: The coverage metric as a fraction, or np.nan if no relevant instances exist.
#     """
#     y = np.array(y)
#     y_pred = np.array([tree.predict(x) for x in X])
#     relevant = (y == 1)
#     selected = (y_pred == 1)
#     total_relevant = np.sum(relevant)
#     if total_relevant == 0:
#         return np.nan
#     return np.sum(relevant & selected) / total_relevant
#
#
# def compute_density(tree, X, y):
#     """
#     Compute the density metric as the proportion of instances within the selected region
#     (where the model predicts 1) that are actually of interest (where y == 1).
#
#     Parameters:
#         tree: A trained model with a `predict` method.
#         X (iterable): An iterable of input feature vectors.
#         y (iterable): An iterable of true binary labels.
#
#     Returns:
#         float: The density metric as a fraction, or np.nan if no instances are selected.
#     """
#     y = np.array(y)
#     y_pred = np.array([tree.predict(x) for x in X])
#     selected = (y_pred == 1)
#     total_selected = np.sum(selected)
#     if total_selected == 0:
#         return np.nan
#     return np.sum((y == 1) & selected) / total_selected


def compute_f_score(tree, X, y):
    """
    Compute the F metric as the harmonic mean of density and coverage.

    In the context of scenario discovery, this is defined as:
        F = 2 * (density * coverage) / (density + coverage)

    Parameters:
        tree: A trained model with a `predict` method.
        X (iterable): Input feature vectors.
        y (iterable): True binary labels.

    Returns:
        float: The F metric value, or np.nan if either coverage or density is undefined.
    """
    coverage = compute_coverage(tree, X, y)
    density = compute_density(tree, X, y)
    if np.isnan(coverage) or np.isnan(density):
        return np.nan
    if coverage + density == 0:
        return 0.0
    return 2 * coverage * density / (coverage + density)


# =============================================================================
# INTERPRETABILITY & COMPLEXITY METRICS
# =============================================================================
def compute_average_active_feature_count(tree):
    """
    Compute the average active feature count per decision node.

    Each node (accessed via tree.root.children) is expected to have a 'weights' attribute.

    Returns:
        float: Average number of nonzero weights per node, or 0.0 if no weights are found.
    """
    active_feature_counts = [
        np.count_nonzero(node.weights)
        for node in tree.root.children if hasattr(node, 'weights')
    ]
    return np.mean(active_feature_counts) if active_feature_counts else 0.0


def compute_feature_utilisation_ratio(tree):
    """
    Compute the average feature utilization ratio per decision node.

    For each node with a 'weights' attribute, this is the ratio of nonzero weights to total features.

    Returns:
        float: The average utilization ratio, or 0.0 if no nodes with weights are found.
    """
    ratios = []
    for node in tree.root.children:
        if hasattr(node, 'weights'):
            weights = node.weights
            total_features = len(weights)
            nonzero_count = np.count_nonzero(weights)
            ratio = nonzero_count / total_features if total_features > 0 else 0
            ratios.append(ratio)
    return np.mean(ratios) if ratios else 0.0


def compute_tree_level_sparsity_index(tree):
    """
    Compute the overall tree-level sparsity index.

    Defined as:
        Sparsity = 1 – (Total nonzero weights / Total weights)

    Returns:
        float: The sparsity index, or np.nan if no weights are found.
    """
    total_nonzero = 0
    total_weights = 0
    for node in tree.root.children:
        if hasattr(node, 'weights'):
            weights = node.weights
            total_nonzero += np.count_nonzero(weights)
            total_weights += len(weights)
    if total_weights == 0:
        return np.nan
    return 1 - (total_nonzero / total_weights)


def composite_interpretability_score(tree):
    """
    Compute a composite interpretability score for the tree.

    This score integrates:
      – Average active feature count,
      – Feature utilization ratio, and
      – Tree-level sparsity index.

    Lower values are assumed to indicate higher interpretability.

    Returns:
         float: The interpretability score, or np.nan if sparsity is zero.
    """
    active_count = compute_average_active_feature_count(tree)
    util_ratio = compute_feature_utilisation_ratio(tree)
    sparsity_index = compute_tree_level_sparsity_index(tree)
    if sparsity_index == 0 or np.isnan(sparsity_index):
        return np.nan
    return (active_count * (1 - util_ratio)) / sparsity_index
