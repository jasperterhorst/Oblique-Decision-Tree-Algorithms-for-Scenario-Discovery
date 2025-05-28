import numpy as np
from sklearn.metrics import accuracy_score

from HHCART.tree import DecisionNode


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
    y_pred = np.array([tree.predict(x) for x in X]).astype(int)
    return accuracy_score(y, y_pred)


def compute_coverage(tree, X, y):
    """
    Compute the coverage metric as the proportion of relevant instances (where y == 1)
    that are correctly captured by the model.

    Parameters:
        tree: A trained model with a `predict` method.
        X (iterable): An iterable of input feature vectors.
        y (iterable): An iterable of true labels.

    Returns:
        float: The coverage metric as a fraction, or np.nan if no relevant instances exist.
    """
    y = np.array(y)
    y_pred = np.array([tree.predict(x) for x in X])
    relevant = (y == 1)
    selected = (y_pred == 1)
    total_relevant = np.sum(relevant)
    if total_relevant == 0:
        return np.nan
    return np.sum(relevant & selected) / total_relevant


def compute_density(tree, X, y):
    """
    Compute the density metric as the proportion of instances within the selected region
    (where the model predicts 1) that are actually of interest (where y == 1).

    Parameters:
        tree: A trained model with a `predict` method.
        X (iterable): An iterable of input feature vectors.
        y (iterable): An iterable of true binary labels.

    Returns:
        float: The density metric as a fraction, or np.nan if no instances are selected.
    """
    y = np.array(y)
    y_pred = np.array([tree.predict(x) for x in X])
    selected = (y_pred == 1)
    total_selected = np.sum(selected)
    if total_selected == 0:
        return np.nan
    return np.sum((y == 1) & selected) / total_selected


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
def count_total_constrained_dimensions(tree):
    """
    Count the number of unique features (dimensions) used across all splits in the decision tree.

    Parameters:
        tree (DecisionTree): The decision tree object.

    Returns:
        int: Number of distinct non-zero feature indices used in decision nodes.
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
