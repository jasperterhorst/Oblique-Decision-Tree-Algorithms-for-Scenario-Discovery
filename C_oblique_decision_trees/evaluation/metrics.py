import numpy as np
from sklearn.metrics import accuracy_score

from C_oblique_decision_trees.core.tree import DecisionNode


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


def compute_leafwise_coverage_density(tree, X, y, filter_class=None):
    """
    Compute per-leaf coverage and density.

    Parameters:
        tree: DecisionTree object.
        X: Feature matrix.
        y: True binary labels.
        filter_class (int or None): If set to 1 or 0, only include leaves predicting that class.

    Returns:
        Tuple[List[float], List[float]]: List of coverages and densities per leaf.
    """
    y = np.array(y)
    total_positives = np.sum(y == 1)
    leaf_assignments = {}

    for xi, yi in zip(X, y):
        node = tree.root
        while not node.is_leaf():
            if isinstance(node, DecisionNode):
                direction = node.decision(xi)
                node = node.children[direction]
            else:
                break
        if node.is_leaf():
            # Filter if specified
            if filter_class is not None and node.prediction != filter_class:
                continue
            if node.node_id not in leaf_assignments:
                leaf_assignments[node.node_id] = {"tp": 0, "fp": 0}
            if yi == 1:
                leaf_assignments[node.node_id]["tp"] += 1
            else:
                leaf_assignments[node.node_id]["fp"] += 1

    coverage_list = []
    density_list = []

    for stats in leaf_assignments.values():
        tp = stats["tp"]
        fp = stats["fp"]
        total = tp + fp
        coverage = tp / total_positives if total_positives > 0 else 0.0
        density = tp / total if total > 0 else 0.0
        coverage_list.append(coverage)
        density_list.append(density)

    return coverage_list, density_list


def gini_coefficient(values):
    """
    Compute the Gini coefficient of a list of non-negative values.
    Returns 0 if all values are 0, or the list is empty.
    """
    values = np.array(values)
    if len(values) == 0 or np.all(values == 0):
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1) / n


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
