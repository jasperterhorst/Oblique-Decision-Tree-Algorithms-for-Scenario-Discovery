# evaluation/metrics.py
import numpy as np
from sklearn.metrics import accuracy_score
from D_oblique_decision_trees.tree import DecisionNode


def compute_accuracy(tree, X, y):
    y_pred = [tree.predict(x) for x in X]
    return accuracy_score(y, y_pred)


def compute_coverage(tree, X, y):
    y_pred = np.array([tree.predict(x) for x in X])
    relevant = y == 1
    selected = y_pred == 1
    if relevant.sum() == 0:
        return 0.0
    return np.sum(relevant & selected) / np.sum(relevant)


def compute_density(tree, X, y):
    y_pred = np.array([tree.predict(x) for x in X])
    selected = y_pred == 1
    if selected.sum() == 0:
        return 0.0
    return np.sum((y == 1) & selected) / np.sum(selected)


def compute_f_score(tree, X, y):
    coverage = compute_coverage(tree, X, y)
    density = compute_density(tree, X, y)
    if coverage + density == 0:
        return 0.0
    return 2 * coverage * density / (coverage + density)


def interpretability_score(tree):
    """Average number of features used per decision node."""
    weights = [
        np.count_nonzero(node.weights)
        for node in tree.root.children if hasattr(node, 'weights')
    ]
    return np.mean(weights) if weights else 0


def sparsity(tree):
    """Proportion of zero weights across all splits."""
    total_weights = 0
    zero_weights = 0
    def count_weights(node):
        nonlocal total_weights, zero_weights
        if isinstance(node, DecisionNode):
            total_weights += node.weights.size
            zero_weights += np.sum(node.weights == 0)
    tree.traverse(count_weights)
    return zero_weights / total_weights if total_weights > 0 else 0


def compute_convergence_length(log):
    """If convergence logs are recorded per split, return average length."""
    return np.mean([len(trace) for trace in log.values()]) if log else None


def compute_training_time(start_time, end_time):
    """Simple wrapper to get duration in seconds."""
    return end_time - start_time


def seed_stability(scores):
    """Given a list of accuracy scores across seeds, return std deviation."""
    return np.std(scores)
