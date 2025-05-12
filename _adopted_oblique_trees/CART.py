"""
CARTClassifier.py
==================

This module implements an axis-aligned Classification and Regression Tree (CART) that integrates into the
unified oblique decision tree evaluation and visualization framework. It mirrors the structure of other models
like HHCartAClassifier and ObliqueClassifier1.

Key components:
- `CARTNode`: Represents internal or leaf nodes of the tree
- `CART`: Core implementation including recursive fitting and prediction
- `CARTClassifier`: Scikit-learn compatible wrapper for use in benchmarking

Supports:
- Configurable `impurity` measure (e.g., "gini", "entropy")
- Maximum tree depth
- Minimum samples per split

This version aligns with the project’s model registration and conversion logic (e.g., for visualization).
"""

import numpy as np
from _adopted_oblique_trees.segmentor import CARTSegmentor
from sklearn.base import ClassifierMixin


# =============================================================================
# Tree Node Structure
# =============================================================================
class CARTNode:
    def __init__(self, depth, labels, **kwargs):
        self.depth = depth
        self.labels = labels
        self.is_leaf = kwargs.get('is_leaf', False)
        self._split_rules = kwargs.get('split_rules', None)
        self._weights = kwargs.get('weights', None)
        self._left_child = kwargs.get('left_child', None)
        self._right_child = kwargs.get('right_child', None)

    def get_child(self, datum):
        if self.is_leaf:
            raise Warning("Leaf node has no children.")
        return self._left_child if datum.dot(self._weights[:-1]) - self._weights[-1] < 0 else self._right_child

    @property
    def label(self):
        if not hasattr(self, '_label'):
            classes, counts = np.unique(self.labels, return_counts=True)
            self._label = classes[np.argmax(counts)]
        return self._label

    @property
    def left_child(self):
        return self._left_child

    @property
    def right_child(self):
        return self._right_child


# =============================================================================
# Base CART Model (Axis-Aligned)
# =============================================================================
class CART:
    def __init__(self, impurity, max_depth=10, min_samples_split=2):
        self.impurity = impurity
        self.segmentor = CARTSegmentor()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self._root = None
        self._nodes = []

    def _terminate(self, y, depth):
        return (
            depth == self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        )

    def _generate_leaf_node(self, depth, y):
        node = CARTNode(depth, y, is_leaf=True)
        self._nodes.append(node)
        return node

    def _generate_node(self, X, y, depth):
        if self._terminate(y, depth):
            return self._generate_leaf_node(depth, y)

        impurity, rule, left_idx, right_idx = self.segmentor(X, y, self.impurity)
        if rule is None:
            return self._generate_leaf_node(depth, y)

        i, thr = rule
        weights = np.zeros(X.shape[1] + 1)
        weights[i] = 1.0
        weights[-1] = thr

        node = CARTNode(
            depth, y,
            split_rules=rule,
            weights=weights,
            left_child=self._generate_node(X[left_idx], y[left_idx], depth + 1),
            right_child=self._generate_node(X[right_idx], y[right_idx], depth + 1),
            is_leaf=False
        )
        self._nodes.append(node)
        return node

    def fit(self, X, y):
        self._root = self._generate_node(X, y, 0)

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        node = self._root
        while not node.is_leaf:
            node = node.get_child(x)
        return node.label

    @property
    def root(self):
        return self._root

    @property
    def nodes(self):
        return self._nodes

    @property
    def num_leaves(self):
        return sum(n.is_leaf for n in self._nodes)

    @property
    def num_splits(self):
        return sum(not n.is_leaf for n in self._nodes)


# =============================================================================
# Scikit-learn Compatible Wrapper
# =============================================================================
class CARTClassifier(ClassifierMixin, CART):
    def __init__(self, impurity, max_depth=10, min_samples_split=2, random_state=None):
        super().__init__(impurity=impurity, max_depth=max_depth, min_samples_split=min_samples_split)
