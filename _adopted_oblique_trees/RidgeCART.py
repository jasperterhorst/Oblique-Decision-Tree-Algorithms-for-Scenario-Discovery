"""
RidgeCART.py
=============

This module implements the RidgeCART oblique decision tree algorithm and its Scikit-learn-compatible wrapper
`RidgeCARTClassifier`, used for benchmarking and visualization within the unified decision tree framework.

RidgeCART uses Ridge Regression to define projection directions and selects splits based on impurity measures.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.base import ClassifierMixin, BaseEstimator
from _adopted_oblique_trees.segmentor import CARTSegmentor


class RidgeCARTNode:
    def __init__(self, depth, labels, **kwargs):
        self.depth = depth
        self.labels = labels
        self.is_leaf = kwargs.get("is_leaf", False)
        self._split_rules = kwargs.get("split_rules", None)
        self._weights = kwargs.get("weights", None)
        self._left_child = kwargs.get("left_child", None)
        self._right_child = kwargs.get("right_child", None)

    def get_child(self, datum):
        if self.is_leaf:
            raise ValueError("Leaf node has no children.")
        return self._left_child if datum.dot(self._weights[:-1]) - self._weights[-1] < 0 else self._right_child

    @property
    def label(self):
        if not hasattr(self, "_label"):
            classes, counts = np.unique(self.labels, return_counts=True)
            self._label = classes[np.argmax(counts)]
        return self._label

    @property
    def left_child(self):
        return self._left_child

    @property
    def right_child(self):
        return self._right_child


class Ridge_CART:
    def __init__(self, impurity, segmentor, alpha=1.0, method="usual",
                 max_depth=10, min_samples=2, random_state=None):
        self.impurity = impurity
        self.segmentor = segmentor
        self.alpha = alpha
        self.method = method
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.random_state = random_state
        self._root = None
        self._nodes = []
        self.rng = np.random.RandomState(random_state)

    def _terminate(self, y, depth):
        return (depth == self.max_depth or len(y) < self.min_samples or len(np.unique(y)) == 1)

    def _generate_leaf_node(self, depth, y):
        node = RidgeCARTNode(depth, y, is_leaf=True)
        self._nodes.append(node)
        return node

    def _generate_node(self, X, y, depth):
        if self._terminate(y, depth):
            return self._generate_leaf_node(depth, y)

        ridge = Ridge(alpha=self.alpha)
        ridge.fit(X, y)
        projection = X @ ridge.coef_.T

        if self.method == "usual":
            X_proj = np.hstack([X, projection.reshape(-1, 1)])
        else:
            X_proj = projection.reshape(-1, 1)

        impurity, rule, left_idx, right_idx = self.segmentor(X_proj, y, self.impurity)
        if rule is None:
            return self._generate_leaf_node(depth, y)

        i, thr = rule

        # This is for DecisionTree compatibility
        weights = np.zeros(self.input_dim + 1)

        if self.method == "usual":
            if i == X.shape[1]:  # projection feature
                weights[:-1] = ridge.coef_
            else:
                weights[i] = 1.0
        else:
            weights[:-1] = ridge.coef_

        weights[-1] = thr

        # This is for internal split logic based on projected space
        split_weights = np.zeros(X_proj.shape[1])
        split_weights[i] = 1.0

        left_mask = X_proj.dot(split_weights) - thr < 0
        right_mask = ~left_mask

        left_child = self._generate_node(X[left_mask], y[left_mask], depth + 1)
        right_child = self._generate_node(X[right_mask], y[right_mask], depth + 1)

        node = RidgeCARTNode(depth, y, split_rules=rule, weights=weights,
                              left_child=left_child, right_child=right_child, is_leaf=False)
        self._nodes.append(node)
        return node

    def fit(self, X, y):
        self.input_dim = X.shape[1]  # Track full input dimension for proper weight vector shape
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


class RidgeCARTClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, impurity, segmentor, alpha=1.0, method="usual",
                 max_depth=10, min_samples_split=2, random_state=None):
        self.impurity = impurity
        self.segmentor = segmentor
        self.alpha = alpha
        self.method = method
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

    def fit(self, X, y):
        self.model_ = Ridge_CART(
            impurity=self.impurity,
            segmentor=self.segmentor,
            alpha=self.alpha,
            method=self.method,
            max_depth=self.max_depth,
            min_samples=self.min_samples_split,
            random_state=self.random_state
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    @property
    def root(self):
        return getattr(self.model_, '_root', None)

    @property
    def nodes(self):
        return getattr(self.model_, '_nodes', [])

    @property
    def num_splits(self):
        return sum(not n.is_leaf for n in self.nodes)

    @property
    def num_leaves(self):
        return sum(n.is_leaf for n in self.nodes)

