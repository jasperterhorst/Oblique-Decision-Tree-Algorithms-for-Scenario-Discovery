"""
Sparse_HouseHolder_CART_D.py
=============================

This module implements a sparse variant of HHCART(D), where Sparse PCA is used to compute
sparse dominant eigenvectors for Householder reflections, improving interpretability and robustness
in high-dimensional or noisy settings.
"""

import numpy as np
from copy import deepcopy
from scipy.linalg import norm
from sklearn.decomposition import SparsePCA
from sklearn.base import BaseEstimator, ClassifierMixin
from _adopted_oblique_trees.segmentor import CARTSegmentor


class SparseHHCARTNode:
    def __init__(self, depth, labels, **kwargs):
        self.depth = depth
        self.labels = labels
        self.is_leaf = kwargs.get('is_leaf', False)
        self._split_rules = kwargs.get('split_rules', None)
        self._weights = kwargs.get('weights', None)
        self._left_child = kwargs.get('left_child', None)
        self._right_child = kwargs.get('right_child', None)
        self._label = None

        if not self.is_leaf:
            assert self._split_rules is not None
            assert self._left_child is not None
            assert self._right_child is not None

    def get_child(self, datum):
        if self.is_leaf:
            raise Warning("Leaf node does not have children.")
        X = deepcopy(datum)
        if X.dot(np.array(self._weights[:-1]).T) - self._weights[-1] < 0:
            return self.left_child
        else:
            return self.right_child

    @property
    def label(self):
        if self._label is None:
            classes, counts = np.unique(self.labels, return_counts=True)
            self._label = classes[np.argmax(counts)]
        return self._label

    @property
    def split_rules(self):
        if self.is_leaf:
            raise Warning("Leaf node does not have split rule.")
        return self._split_rules

    @property
    def left_child(self):
        if self.is_leaf:
            raise Warning("Leaf node does not have a left child.")
        return self._left_child

    @property
    def right_child(self):
        if self.is_leaf:
            raise Warning("Leaf node does not have a right child.")
        return self._right_child


class SparseHouseHolderCARTD(BaseEstimator):
    def __init__(self, impurity, segmentor=CARTSegmentor(), max_depth=50, min_samples_split=2,
                 alpha=1.0, tau=1e-4, random_state=None, **kwargs):
        self.impurity = impurity
        self.segmentor = segmentor
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.alpha = alpha
        self.tau = tau
        self.random_state = random_state
        self._root = None
        self._nodes = []
        self.debug = kwargs.get("debug", False)

    def fit(self, X, y):
        self._root = self._generate_node(X, y, 0)
        return self

    def _terminate(self, X, y, cur_depth):
        return (self.max_depth is not None and cur_depth == self.max_depth) or \
               (y.size < self.min_samples_split) or \
               (np.unique(y).size == 1)

    def _generate_leaf_node(self, cur_depth, y):
        node = SparseHHCARTNode(cur_depth, y, is_leaf=True)
        self._nodes.append(node)
        return node

    def _generate_node(self, X, y, cur_depth):
        if self._terminate(X, y, cur_depth):
            return self._generate_leaf_node(cur_depth, y)

        n_objects, n_features = X.shape
        best_H = np.eye(n_features)
        impurity_best = float('inf')
        sr = None

        classes = np.unique(y)
        for c in classes:
            X_c = X[y == c]
            if X_c.shape[0] <= 1:
                continue

            spca = SparsePCA(n_components=1, alpha=self.alpha, random_state=self.random_state)
            spca.fit(X_c)
            mu_sparse = spca.components_[0]

            if np.allclose(mu_sparse, 0):
                continue

            check_ = np.sqrt(((np.eye(n_features) - mu_sparse) ** 2).sum(axis=1))
            if (check_ > self.tau).sum() > 0:
                i = np.argmax(check_)
                e = np.zeros(n_features)
                e[i] = 1.0
                w = (e - mu_sparse) / norm(e - mu_sparse)
                H = np.eye(n_features) - 2 * np.outer(w, w)
                X_reflected = X @ H

                impurity_ref, sr_ref, _, _ = self.segmentor(X_reflected, y, self.impurity)
                if impurity_ref < impurity_best:
                    impurity_best = impurity_ref
                    sr = sr_ref
                    best_H = H

        if not sr:
            return self._generate_leaf_node(cur_depth, y)

        i, threshold = sr
        weights = np.zeros(n_features + 1)
        weights[:-1] = best_H[:, i]
        weights[-1] = threshold

        left_indices = X.dot(weights[:-1]) - weights[-1] < 0
        right_indices = ~left_indices
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]

        if len(y_right) <= self.min_samples_split or len(y_left) <= self.min_samples_split:
            return self._generate_leaf_node(cur_depth, y)

        node = SparseHHCARTNode(
            cur_depth, y,
            split_rules=sr,
            weights=weights,
            left_child=self._generate_node(X_left, y_left, cur_depth + 1),
            right_child=self._generate_node(X_right, y_right, cur_depth + 1),
            is_leaf=False
        )
        self._nodes.append(node)
        return node

    def predict(self, X):
        def predict_single(datum):
            cur_node = self._root
            while not cur_node.is_leaf:
                cur_node = cur_node.get_child(datum)
            return cur_node.label

        if not self._root:
            raise Warning("Decision tree has not been trained.")
        size = X.shape[0]
        predictions = np.empty((size,), dtype=int)
        for i in range(size):
            predictions[i] = predict_single(X[i, :])
        return predictions

    def score(self, data, labels):
        predictions = self.predict(data)
        correct_count = np.count_nonzero(predictions == labels)
        return correct_count / labels.shape[0]


class SparseHHCARTDClassifier(ClassifierMixin, SparseHouseHolderCARTD):
    def __init__(self, impurity, segmentor=CARTSegmentor(), max_depth=50, min_samples_split=2,
                 alpha=1.0, tau=1e-4, random_state=None, **kwargs):
        super().__init__(impurity=impurity, segmentor=segmentor, max_depth=max_depth,
                         min_samples_split=min_samples_split, alpha=alpha, tau=tau,
                         random_state=random_state, **kwargs)
