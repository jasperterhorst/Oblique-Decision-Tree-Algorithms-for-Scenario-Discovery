"""
Sparse_OC1.py
=============

This module implements Sparse OC1: an enhanced version of Oblique Classifier 1 (OC1) with sparsity controls.
It introduces:
  - L1 regularisation during split search.
  - Hard thresholding to enforce zero coefficients.
  - User-defined parameters for sparsity strength.
"""

import numpy as np
from scipy.stats import mode
from _adopted_oblique_trees.OC1_tree_structure import Tree, Node, LeafNode
from _adopted_oblique_trees import split_criteria
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier


class BaseSparseObliqueTree(BaseEstimator):

    def __init__(self, impurity='gini', max_depth=3, min_samples_split=2, min_features_split=1,
                 random_state=None, n_restarts=20, bias_steps=20,
                 lambda_reg=0.01, threshold_value=0.01):

        self.impurity = impurity
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_features_split = min_features_split
        self.tree_ = None
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state) if random_state is not None else np.random
        self.n_restarts = n_restarts
        self.bias_steps = bias_steps
        self.lambda_reg = lambda_reg
        self.threshold_value = threshold_value

        self.root_node = None
        self.learned_depth = None

    def fit(self, X, y):
        if X.ndim == 1:
            X = X.reshape(-1, 1) if len(y) > 1 else X.reshape(1, -1)

        if self.impurity == 'gini':
            self.impurity = split_criteria.gini
        elif self.impurity == 'twoing':
            self.impurity = split_criteria.twoing
        else:
            raise ValueError('Unrecognized split impurity specified.')

        n_samples, n_features = X.shape

        min_samples_split = max(2, self.min_samples_split)
        min_features_split = max(1, self.min_features_split)

        self.root_node, self.learned_depth = build_sparse_oblique_tree_oc1(
            X, y, is_classifier(self), self.impurity, self.max_depth, min_samples_split,
            min_features_split, rng=self.rng, n_restarts=self.n_restarts, bias_steps=self.bias_steps,
            lambda_reg=self.lambda_reg, threshold_value=self.threshold_value
        )

        self.tree_ = Tree(n_features=n_features, is_classifier=is_classifier(self))
        self.tree_.set_root_node(self.root_node)
        self.tree_.set_depth(self.learned_depth)

    def predict(self, X):
        return self.tree_.root_node.predict(X).astype(int)


class SparseObliqueClassifier1(ClassifierMixin, BaseSparseObliqueTree):
    def __init__(self, impurity="gini", max_depth=3, min_samples_split=2, min_features_split=1,
                 random_state=None, n_restarts=20, bias_steps=20, lambda_reg=0.01, threshold_value=0.01):
        super().__init__(impurity=impurity, max_depth=max_depth, min_samples_split=min_samples_split,
                         min_features_split=min_features_split, random_state=random_state,
                         n_restarts=n_restarts, bias_steps=bias_steps,
                         lambda_reg=lambda_reg, threshold_value=threshold_value)


def build_sparse_oblique_tree_oc1(X, y, is_classification, impurity, max_depth, min_samples_split,
                                  min_features_split, rng, n_restarts, bias_steps, lambda_reg, threshold_value,
                                  current_depth=0, current_features=None, current_samples=None):

    n_samples, n_features = X.shape

    if current_depth == 0:
        current_features = np.arange(n_features)
        current_samples = np.arange(n_samples)

    y = np.atleast_1d(y)

    if is_classification:
        if y.size == 1:
            label, conf = y[0], 1.0
        else:
            majority, count = mode(y, keepdims=True)
            label, conf = majority[0], count[0] / y.size
    else:
        std = np.std(y)
        label = np.mean(y)
        conf = np.sum((-std <= y) & (y <= std)) / X.shape[0]

    if (current_depth == max_depth or n_samples <= min_samples_split or
            n_features <= min_features_split or conf >= 0.95):
        return LeafNode(is_classifier=is_classification, value=label, conf=conf,
                        samples=current_samples, features=current_features), current_depth

    best_score = np.inf
    best_w, best_b = None, None

    for _ in range(n_restarts):
        w = rng.randn(n_features)
        w /= np.linalg.norm(w) + 1e-12
        b = 0.0

        for _ in range(5):
            direction = rng.randn(n_features)
            direction /= np.linalg.norm(direction) + 1e-12

            for alpha in np.linspace(-1, 1, 11):
                w_perturbed = w + alpha * direction

                for beta in np.linspace(-1, 1, bias_steps):
                    b_perturbed = b + beta
                    margin = np.dot(X, w_perturbed) + b_perturbed
                    left, right = y[margin <= 0], y[margin > 0]

                    if len(left) == 0 or len(right) == 0:
                        continue

                    score = impurity(left, right) + lambda_reg * np.linalg.norm(w_perturbed, ord=1)

                    if score < best_score:
                        best_score = score
                        best_w = w_perturbed.copy()
                        best_b = b_perturbed

    if best_w is None:
        return LeafNode(is_classifier=is_classification, value=label, conf=conf,
                        samples=current_samples, features=current_features), current_depth

    best_w[np.abs(best_w) < threshold_value] = 0

    margin = np.dot(X, best_w) + best_b
    left_idx, right_idx = margin <= 0, margin > 0

    if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
        return LeafNode(is_classifier=is_classification, value=label, conf=conf,
                        samples=current_samples, features=current_features), current_depth

    node = Node(best_w, best_b, is_classifier=is_classification, value=label, conf=conf,
                samples=current_samples, features=current_features)

    left_child, left_depth = build_sparse_oblique_tree_oc1(X[left_idx], y[left_idx], is_classification, impurity,
                                                           max_depth, min_samples_split, min_features_split, rng,
                                                           n_restarts, bias_steps, lambda_reg, threshold_value,
                                                           current_depth + 1, current_features,
                                                           current_samples[left_idx])

    right_child, right_depth = build_sparse_oblique_tree_oc1(X[right_idx], y[right_idx], is_classification, impurity,
                                                             max_depth, min_samples_split, min_features_split, rng,
                                                             n_restarts, bias_steps, lambda_reg, threshold_value,
                                                             current_depth + 1, current_features,
                                                             current_samples[right_idx])

    node.add_left_child(left_child)
    node.add_right_child(right_child)

    return node, max(left_depth, right_depth)
