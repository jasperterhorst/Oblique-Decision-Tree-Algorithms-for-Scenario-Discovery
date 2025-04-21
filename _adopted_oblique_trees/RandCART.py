# Implementation of Randomized CART
# .....Importing all the packages................

import numpy as np
from copy import deepcopy
from scipy.linalg import qr
from sklearn.base import BaseEstimator, ClassifierMixin


class Node:
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
            assert self._split_rules
            assert self._left_child
            assert self._right_child

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
            if len(self.labels) == 0:
                return 0  # fallback label to avoid crash
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
            raise Warning("Leaf node does not have split rule.")
        return self._left_child

    @property
    def right_child(self):
        if self.is_leaf:
            raise Warning("Leaf node does not have split rule.")
        return self._right_child


class Rand_CART(BaseEstimator):
    def __init__(self, impurity, segmentor, max_depth, min_samples_split=2, random_state=None,
                 n_rotations=1, compare_with_cart=False):
        self.impurity = impurity
        self.segmentor = segmentor
        self._max_depth = max_depth
        self._min_samples = min_samples_split
        self._compare_with_cart = compare_with_cart
        self._root = None
        self._nodes = []
        self.n_rotations = n_rotations
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def _terminate(self, X, y, cur_depth):
        return (
            self._max_depth is not None and cur_depth == self._max_depth
            or y.size < self._min_samples
            or np.unique(y).size == 1
        )

    def _generate_leaf_node(self, cur_depth, y):
        node = Node(cur_depth, y, is_leaf=True)
        self._nodes.append(node)
        return node

    def _generate_node(self, X, y, cur_depth):
        if self._terminate(X, y, cur_depth):
            return self._generate_leaf_node(cur_depth, y)

        n_objects, n_features = X.shape
        best_score = -np.inf
        best_result = None

        for _ in range(self.n_rotations):
            matrix = self.rng.multivariate_normal(np.zeros(n_features), np.diag(np.ones(n_features)), n_features)
            Q, _ = qr(matrix)
            X_rot = X @ Q
            impurity, sr, left_idx, right_idx = self.segmentor(X_rot, y, self.impurity)

            if sr and impurity > best_score:
                best_score = impurity
                best_result = (Q, sr, left_idx, right_idx)

        if self._compare_with_cart:
            imp_cart, sr_cart, left_cart, right_cart = self.segmentor(X, y, self.impurity)
            if not best_result or imp_cart > best_score:
                Q = np.eye(n_features)
                best_result = (Q, sr_cart, left_cart, right_cart)

        if not best_result:
            return self._generate_leaf_node(cur_depth, y)

        Q, sr, left_idx, right_idx = best_result
        i, threshold = sr
        weights = np.zeros(n_features + 1)
        weights[:-1] = Q[:, i]
        weights[-1] = threshold

        left_mask = X @ weights[:-1] - weights[-1] < 0
        right_mask = ~left_mask
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        if len(y_left) <= self._min_samples or len(y_right) <= self._min_samples:
            return self._generate_leaf_node(cur_depth, y)

        node = Node(cur_depth, y, split_rules=sr, weights=weights,
                    left_child=self._generate_node(X_left, y_left, cur_depth + 1),
                    right_child=self._generate_node(X_right, y_right, cur_depth + 1),
                    is_leaf=False)
        self._nodes.append(node)
        return node

    def fit(self, X, y):
        self._root = self._generate_node(X, y, 0)

    def get_params(self, deep=True):
        return {
            'max_depth': self._max_depth,
            'min_samples_split': self._min_samples,
            'impurity': self.impurity,
            'segmentor': self.segmentor,
            'random_state': self.random_state,
            'n_rotations': self.n_rotations,
            'compare_with_cart': self._compare_with_cart
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

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
        if not self._root:
            raise Warning("Decision tree has not been trained.")
        predictions = self.predict(data)
        correct_count = np.count_nonzero(predictions == labels)
        return correct_count / labels.shape[0]


class RandCARTClassifier(ClassifierMixin, Rand_CART):
    def __init__(self, impurity, segmentor, max_depth=50, min_samples_split=2,
                 random_state=None, n_rotations=1, compare_with_cart=False):
        super().__init__(impurity=impurity, segmentor=segmentor, max_depth=max_depth,
                         min_samples_split=min_samples_split, random_state=random_state,
                         n_rotations=n_rotations, compare_with_cart=compare_with_cart)


# # Implementation of Randomized CART
#
# # .....Importing all the packages................
# import numpy as np
# from copy import deepcopy
# from scipy.linalg import qr
# from sklearn.base import BaseEstimator, ClassifierMixin
#
#
# class Node:                                                                 # definition of the Node
#
#     def __init__(self, depth, labels, **kwargs):
#         self.depth = depth
#         self.labels = labels
#         self.is_leaf = kwargs.get('is_leaf', False)
#         self._split_rules = kwargs.get('split_rules', None)
#         self._weights = kwargs.get('weights', None)
#         self._left_child = kwargs.get('left_child', None)
#         self._right_child = kwargs.get('right_child', None)
#         self._label = None
#
#         if not self.is_leaf:
#             assert self._split_rules
#             assert self._left_child
#             assert self._right_child
#
#     def get_child(self, datum):
#         if self.is_leaf:
#             raise Warning("Leaf node does not have children.")
#         X = deepcopy(datum)
#
#         if X.dot(np.array(self._weights[:-1]).T) - self._weights[-1] < 0:
#             return self.left_child
#         else:
#             return self.right_child
#
#     @property
#     def label(self):
#         if not hasattr(self, '_label'):
#             classes, counts = np.unique(self.labels, return_counts=True)
#             self._label = classes[np.argmax(counts)]
#         return self._label
#
#     @property
#     def split_rules(self):
#         if self.is_leaf:
#             raise Warning("Leaf node does not have split rule.")
#         return self._split_rules
#
#     @property
#     def left_child(self):
#         if self.is_leaf:
#             raise Warning("Leaf node does not have split rule.")
#         return self._left_child
#
#     @property
#     def right_child(self):
#         if self.is_leaf:
#             raise Warning("Leaf node does not have split rule.")
#         return self._right_child
#
#
# class RandCART(BaseEstimator):
#     def __init__(self, impurity, segmentor, max_depth, min_samples_split=2, random_state=None,
#                  n_rotations=1, compare_with_cart=False):
#         self.impurity = impurity
#         self.segmentor = segmentor
#         self._max_depth = max_depth
#         self._min_samples = min_samples_split
#         self._compare_with_cart = compare_with_cart
#         self._root = None
#         self._nodes = []
#         self.n_rotations = n_rotations
#         self.random_state = random_state
#         self.rng = np.random.RandomState(random_state)
#
#     def _terminate(self, X, y, cur_depth):
#         return (
#             self._max_depth is not None and cur_depth == self._max_depth
#             or y.size < self._min_samples
#             or np.unique(y).size == 1
#         )
#
#     def _generate_leaf_node(self, cur_depth, y):
#         node = Node(cur_depth, y, is_leaf=True)
#         self._nodes.append(node)
#         return node
#
#     def _generate_node(self, X, y, cur_depth):
#         if self._terminate(X, y, cur_depth):
#             return self._generate_leaf_node(cur_depth, y)
#
#         n_objects, n_features = X.shape
#         best_score = -np.inf
#         best_result = None
#
#         for _ in range(self.n_rotations):
#             matrix = self.rng.multivariate_normal(np.zeros(n_features), np.diag(np.ones(n_features)), n_features)
#             Q, _ = qr(matrix)
#             X_rot = X @ Q
#             impurity, sr, left_idx, right_idx = self.segmentor(X_rot, y, self.impurity)
#
#             if sr and impurity > best_score:
#                 best_score = impurity
#                 best_result = (Q, sr, left_idx, right_idx)
#
#         if self._compare_with_cart:
#             imp_cart, sr_cart, left_cart, right_cart = self.segmentor(X, y, self.impurity)
#             if not best_result or imp_cart > best_score:
#                 Q = np.eye(n_features)
#                 best_result = (Q, sr_cart, left_cart, right_cart)
#
#         if not best_result:
#             return self._generate_leaf_node(cur_depth, y)
#
#         Q, sr, left_idx, right_idx = best_result
#         i, threshold = sr
#         weights = np.zeros(n_features + 1)
#         weights[:-1] = Q[:, i]
#         weights[-1] = threshold
#
#         left_mask = X @ weights[:-1] - weights[-1] < 0
#         right_mask = ~left_mask
#         X_left, y_left = X[left_mask], y[left_mask]
#         X_right, y_right = X[right_mask], y[right_mask]
#
#         if len(y_left) <= self._min_samples or len(y_right) <= self._min_samples:
#             return self._generate_leaf_node(cur_depth, y)
#
#         node = Node(cur_depth, y, split_rules=sr, weights=weights,
#                     left_child=self._generate_node(X_left, y_left, cur_depth + 1),
#                     right_child=self._generate_node(X_right, y_right, cur_depth + 1),
#                     is_leaf=False)
#         self._nodes.append(node)
#         return node
#
#     def fit(self, X, y):
#         self._root = self._generate_node(X, y, 0)
#
#     def get_params(self, deep=True):
#         return {
#             'max_depth': self._max_depth,
#             'min_samples_split': self._min_samples,
#             'impurity': self.impurity,
#             'segmentor': self.segmentor,
#             'random_state': self.random_state,
#             'n_rotations': self.n_rotations,
#             'compare_with_cart': self._compare_with_cart
#         }
#
#     def set_params(self, **parameters):
#         for parameter, value in parameters.items():
#             setattr(self, parameter, value)
#         return self
#
#     def predict(self, X):
#         def predict_single(datum):
#             cur_node = self._root
#             while not cur_node.is_leaf:
#                 cur_node = cur_node.get_child(datum)
#             return cur_node.label
#
#         if not self._root:
#             raise Warning("Decision tree has not been trained.")
#         size = X.shape[0]
#         predictions = np.empty((size,), dtype=int)
#         for i in range(size):
#             predictions[i] = predict_single(X[i, :])
#
#         return predictions
#
#     def score(self, data, labels):
#         if not self._root:
#             raise Warning("Decision tree has not been trained.")
#         predictions = self.predict(data)
#         correct_count = np.count_nonzero(predictions == labels)
#         return correct_count / labels.shape[0]
#
#
# # Definition of classes provided: HHCartClassifier
# class RandCARTClassifier(ClassifierMixin, RandCART):
#     def __init__(self, impurity, segmentor, max_depth=50, min_samples_split=2,
#                  random_state=None, n_rotations=1, compare_with_cart=False):
#         super().__init__(impurity=impurity, segmentor=segmentor, max_depth=max_depth,
#                          min_samples_split=min_samples_split, random_state=random_state,
#                          n_rotations=n_rotations, compare_with_cart=compare_with_cart)
