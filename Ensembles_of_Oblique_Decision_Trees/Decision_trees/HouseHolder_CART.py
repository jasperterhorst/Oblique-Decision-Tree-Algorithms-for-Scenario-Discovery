# This file is an adaptation of the original HouseHolder CART implementation by Torsha Majumder.
# Although the file was originally labeled as "Implementation of HouseHolder CART-A by [Wickramarachchi et al.]",
# the algorithmic logic did not fully conform to the HHCART(A) methodology described by Wickramarachchi et al. (2015).
#
# To ensure consistency with the theoretical framework, the following modifications have been introduced:
#
# 1. Two distinct algorithmic variants are now supported:
#    - HHCartAClassifier (HHCART(A)): This variant evaluates all nonzero eigenvectors from each class's covariance
#      matrix, constructing a comprehensive set of Householder reflections. This exhaustive search improves split
#      accuracy, albeit at a higher computational cost.
#    - HHCartDClassifier (HHCART(D)): This variant utilizes only the dominant eigenvector (associated with the largest
#      eigenvalue) for generating the Householder reflection at each node, resulting in significant speed improvements
#      with similar performance.
#
# 2. The reflection and split selection mechanisms have been refined:
#    In the A variant, the algorithm iterates over every candidate eigenvector for each class and selects the split that
#    minimizes the impurity score across all reflections, whereas in the D variant the inner loop is replaced by a
#    single selection of the dominant eigenvector.
#
# 3. The default segmentor is set to use CARTSegmentor, which performs a full CART-style threshold search by evaluating
#    every valid midpoint between sorted feature values, thereby ensuring an exhaustive and correct split search.
#
# The remaining tree-building logic, including node creation, termination conditions, and prediction routines,
# remains unchanged.
#
# These adaptations yield two usable implementations: one that fully adheres to HHCART(A)'s exhaustive split search
# (HHCartAClassifier), and another (HHCartDClassifier) that reduces runtime by employing only the dominant eigenvector,
# trading a minor loss in accuracy for greater efficiency.

import numpy as np
from copy import deepcopy
from scipy.linalg import norm
from sklearn.base import BaseEstimator, ClassifierMixin

# Use the CARTSegmentor from the package for full threshold search
from Ensembles_of_Oblique_Decision_Trees.Decision_trees.segmentor import CARTSegmentor


class HHCARTNode:
    def __init__(self, depth, labels, **kwargs):  # Defining the node structure
        self.depth = depth                    # Depth of the node
        self.labels = labels                  # Label associated with the node
        self.is_leaf = kwargs.get('is_leaf', False)  # 'is_leaf' is set to 'False' for internal nodes
        self._split_rules = kwargs.get('split_rules', None)  # splitting rule (tuple: (feature index, threshold))
        self._weights = kwargs.get('weights', None)  # weight vector (oblique hyperplane)
        self._left_child = kwargs.get('left_child', None)  # left child node
        self._right_child = kwargs.get('right_child', None)  # right child node

        if not self.is_leaf:
            assert self._split_rules is not None
            assert self._left_child is not None
            assert self._right_child is not None

    def get_child(self, datum):
        if self.is_leaf:
            raise Warning("Leaf node does not have children.")
        # Evaluate the linear decision function on the datum
        X = deepcopy(datum)
        if X.dot(np.array(self._weights[:-1]).T) - self._weights[-1] < 0:
            return self.left_child
        else:
            return self.right_child

    @property
    def label(self):
        if not hasattr(self, '_label'):
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


class HouseHolderCART(BaseEstimator):
    def __init__(self, impurity, segmentor, max_depth, min_samples_split=2, method='eig',
                 tau=1e-4, random_state=None, **kwargs):
        self.impurity = impurity
        self.segmentor = segmentor  # segmentor is expected to perform CART-style split search
        self.method = method
        self.tau = tau
        self._max_depth = max_depth
        self._min_samples = min_samples_split
        self._alpha = kwargs.get('alpha', None)  # for potential extension to linear regression splits
        self._root = None
        self._nodes = []
        self.random_state = random_state
        self.debug = kwargs.get("debug", False)

    def _terminate(self, X, y, cur_depth):
        # Termination conditions: maximum depth, insufficient samples, or homogeneous node
        if self._max_depth is not None and cur_depth == self._max_depth:
            return True
        elif y.size < self._min_samples:
            return True
        elif np.unique(y).size == 1:
            return True
        else:
            return False

    def _generate_leaf_node(self, cur_depth, y):
        node = HHCARTNode(cur_depth, y, is_leaf=True)
        self._nodes.append(node)
        return node

    # NOTE: _generate_node() in this base class will be used by HHCartAClassifier (HHCART(A))
    def _generate_node(self, X, y, cur_depth):
        if self._terminate(X, y, cur_depth):
            return self._generate_leaf_node(cur_depth, y)

        n_objects, n_features = X.shape
        best_H = np.eye(n_features)  # fallback identity if no valid reflection is found
        impurity_best = float('inf')
        sr = None
        left_indices = right_indices = None

        classes = np.unique(y)
        for c in classes:
            X_c = X[y == c]
            if X_c.shape[0] <= 1:
                continue
            S = np.cov(X_c, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(S)  # ascending order

            # Loop over all eigenvectors (HHCART(A))
            for j in range(eigvecs.shape[1]):
                mu = eigvecs[:, j]
                if np.allclose(mu, 0):
                    continue
                check_ = np.sqrt(((np.eye(n_features) - mu) ** 2).sum(axis=1))
                if (check_ > self.tau).sum() > 0:
                    i = np.argmax(check_)
                    e = np.zeros(n_features)
                    e[i] = 1.0
                    w = (e - mu) / norm(e - mu)
                    H = np.eye(n_features) - 2 * np.outer(w, w)
                    X_reflected = X @ H

                    impurity_ref, sr_ref, left_i, right_i = self.segmentor(X_reflected, y, self.impurity)
                    if impurity_ref < impurity_best:
                        impurity_best = impurity_ref
                        sr = sr_ref
                        left_indices = left_i
                        right_indices = right_i
                        best_H = H

        if not sr:
            return self._generate_leaf_node(cur_depth, y)

        i, treshold = sr
        weights = np.zeros(n_features + 1)
        weights[:-1] = best_H[:, i]
        weights[-1] = treshold
        left_indices = X.dot(weights[:-1]) - weights[-1] < 0
        right_indices = ~left_indices
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]

        if len(y_right) <= self._min_samples or len(y_left) <= self._min_samples:
            return self._generate_leaf_node(cur_depth, y)

        node = HHCARTNode(
            cur_depth, y,
            split_rules=sr,
            weights=weights,
            left_child=self._generate_node(X_left, y_left, cur_depth + 1),
            right_child=self._generate_node(X_right, y_right, cur_depth + 1),
            is_leaf=False
        )
        self._nodes.append(node)
        return node

    def fit(self, X, y):
        self._root = self._generate_node(X, y, 0)

    def get_params(self, deep=True):
        return {'max_depth': self._max_depth, 'min_samples_split': self._min_samples,
                'impurity': self.impurity, 'segmentor': self.segmentor}

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


# The following two classes instantiate different HHCART variants.
# HHCartAClassifier implements HHCART(A), which uses all eigenvectors from each class.
class HHCartAClassifier(ClassifierMixin, HouseHolderCART):
    def __init__(self, impurity, segmentor=CARTSegmentor(), max_depth=50, min_samples_split=2,
                 random_state=None, **kwargs):
        # Use the base _generate_node, which loops over all eigenvectors.
        super().__init__(impurity=impurity, segmentor=segmentor, max_depth=max_depth,
                         min_samples_split=min_samples_split, random_state=random_state, **kwargs)


# HHCartDClassifier implements HHCART(D), which uses only the dominant eigenvector from each class.
class HHCartDClassifier(ClassifierMixin, HouseHolderCART):
    def _generate_node(self, X, y, cur_depth):
        if self._terminate(X, y, cur_depth):
            return self._generate_leaf_node(cur_depth, y)

        n_objects, n_features = X.shape
        best_H = np.eye(n_features)
        impurity_best = float('inf')
        sr = None
        left_indices = right_indices = None

        classes = np.unique(y)
        for c in classes:
            X_c = X[y == c]
            if X_c.shape[0] <= 1:
                continue
            S = np.cov(X_c, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(S)  # ascending order
            # Use only the dominant eigenvector: the one associated with the largest eigenvalue.
            mu = eigvecs[:, -1]
            if np.allclose(mu, 0):
                continue

            check_ = np.sqrt(((np.eye(n_features) - mu) ** 2).sum(axis=1))
            if (check_ > self.tau).sum() > 0:
                i = np.argmax(check_)
                e = np.zeros(n_features)
                e[i] = 1.0
                w = (e - mu) / norm(e - mu)
                H = np.eye(n_features) - 2 * np.outer(w, w)
                X_reflected = X @ H

                impurity_ref, sr_ref, left_i, right_i = self.segmentor(X_reflected, y, self.impurity)
                if impurity_ref < impurity_best:
                    impurity_best = impurity_ref
                    sr = sr_ref
                    left_indices = left_i
                    right_indices = right_i
                    best_H = H

        if not sr:
            return self._generate_leaf_node(cur_depth, y)

        i, treshold = sr
        weights = np.zeros(n_features + 1)
        weights[:-1] = best_H[:, i]
        weights[-1] = treshold
        left_indices = X.dot(weights[:-1]) - weights[-1] < 0
        right_indices = ~left_indices
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]

        if len(y_right) <= self._min_samples or len(y_left) <= self._min_samples:
            return self._generate_leaf_node(cur_depth, y)

        node = HHCARTNode(
            cur_depth, y,
            split_rules=sr,
            weights=weights,
            left_child=self._generate_node(X_left, y_left, cur_depth + 1),
            right_child=self._generate_node(X_right, y_right, cur_depth + 1),
            is_leaf=False
        )
        self._nodes.append(node)
        return node

    def __init__(self, impurity, segmentor=CARTSegmentor(), max_depth=50, min_samples_split=2,
                 random_state=None, **kwargs):
        # Call the parent constructor; the _generate_node method has been overridden.
        super().__init__(impurity=impurity, segmentor=segmentor, max_depth=max_depth,
                         min_samples_split=min_samples_split, random_state=random_state, **kwargs)


# import numpy as np
# from copy import deepcopy
# from scipy.linalg import norm
# from sklearn.base import BaseEstimator, ClassifierMixin
#
# from Ensembles_of_Oblique_Decision_Trees.Decision_trees.segmentor import CARTSegmentor
#
#
# class HHCARTNode:
#
#     def __init__(self, depth, labels, **kwargs):  # Defining the node structure
#         self.depth = depth  # Depth of the node
#         self.labels = labels  # Label associated with the node
#         self.is_leaf = kwargs.get('is_leaf', False)  # 'is_leaf' is set to 'False' for internal nodes
#         self._split_rules = kwargs.get('split_rules', None)  # 'split_rules' is set to 'None'
#         self._weights = kwargs.get('weights', None)  # weights associated with the node
#         self._left_child = kwargs.get('left_child', None)  # left_child index
#         self._right_child = kwargs.get('right_child', None)  # right_child index
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
# class HouseHolderCART(BaseEstimator):
#
#     def __init__(self, impurity, segmentor, max_depth, min_samples_split=2, method='eig',
#                  tau=1e-4, random_state=None, **kwargs):
#         self.impurity = impurity
#         self.segmentor = segmentor
#         self.method = method
#         self.tau = tau
#         self._max_depth = max_depth
#         self._min_samples = min_samples_split
#         self._alpha = kwargs.get('alpha', None)  # only for linreg method
#         self._root = None
#         self._nodes = []
#         self.random_state = random_state
#         self.debug = kwargs.get("debug", False)
#
#     def _terminate(self, X, y, cur_depth):  # termination conditions
#
#         if self._max_depth != None and cur_depth == self._max_depth:  # maximum depth is reached
#             return True
#         elif y.size < self._min_samples:  # minimum number of samples has been reached
#             return True
#         elif np.unique(y).size == 1:  # terminate if the node is homogeneous
#             return True
#         else:
#             return False
#
#     def _generate_leaf_node(self, cur_depth, y):
#
#         node = HHCARTNode(cur_depth, y, is_leaf=True)
#         self._nodes.append(node)
#         return node
#
#     def _generate_node(self, X, y, cur_depth):
#         if self._terminate(X, y, cur_depth):
#             return self._generate_leaf_node(cur_depth, y)
#
#         n_objects, n_features = X.shape
#         best_H = np.eye(n_features)
#         impurity_best = float('inf')
#         sr = None
#         left_indices = right_indices = None
#
#         classes = np.unique(y)
#         for c in classes:
#             X_c = X[y == c]
#             if X_c.shape[0] <= 1:
#                 continue
#             S = np.cov(X_c, rowvar=False)
#             eigvals, eigvecs = np.linalg.eigh(S)  # ascending order
#
#             for j in range(eigvecs.shape[1]):
#                 mu = eigvecs[:, j]
#                 if np.allclose(mu, 0):
#                     continue
#
#                 check_ = np.sqrt(((np.eye(n_features) - mu) ** 2).sum(axis=1))
#                 if (check_ > self.tau).sum() > 0:
#                     i = np.argmax(check_)
#                     e = np.zeros(n_features)
#                     e[i] = 1.0
#                     w = (e - mu) / norm(e - mu)
#                     H = np.eye(n_features) - 2 * np.outer(w, w)
#                     X_reflected = X @ H
#
#                     impurity_ref, sr_ref, left_i, right_i = self.segmentor(X_reflected, y, self.impurity)
#                     if impurity_ref < impurity_best:
#                         impurity_best = impurity_ref
#                         sr = sr_ref
#                         left_indices = left_i
#                         right_indices = right_i
#                         best_H = H
#
#         if not sr:
#             return self._generate_leaf_node(cur_depth, y)
#
#         i, treshold = sr
#         weights = np.zeros(n_features + 1)
#         weights[:-1] = best_H[:, i]
#         weights[-1] = treshold
#         left_indices = X.dot(weights[:-1]) - weights[-1] < 0
#         right_indices = ~left_indices
#         X_left, y_left = X[left_indices], y[left_indices]
#         X_right, y_right = X[right_indices], y[right_indices]
#
#         if len(y_right) <= self._min_samples or len(y_left) <= self._min_samples:
#             return self._generate_leaf_node(cur_depth, y)
#
#         node = HHCARTNode(
#             cur_depth, y,
#             split_rules=sr,
#             weights=weights,
#             left_child=self._generate_node(X_left, y_left, cur_depth + 1),
#             right_child=self._generate_node(X_right, y_right, cur_depth + 1),
#             is_leaf=False
#         )
#         self._nodes.append(node)
#         return node
#
#     def fit(self, X, y):
#
#         self._root = self._generate_node(X, y, 0)
#
#     def get_params(self, deep=True):
#
#         return {'max_depth': self._max_depth, 'min_samples_split': self._min_samples,
#                 'impurity': self.impurity, 'segmentor': self.segmentor}
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
# #
# # Definition of classes provided: HHCartClassifier
# #
# class HHCartClassifier(ClassifierMixin, HouseHolderCART):
#     def __init__(self, impurity, segmentor=CARTSegmentor(), max_depth=50, min_samples_split=2,
#     random_state=None, **kwargs):
#         super().__init__(impurity=impurity, segmentor=segmentor, max_depth=max_depth,
#                          min_samples_split=min_samples_split, random_state=random_state, **kwargs)
