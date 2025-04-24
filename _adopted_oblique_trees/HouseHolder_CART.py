"""
HouseHolder CART (HHCART) – Corrected and Extended Implementation
-----------------------------------------------------------------

This module provides a revised implementation of the HouseHolder CART (HHCART) algorithm,
originally described by Wickramarachchi et al. (2015). The version found in the
"Ensembles of Oblique Decision Trees" package, developed by Torsha Majumder, was labelled
as HHCART(A), but its internal logic diverged from the formal HHCART(A) methodology.
Various aspects such as reflection construction, split generation, and selection
strategies did not align with the original algorithm.

To address these discrepancies, this implementation introduces a series of corrections
to better reproduce HHCART(A). Additionally, it extends functionality by incorporating
the HHCART(D) variant, as proposed by Wickramarachchi et al. (2015), which offers a more
computationally efficient alternative with minimal loss in performance — a valuable
feature for applications like scenario discovery. Furthermore, optional sparsity has been
added to HHCART(D) to enhance interpretability by reducing the complexity of decision boundaries.

Key Corrections and Enhancements:
---------------------------------
1. Class-Specific Covariance Analysis:
   - Original: Applied PCA to the full dataset at each node.
   - Revision: Computes covariance matrices per class at each node to capture
     within-class geometric structure, followed by eigen decomposition to identify
     class-specific reflection directions.

2. Exhaustive Reflection Search (HHCART(A)):
   - Original: Used only the first principal component as the reflection axis.
   - Revision: Iterates over all non-zero eigenvectors from each class, constructing
     Householder reflections for each to explore a richer set of oblique split orientations.

3. Accurate Split Generation – CARTSegmentor:
   - Original: Employed a MeanSegmentor, evaluating a single threshold per feature.
   - Revision: Introduced CARTSegmentor, performing a full threshold search by
     evaluating all midpoints between sorted feature values, consistent with CART methodology.

4. Global Split Selection Strategy:
   - Original: Accepted the first split meeting a basic impurity criterion.
   - Revision: Implements a global optimisation, selecting the split and reflection
     pair that minimises impurity across all candidates.

5. Correct Oblique Boundary Construction:
   - Original: Derived decision hyperplanes directly from PCA components.
   - Revision: Constructs the oblique decision boundary using the optimal column from
     the Householder reflection matrix, ensuring alignment between axis-aligned splits
     in reflected space and oblique boundaries in the original feature space.

6. Optional Sparsity Control for HHCART(D) (Extension):
   - Extension: Integrated SparsePCA into HHCART(D). By configuring the `alpha` parameter,
     users can induce sparsity in the dominant eigenvector, simplifying decision boundaries
     without significantly compromising performance. When `alpha=0`, the algorithm defaults
     to standard dense behaviour.

Algorithm Variants:
-------------------
- `HHCartAClassifier` (HHCART(A)):
   - Performs exhaustive search across all class-specific eigenvectors.
   - Focuses on maximising split accuracy through comprehensive reflection evaluation.
   - Operates in dense mode only (no sparsity).

- `HHCartDClassifier` (HHCART(D)):
   - Selects only the dominant eigenvector per class for efficiency.
   - Supports sparse reflections via SparsePCA (`alpha > 0`), enhancing interpretability
     by reducing active features in splits.
"""


import numpy as np
from copy import deepcopy
from scipy.linalg import norm
from sklearn.decomposition import SparsePCA
from sklearn.base import BaseEstimator, ClassifierMixin

# Use the CARTSegmentor from the package for full threshold search
from _adopted_oblique_trees.segmentor import CARTSegmentor


class HHCARTNode:
    def __init__(self, depth, labels, **kwargs):                # Defining the node structure
        self.depth = depth                                      # Depth of the node
        self.labels = labels                                    # Label associated with the node
        self.is_leaf = kwargs.get('is_leaf', False)             # 'is_leaf' is set to 'False' for internal nodes
        self._split_rules = kwargs.get('split_rules', None)     # splitting rule (tuple: (feature index, threshold))
        self._weights = kwargs.get('weights', None)             # weight vector (oblique hyperplane)
        self._left_child = kwargs.get('left_child', None)       # left child node
        self._right_child = kwargs.get('right_child', None)     # right child node
        self._alpha = kwargs.get('alpha', 0.0)
        self._label = None

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

            # Use SparsePCA to find the dominant eigenvector
            if self._alpha > 0:
                spca = SparsePCA(n_components=1, alpha=self._alpha, random_state=self.random_state)
                spca.fit(X_c)
                mu = spca.components_[0]
            else:
                S = np.cov(X_c, rowvar=False)
                eigvals, eigvecs = np.linalg.eigh(S)
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
                 random_state=None, alpha=0.0, **kwargs):
        super().__init__(impurity=impurity, segmentor=segmentor, max_depth=max_depth,
                         min_samples_split=min_samples_split, random_state=random_state,
                         alpha=alpha, **kwargs)
