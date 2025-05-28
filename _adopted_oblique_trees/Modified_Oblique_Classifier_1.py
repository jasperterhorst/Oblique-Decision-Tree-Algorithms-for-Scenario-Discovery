"""
Modified Oblique Classifier 1 (MOC1) – Corrected and Simplified Implementation
------------------------------------------------------------------------------

This module provides a modified and partially corrected implementation of the
Oblique Classifier 1 (OC1) decision tree algorithm, originally introduced by
Murthy et al. (1994). It builds upon the OC1 variant included in the
"Ensembles of Oblique Decision Trees" package by Torsha Majumder (2020),
which diverged from Murthy's design in multiple respects.

The present implementation reintroduces key aspects of the original OC1 logic
while deliberately omitting others for the sake of runtime tractability. It
restores the overall structure of oblique split search through random restarts
and bias perturbation, but does not fully reproduce Murthy's coordinate-wise
local search or probabilistic acceptance rules.

Comparison of OC1 Variants:
---------------------------
- Original OC1 (Murthy et al., 1994):
    - Coordinate-wise weight perturbation
    - Probabilistic acceptance of non-improving moves (P = exp(-k))
    - Bias term adjustment
    - Multiple restarts
    - Greedy tree induction with global impurity-based split selection

- Adopted OC1 (Majumder, 2020):
    - Arbitrary vector perturbation (no local search logic)
    - No restarts or probabilistic acceptance
    - Simplified tree structure and impurity handling

- Modified OC1 (MOC1) (this implementation):
    - Full-vector perturbation along random directions (no coordinate-wise search)
    - Bias term perturbation via `bias_steps`
    - Multiple restarts via `n_restarts`
    - Global impurity-based split selection (greedy but exhaustive per node)
    - Fully reproducible via seeded RNG
    - Greedy acceptance only (no probabilistic logic)

Implemented Enhancements:
--------------------------
1. Stochastic Restarts for Global Exploration
   Randomly initialises the hyperplane `n_restarts` times to avoid local optima.

2. Bias Term Perturbation*
   Systematically shifts the bias (threshold) term along the hyperplane normal
   using `bias_steps`, expanding the space of evaluated splits.

3. Global Impurity-Based Split Selection
   Evaluates all candidate (w, b) combinations across perturbations and restarts,
   selecting the globally best split at each node.

4. Reproducible Randomness
   All random elements are generated via a local `RandomState` seeded with
   `random_state`, ensuring deterministic behaviour for benchmarking.

Not Implemented:
----------------
- Coordinate-Wise Perturbation
  The current version uses full-vector random direction perturbations instead
  of feature-wise adjustments. Coordinate-wise perturbation was found to be very
  computationally expensive, and was therefore not implemented in this version.

- Probabilistic Acceptance of Non-Improving Splits
  The implementation uses greedy acceptance only; no stochastic acceptance
  rule is applied.

Tree Construction:
------------------
- Begins with `n_restarts` random hyperplanes.
- For each, performs random direction perturbations of weights and bias.
- Evaluates Gini impurity (or Twoing) to score candidate splits.
- Tracks the split with the lowest impurity across all trials.
- Recursively applies this process to build the full tree.

"""

from warnings import warn
import numpy as np
from collections import Counter
from _adopted_oblique_trees.OC1_tree_structure import Tree, Node, LeafNode
from _adopted_oblique_trees import split_criteria
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier

epsilon = 1e-6


# Definition of classes provided: ObliqueClassifier1
class BaseObliqueTree(BaseEstimator):

    def __init__(self, impurity, max_depth, min_samples_split, min_features_split, random_state=None,
                 n_restarts=20, bias_steps=20):

        # Get the options for tree learning
        self.impurity = impurity  # splitting impurity - default is 'gini-index'
        self.max_depth = max_depth  # maximum depth of the tree
        self.min_samples_split = min_samples_split  # minimum number of samples needed for a split
        self.min_features_split = min_features_split  # minimum number of features needed for a split
        self.tree_ = None  # Internal tree - initially set as 'None'
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state) if random_state is not None else np.random
        self.n_restarts = n_restarts
        self.bias_steps = bias_steps

        self.root_node = None
        self.learned_depth = None

    # Find the minimum number of samples per split
    def get_min_samples_split(self, n_samples):
        if isinstance(self.min_samples_split, int):
            if self.min_samples_split < 2:
                self.min_samples_split = 2
                warn('min_samples_split specified incorrectly; setting to default value of 2.')
            min_samples = self.min_samples_split
        elif isinstance(self.min_samples_split, float):
            if not 0. < self.min_samples_split <= 1.:
                self.min_samples_split = 2 / n_samples
                warn('min_samples_split not between [0, 1]; setting to default value of 2.')
                min_samples = 2
            else:
                min_samples = int(np.ceil(self.min_samples_split * n_samples))
        else:
            warn('Invalid value for min_samples_split; setting to default value of 2')
            min_samples = 2

        return min_samples

    # Find the minimum number of features per split
    def get_min_features_split(self, n_features):
        if isinstance(self.min_features_split, int):
            if self.min_features_split < 1:
                self.min_features_split = 1
                warn('min_features_split specified incorrectly; setting to default value of 1.')
            min_features = self.min_features_split
        elif isinstance(self.min_features_split, float):
            if not 0. < self.min_features_split <= 1.:
                self.min_features_split = 1 / n_features
                warn('min_features_split not between [0, 1]; setting to default value of 1.')
                min_features = 1
            else:
                min_features = int(np.ceil(self.min_features_split * n_features))
        else:
            warn('Invalid value for min_features_split; setting to default value of 1')
            min_features = 1

        return min_features

    def fit(self, X, y):
        # Ensure that X is a 2d array of shape (n_samples, n_features)
        if X.ndim == 1:  # single feature from all examples
            if len(y) > 1:
                X = X.reshape(-1, 1)
            elif len(y) == 1:  # single training example
                X = X.reshape(1, -1)
            else:
                ValueError('Invalid X and y')

        if self.impurity == 'gini':
            self.impurity = split_criteria.gini
        elif self.impurity == 'twoing':
            self.impurity = split_criteria.twoing
        else:
            ValueError('Unrecognized split impurity specified. Allowed split criteria are:\n'
                       '[classification] "gini": Gini impurity, "twoing": Twoing rule')

        n_samples, n_features = X.shape
        min_samples_split = self.get_min_samples_split(n_samples)
        min_features_split = self.get_min_features_split(n_features)

        self.root_node, self.learned_depth = build_oblique_tree_moc1(
            X, y, is_classifier(self), self.impurity, self.max_depth, min_samples_split,
            min_features_split, rng=self.rng, n_restarts=self.n_restarts, bias_steps=self.bias_steps
        )

        # Create a tree object
        self.tree_ = Tree(n_features=n_features, is_classifier=is_classifier(self))
        self.tree_.set_root_node(self.root_node)
        self.tree_.set_depth(self.learned_depth)

    def predict(self, X):
        y = self.tree_.root_node.predict(X)
        y = np.array(y, dtype=int)
        return y

    def get_params(self, deep=False):
        return {'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'impurity': self.impurity,
                'min_features_split': self.min_features_split,
                'n_restarts': self.n_restarts,
                'bias_steps': self.bias_steps
                }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


# Definition of classes provided: ObliqueClassifier1
class ModifiedObliqueClassifier1(ClassifierMixin, BaseObliqueTree):
    def __init__(self, impurity="gini", max_depth=3, min_samples_split=2, min_features_split=1,
                 random_state=None, n_restarts=5, bias_steps=5):
        super().__init__(impurity=impurity, max_depth=max_depth, min_samples_split=min_samples_split,
                         min_features_split=min_features_split, random_state=random_state,
                         n_restarts=n_restarts, bias_steps=bias_steps)


# Implements algorithm to learn an oblique decision tree via random perturbations
def build_oblique_tree_moc1(X, y, is_classification, impurity,
                            max_depth, min_samples_split, min_features_split, rng=None,
                            n_restarts=5, bias_steps=5, current_depth=0, current_features=None,
                            current_samples=None, debug=False):
    """
    Builds an MOC1-style Oblique Decision Tree.

    Implements:
    - Random restarts of oblique hyperplanes
    - Vector perturbation of weights
    - Bias term adjustment
    - Greedy impurity minimisation

    Parameters:
    - X: ndarray of shape (n_samples, n_features)
    - y: labels
    - is_classification: bool
    - impurity: impurity function (e.g. Gini, Twoing)
    """

    n_samples, n_features = X.shape

    # Initialize feature/sample tracking for root node
    if current_depth == 0:
        current_features = np.arange(n_features)
        current_samples = np.arange(n_samples)

    y = np.atleast_1d(y)  # Ensure y is at least a 1D array

    # ------------------ LEAF LABEL -------------------------
    # Estimate the label of this node
    if is_classification:
        # If there is only one sample, use it directly.
        if y.size == 1:
            label = y[0]
            conf = 1.0
        else:
            label_counts = Counter(y)
            max_count = max(label_counts.values())
            label = min(k for k, v in label_counts.items() if v == max_count)
            conf = max_count / y.size

    else:
        std = np.std(y)
        label = np.mean(y)
        conf = np.sum((-std <= y) & (y <= std)) / X.shape[0]

    # ------------------ TERMINATION CHECK -------------------------
    if (current_depth == max_depth or  # max depth reached
            n_samples <= min_samples_split or  # not enough samples to split on
            n_features <= min_features_split or  # not enough features to split on
            conf >= 0.95):  # node is very homogeneous

        return LeafNode(is_classifier=is_classification, value=label, conf=conf,
                        samples=current_samples, features=current_features), current_depth

    # ------------------ INIT SPLIT FROM AXIS -------------------------
    feature_splits = get_best_splits(X, y, impurity=impurity)  # Get the best split for each feature
    f = np.argmin(feature_splits[:, 1])  # Find the best feature to split on
    best_split_score = feature_splits[f, 1]  # Save the score corresponding to the best split
    w, b = np.eye(1, n_features, f).squeeze(), -feature_splits[f, 0]  # Construct a (w, b) from the best split
    # X[f] <= s becomes 1. X[f] + 0. X[rest] - s <= 0

    # ------------------ MOC1 PERTURBATION SEARCH -------------------------
    # This block performs the true MOC1 split selection with:
    # 1. Multiple random restarts of (w, b)
    # 2. Vector perturbation of weights (w)
    # 3. Bias term (b) perturbation
    # 4. Tracks the globally best (w, b) pair

    best_score = np.inf
    best_w = None
    best_b = None

    for _ in range(n_restarts):
        # Step 1: Randomly initialize a direction vector w (unit length)
        w = rng.randn(n_features)
        w /= np.linalg.norm(w) + 1e-12
        b = 0.0  # Start with neutral bias

        # Step 2: Evaluate base score for initial (w, b)
        margin = np.dot(X, w) + b
        left, right = y[margin <= 0], y[margin > 0]
        base_score = impurity(left, right)

        # Step 3: Try several random direction vectors (for weight space)
        for _ in range(5):  # number of directions to try
            direction = rng.randn(n_features)
            direction /= np.linalg.norm(direction) + 1e-12

            for alpha in np.linspace(-1, 1, 11):  # step sizes for weight perturbation
                w_perturbed = w + alpha * direction

                # Step 4: For each w_perturbed, try perturbing bias as well
                for beta in np.linspace(-1, 1, bias_steps):
                    b_perturbed = b + beta

                    margin = np.dot(X, w_perturbed) + b_perturbed
                    left, right = y[margin <= 0], y[margin > 0]

                    if len(left) == 0 or len(right) == 0:
                        continue  # skip invalid splits

                    score = impurity(left, right)

                    if score < best_score:
                        best_score = score
                        best_w = w_perturbed.copy()
                        best_b = b_perturbed

    # ------------------------- FINAL VALIDATION -----------------------
    if best_w is None:
        # Could not find valid split — make leaf node
        return LeafNode(is_classifier=is_classification, value=label, conf=conf,
                        samples=current_samples, features=current_features), current_depth

    w, b = best_w, best_b

    # ------------------------- SANITY CHECKS -------------------------
    idx = np.where(w == np.inf)[0]
    if len(idx) != 0:
        w[idx] = rng.rand(len(idx))
    idx = np.where(w == -np.inf)[0]
    if len(idx) != 0:
        w[idx] = -1 * rng.rand(len(idx))
    if b == np.inf:
        b = 10
    if b == -np.inf:
        b = -10

    # ------------------------- PERFORM FINAL SPLIT -------------------------
    margin = np.dot(X, w) + b
    left, right = margin <= 0, margin > 0

    if len(y[left]) == 0 or len(y[right]) == 0:
        return LeafNode(is_classifier=is_classification, value=label, conf=conf,
                        samples=current_samples, features=current_features), current_depth

    decision_node = Node(w, b, is_classifier=is_classification, value=label, conf=conf,
                         samples=current_samples, features=current_features)

    left_node, left_depth = build_oblique_tree_moc1(X[left], y[left], is_classification, impurity,
                                                    max_depth, min_samples_split, min_features_split,
                                                    rng=rng, n_restarts=n_restarts, bias_steps=bias_steps,
                                                    current_depth=current_depth + 1,
                                                    current_features=current_features,
                                                    current_samples=current_samples[left])

    decision_node.add_left_child(left_node)

    right_node, right_depth = build_oblique_tree_moc1(X[right], y[right], is_classification, impurity,
                                                      max_depth, min_samples_split, min_features_split,
                                                      rng=rng, n_restarts=n_restarts, bias_steps=bias_steps,
                                                      current_depth=current_depth + 1,
                                                      current_features=current_features,
                                                      current_samples=current_samples[right])

    decision_node.add_right_child(right_node)

    return decision_node, max(left_depth, right_depth)


# Get the best splitting threshold for each feature/attribute by considering them independently of the others
def get_best_splits(X, y, impurity=split_criteria.gini):
    n_samples, n_features = X.shape
    all_splits = np.zeros((n_features, 2))

    for f in range(n_features):
        feature_values = np.sort(X[:, f])
        feature_splits = np.convolve(feature_values, [0.5, 0.5])[1:-1]
        scores = np.empty_like(feature_splits)
        best_split = None
        best_score = np.inf

        # Compute the scores
        for i, s in enumerate(feature_splits):
            left, right = y[X[:, f] <= s], y[X[:, f] > s]
            scores[i] = impurity(left, right)
            if scores[i] < best_score:
                best_score = scores[i]
                best_split = s

        all_splits[f, :] = [best_split, best_score]

    return all_splits


# """
# Oblique Classifier 1 (OC1) – Corrected Implementation Overview
# ---------------------------------------------------------------
#
# This module provides a revised implementation of the Oblique Classifier 1 (OC1) algorithm,
# originally introduced by Murthy et al. (1994). The version included in the
# "Ensembles of Oblique Decision Trees" package by Torsha Majumder deviated from several
# core principles underpinning OC1’s design. This revision addresses those inconsistencies
# to restore theoretical alignment, thereby also enhancing split quality. And increasing runtime a lot :(.
#
# Key Corrections and Enhancements:
# ---------------------------------
# 1. Coordinate-Wise Perturbation (Local Search Restoration):
#    - Original: Applied arbitrary full-vector perturbations during split search.
#    - Revision: Implements component-wise (coordinate) perturbation, adjusting one
#      feature weight at a time, in line with OC1’s intended local search dynamics
#      for fine-grained impurity minimisation.
#
# 2. Probabilistic Acceptance of Non-Improving Perturbations:
#    - Original: Accepted only strictly improving splits, limiting exploration.
#    - Revision: Reintroduces OC1’s stochastic acceptance rule
#      (P_update = exp(-k)) to allow occasional acceptance of non-improving moves,
#      enhancing the ability to escape local minima and flat impurity regions.
#
# 3. Stochastic Restarts for Global Exploration:
#    - Original: Used a fixed or single initial hyperplane per node.
#    - Revision: Adds a configurable `n_restarts` parameter, enabling multiple
#      random restarts to improve coverage of the oblique split space, selecting
#      the globally best split across all attempts.
#
# 4. Reproducible Randomness:
#    - Original: Relied on global NumPy RNG, causing inconsistent outcomes.
#    - Revision: Utilises a local `RandomState` seeded via `random_state`, ensuring
#      deterministic behaviour across runs for benchmarking and experimentation.
#
# 5. Global Impurity-Based Split Selection:
#    - Original: Accepted the first split that improved impurity.
#    - Revision: Evaluates all candidate splits across perturbations, bias shifts,
#      and restarts, selecting the split that minimises impurity globally.
#
# Algorithm Workflow:
# -------------------
# - Random Initialisation: Begins with multiple random hyperplanes (`n_restarts`).
# - Local Search: Performs coordinate-wise perturbations of weights and bias adjustments.
# - Stochastic Acceptance: Applies probabilistic rules to avoid premature convergence.
# - Global Selection: Retains the split achieving minimal impurity across all trials.
# - Tree Construction: Recursively builds the tree structure using the selected splits.
# """
#
# from warnings import warn
# import numpy as np
# from scipy.stats import mode
# from _adopted_oblique_trees.OC1_tree_structure import Tree, Node, LeafNode
# from _adopted_oblique_trees import split_criteria
# from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
#
# epsilon = 1e-6
#
#
# # Definition of classes provided: ObliqueClassifier1
# class BaseObliqueTree(BaseEstimator):
#
#     def __init__(self, impurity, max_depth, min_samples_split, min_features_split, random_state=None,
#                  n_restarts=10, max_stagnation=10, max_equal_accepts=10):
#
#         # Get the options for tree learning
#         self.impurity = impurity  # splitting impurity - default is 'sum_impurity'
#         self.max_depth = max_depth  # maximum depth of the tree
#         self.min_samples_split = min_samples_split  # minimum number of samples needed for a split
#         self.min_features_split = min_features_split  # minimum number of features needed for a split
#         self.tree_ = None  # Internal tree - initially set as 'None'
#         self.random_state = random_state
#         self.rng = np.random.RandomState(random_state) if random_state is not None else np.random
#         self.n_restarts = n_restarts
#         self.max_stagnation = max_stagnation
#
#         self.root_node = None
#         self.learned_depth = None
#
#     # Find the minimum number of samples per split
#     def get_min_samples_split(self, n_samples):
#         if isinstance(self.min_samples_split, int):
#             if self.min_samples_split < 2:
#                 self.min_samples_split = 2
#                 warn('min_samples_split specified incorrectly; setting to default value of 2.')
#             min_samples = self.min_samples_split
#         elif isinstance(self.min_samples_split, float):
#             if not 0. < self.min_samples_split <= 1.:
#                 self.min_samples_split = 2 / n_samples
#                 warn('min_samples_split not between [0, 1]; setting to default value of 2.')
#                 min_samples = 2
#             else:
#                 min_samples = int(np.ceil(self.min_samples_split * n_samples))
#         else:
#             warn('Invalid value for min_samples_split; setting to default value of 2')
#             min_samples = 2
#
#         return min_samples
#
#     # Find the minimum number of features per split
#     def get_min_features_split(self, n_features):
#         if isinstance(self.min_features_split, int):
#             if self.min_features_split < 1:
#                 self.min_features_split = 1
#                 warn('min_features_split specified incorrectly; setting to default value of 1.')
#             min_features = self.min_features_split
#         elif isinstance(self.min_features_split, float):
#             if not 0. < self.min_features_split <= 1.:
#                 self.min_features_split = 1 / n_features
#                 warn('min_features_split not between [0, 1]; setting to default value of 1.')
#                 min_features = 1
#             else:
#                 min_features = int(np.ceil(self.min_features_split * n_features))
#         else:
#             warn('Invalid value for min_features_split; setting to default value of 1')
#             min_features = 1
#
#         return min_features
#
#     def fit(self, X, y):
#         # Ensure that X is a 2d array of shape (n_samples, n_features)
#         if X.ndim == 1:  # single feature from all examples
#             if len(y) > 1:
#                 X = X.reshape(-1, 1)
#             elif len(y) == 1:  # single training example
#                 X = X.reshape(1, -1)
#             else:
#                 ValueError('Invalid X and y')
#
#         if isinstance(self.impurity, str):
#             if self.impurity == 'gini':
#                 self.impurity = split_criteria.gini
#             elif self.impurity == 'twoing':
#                 self.impurity = split_criteria.twoing
#             elif self.impurity == 'sum_impurity':
#                 self.impurity = split_criteria.sum_impurity
#             else:
#                 raise ValueError('Unrecognized split impurity. Allowed: "gini", "twoing", "sum_impurity"')
#
#         n_samples, n_features = X.shape
#         min_samples_split = self.get_min_samples_split(n_samples)
#         min_features_split = self.get_min_features_split(n_features)
#
#         self.root_node, self.learned_depth = build_oblique_tree_oc1(X, y, is_classifier(self), self.impurity,
#                                                                     self.max_depth, min_samples_split,
#                                                                     min_features_split, rng=self.rng,
#                                                                     n_restarts=self.n_restarts,
#                                                                     max_stagnation=self.max_stagnation
#                                                                     )
#
#         # Create a tree object
#         self.tree_ = Tree(n_features=n_features, is_classifier=is_classifier(self))
#         self.tree_.set_root_node(self.root_node)
#         self.tree_.set_depth(self.learned_depth)
#
#     def predict(self, X):
#         y = self.tree_.root_node.predict(X)
#         y = np.array(y, dtype=int)
#         return y
#
#     def get_params(self, deep=False):
#         return {'max_depth': self.max_depth,
#                 'min_samples_split': self.min_samples_split,
#                 'impurity': self.impurity,
#                 'min_features_split': self.min_features_split,
#                 'n_restarts': self.n_restarts,
#                 }
#
#     def set_params(self, **parameters):
#         for parameter, value in parameters.items():
#             setattr(self, parameter, value)
#         return self
#
#
# # Definition of classes provided: ObliqueClassifier1
# class ObliqueClassifier1(ClassifierMixin, BaseObliqueTree):
#     def __init__(self, impurity="sum_impurity", max_depth=3, min_samples_split=2, min_features_split=1,
#                  random_state=None, n_restarts=10, max_stagnation=10, max_equal_accepts=10):
#         """
#         OC1 classifier entry point. Sets parameters and defers tree building to build_oblique_tree_oc1.
#
#         Parameters:
#         - impurity: str, either "sum_impurity", "gini" or "twoing", defines the split quality metric
#         - max_depth: int, maximum allowed depth of the tree
#         - min_samples_split: minimum samples required to allow a split
#         - min_features_split: minimum number of features needed to allow a split
#         - random_state: for reproducibility
#         - n_restarts: how many times to try different random hyperplanes per node
#         """
#         super().__init__(impurity=impurity, max_depth=max_depth, min_samples_split=min_samples_split,
#                          min_features_split=min_features_split, random_state=random_state,
#                          n_restarts=n_restarts, max_stagnation=max_stagnation, max_equal_accepts=max_equal_accepts)
#
#
# def build_oblique_tree_oc1(X, y, is_classification, impurity,
#                            max_depth, min_samples_split, min_features_split, rng=None,
#                            n_restarts=10, max_stagnation=10, max_equal_accepts=10,
#                            current_depth=0, current_features=None, current_samples=None, debug=False):
#     """
#     Build an OC1-style oblique decision tree recursively.
#
#     Implements:
#     - Multiple random initial hyperplanes (n_restarts)
#     - Coordinate-wise analytic perturbation using midpoint logic (one weight at a time)
#     - Accepts equal impurity moves with decaying probability (stag_prob = exp(-k))
#     - Applies one random vector perturbation to escape local minima using midpoint projections
#     - Terminates coordinate perturbation after fixed stagnation threshold
#     - Recursively constructs left and right branches based on best split
#
#     Parameters:
#     - X: ndarray, input features (n_samples, n_features)
#     - y: ndarray, target labels
#     - is_classification: bool, determines classification vs regression
#     - impurity: callable, impurity function (e.g., Gini or Twoing)
#     - max_depth, min_samples_split, min_features_split: stopping criteria
#     - rng: np.random.RandomState, seeded RNG for reproducibility
#     - n_restarts: number of random hyperplanes to try per node
#     - max_stagnation: max non-improving perturbations before stopping
#     - current_depth, current_features, current_samples: recursion tracking info
#
#     Returns:
#     - Node (internal or leaf), and the maximum depth of subtree
#     """
#     """
#     Build an OC1-style oblique decision tree recursively.
#
#     Implements:
#     - Multiple random initial hyperplanes (n_restarts)
#     - Coordinate-wise analytic perturbation using midpoint logic (one weight at a time)
#     - Accepts equal impurity moves with decaying probability (stag_prob = exp(-k))
#     - Applies one random vector perturbation to escape local minima using midpoint projections
#     - Terminates coordinate perturbation after fixed stagnation threshold
#     - Recursively constructs left and right branches based on best split
#
#     Returns:
#     - Node (internal or leaf), and the maximum depth of subtree
#     """
#
#     n_samples, n_features = X.shape
#     if current_depth == 0:
#         current_features = np.arange(n_features)
#         current_samples = np.arange(n_samples)
#
#     y = np.atleast_1d(y)
#
#     if is_classification:
#         if y.size == 1:
#             label = y[0]
#             conf = 1.0
#         else:
#             mode_result = mode(y, keepdims=True)
#             label = mode_result.mode[0]
#             conf = mode_result.count[0] / y.size
#     else:
#         std = np.std(y)
#         label = np.mean(y)
#         conf = np.sum((-std <= y) & (y <= std)) / X.shape[0]
#
#     if (current_depth == max_depth or n_samples <= min_samples_split or
#             n_features <= min_features_split or conf >= 0.95):
#         print(f"[Leaf] Depth {current_depth}: stopping with conf={conf:.3f}, samples={n_samples}")
#         return LeafNode(is_classifier=is_classification, value=label, conf=conf,
#                         samples=current_samples, features=current_features), current_depth
#
#     X_aug = np.hstack((X, np.ones((n_samples, 1))))
#     best_score = np.inf
#     best_wb = None
#
#     for restart in range(n_restarts):
#         print(f"[Restart {restart+1}/{n_restarts}] Starting new random hyperplane...")
#         w = rng.randn(n_features + 1)
#         w /= np.linalg.norm(w) + 1e-12
#         stagnation_count = 0
#         equal_accepts = 0
#
#         while stagnation_count < max_stagnation:
#             round_improvement = False
#             print(f"  Perturbation round, stagnation_count={stagnation_count}")
#
#             n_perturb_trials = 50  # R-50: 50 random coordinate perturbations per round
#             for _ in range(n_perturb_trials):
#                 j = rng.randint(0, n_features + 1)  # random coefficient index
#                 V = X_aug @ w
#                 with np.errstate(divide='ignore', invalid='ignore'):
#                     U = (V - X_aug[:, j] * w[j]) / X_aug[:, j]
#                 sort_idx = np.argsort(U)
#                 U_sorted, y_sorted = U[sort_idx], y[sort_idx]
#
#                 for i in range(len(U_sorted) - 1):
#                     if y_sorted[i] != y_sorted[i + 1]:
#                         midpoint = (U_sorted[i] + U_sorted[i + 1]) / 2
#                         w_new = w.copy()
#                         w_new[j] = midpoint
#                         margin = X_aug @ w_new
#                         left, right = y[margin <= 0], y[margin > 0]
#                         if len(left) == 0 or len(right) == 0:
#                             continue
#                         score = impurity(left, right)
#
#                         if score < best_score:
#                             print(f"    Improved: score {score:.5f} < {best_score:.5f}")
#                             best_score = score
#                             best_wb = w_new
#                             w = w_new
#                             stagnation_count = 0
#                             round_improvement = True
#
#                         elif score == best_score and equal_accepts < max_equal_accepts:
#                             prob = np.exp(-stagnation_count)
#                             if rng.rand() < prob:
#                                 print("    Equal score accepted probabilistically")
#                                 best_wb = w_new
#                                 w = w_new
#                                 stagnation_count += 1
#                                 equal_accepts += 1
#                                 round_improvement = False
#
#             if not round_improvement:
#                 stagnation_count += 1
#                 print("  No improvement from coordinate-wise: trying random direction fallback")
#                 direction = rng.randn(n_features + 1)
#                 direction /= np.linalg.norm(direction) + 1e-12
#                 proj = X_aug @ direction
#                 sort_idx = np.argsort(proj)
#                 proj_sorted = proj[sort_idx]
#                 proj_labels = y[sort_idx]
#
#                 for i in range(len(proj_sorted) - 1):
#                     if proj_labels[i] != proj_labels[i + 1]:
#                         midpoint = (proj_sorted[i] + proj_sorted[i + 1]) / 2
#                         w_perturbed = direction.copy()
#                         w_perturbed[-1] -= midpoint
#                         margin = X_aug @ w_perturbed
#                         left, right = y[margin <= 0], y[margin > 0]
#                         if len(left) == 0 or len(right) == 0:
#                             continue
#                         score = impurity(left, right)
#                         if score < best_score:
#                             print("    Random direction improved split")
#                             best_score = score
#                             best_wb = w_perturbed
#                             w = w_perturbed
#                             stagnation_count = 0
#                             break
#
#     if best_wb is None:
#         print(f"[Leaf] Depth {current_depth}: no split found")
#         return LeafNode(is_classifier=is_classification, value=label, conf=conf,
#                         samples=current_samples, features=current_features), current_depth
#
#     w, b = best_wb[:-1], best_wb[-1]
#     margin = X @ w + b
#     left, right = margin <= 0, margin > 0
#
#     if len(y[left]) == 0 or len(y[right]) == 0:
#         print(f"[Leaf] Depth {current_depth}: split failed, one side empty")
#         return LeafNode(is_classifier=is_classification, value=label, conf=conf,
#                         samples=current_samples, features=current_features), current_depth
#
#     print(f"[Split] Depth {current_depth}: proceeding with split, best_score={best_score:.5f}")
#     decision_node = Node(w, b, is_classifier=is_classification, value=label, conf=conf,
#                          samples=current_samples, features=current_features)
#
#     left_node, left_depth = build_oblique_tree_oc1(X[left], y[left], is_classification, impurity,
#                                                    max_depth, min_samples_split, min_features_split,
#                                                    rng=rng, n_restarts=n_restarts, max_stagnation=max_stagnation,
#                                                    current_depth=current_depth + 1,
#                                                    current_features=current_features,
#                                                    current_samples=current_samples[left])
#
#     decision_node.add_left_child(left_node)
#
#     right_node, right_depth = build_oblique_tree_oc1(X[right], y[right], is_classification, impurity,
#                                                      max_depth, min_samples_split, min_features_split,
#                                                      rng=rng, n_restarts=n_restarts, max_stagnation=max_stagnation,
#                                                      current_depth=current_depth + 1,
#                                                      current_features=current_features,
#                                                      current_samples=current_samples[right])
#
#     decision_node.add_right_child(right_node)
#
#     return decision_node, max(left_depth, right_depth)
#
#
# def get_best_splits(X, y, impurity=split_criteria.sum_impurity):
#     """
#     Evaluate the best axis-aligned splits for each feature.
#
#     Parameters:
#     - X: ndarray, shape (n_samples, n_features)
#     - y: target labels
#     - impurity: impurity function (callable)
#
#     Returns:
#     - all_splits: array of shape (n_features, 2) where each row holds (best_threshold, score)
#     """
#     n_samples, n_features = X.shape
#     all_splits = np.zeros((n_features, 2))
#
#     for f in range(n_features):
#         feature_values = np.sort(X[:, f])
#         feature_splits = np.convolve(feature_values, [0.5, 0.5])[1:-1]
#         scores = np.empty_like(feature_splits)
#         best_split = None
#         best_score = np.inf
#
#         for i, s in enumerate(feature_splits):
#             left, right = y[X[:, f] <= s], y[X[:, f] > s]
#             scores[i] = impurity(left, right)
#             if scores[i] < best_score:
#                 best_score = scores[i]
#                 best_split = s
#
#         all_splits[f, :] = [best_split, best_score]
#
#     return all_splits
