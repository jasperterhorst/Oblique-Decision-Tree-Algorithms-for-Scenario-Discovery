# Implementation of Oblique Classifier 1 (OC1) by [Murthy et al.]

# ......Importing all the packages............................
from warnings import warn
import numpy as np
from scipy.stats import mode
from _adopted_oblique_trees.OC1_tree_structure import Tree, Node, LeafNode
from _adopted_oblique_trees import split_criteria
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier


epsilon = 1e-6


# Definition of classes provided: ObliqueClassifier1
class BaseObliqueTree(BaseEstimator):

    def __init__(self, impurity, max_depth, min_samples_split, min_features_split, random_state=None,
                 n_restarts=20, bias_steps=20):

        # Get the options for tree learning
        self.impurity = impurity                      # splitting impurity - default is 'gini-index'
        self.max_depth = max_depth                      # maximum depth of the tree
        self.min_samples_split = min_samples_split      # minimum number of samples needed for a split
        self.min_features_split = min_features_split    # minimum number of features needed for a split
        self.tree_ = None                               # Internal tree - initially set as 'None'
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
        if X.ndim == 1:                 # single feature from all examples
            if len(y) > 1:
                X = X.reshape(-1, 1)
            elif len(y) == 1:           # single training example
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

        self.root_node, self.learned_depth = build_oblique_tree_oc1(
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
class ObliqueClassifier1(ClassifierMixin, BaseObliqueTree):
    def __init__(self, impurity="gini", max_depth=3, min_samples_split=2, min_features_split=1,
                 random_state=None, n_restarts=5, bias_steps=5):
        super().__init__(impurity=impurity, max_depth=max_depth, min_samples_split=min_samples_split,
                         min_features_split=min_features_split, random_state=random_state,
                         n_restarts=n_restarts, bias_steps=bias_steps)


# Implements Murthy et al (1994)'s algorithm to learn an oblique decision tree via random perturbations
def build_oblique_tree_oc1(X, y, is_classification, impurity,
                           max_depth, min_samples_split, min_features_split, rng=None,
                           n_restarts=5, bias_steps=5, current_depth=0, current_features=None,
                           current_samples=None, debug=False):
    """
    Builds an OC1-style Oblique Decision Tree.

    Implements:
    - Random restarts of linear separators
    - Oblique split by weight perturbation (one feature at a time)
    - Bias term adjustment
    - Stagnation-driven acceptance of perturbations (P=exp(-k))

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

    y = np.atleast_1d(y)      # Ensure y is at least a 1D array

    # ------------------ LEAF LABEL -------------------------
    # Estimate the label of this node
    if is_classification:
        # If there is only one sample, use it directly.
        if y.size == 1:
            label = y[0]
            conf = 1.0
        else:
            mode_result = mode(y, keepdims=True)  # ensures the result is an array
            majority = mode_result.mode  # should be an array of shape (1,) now
            count = mode_result.count
            label = majority[0]
            conf = count[0] / y.size
    else:
        std = np.std(y)
        label = np.mean(y)
        conf = np.sum((-std <= y) & (y <= std)) / X.shape[0]

    # ------------------ TERMINATION CHECK -------------------------
    if (current_depth == max_depth or            # max depth reached
        n_samples <= min_samples_split or        # not enough samples to split on
        n_features <= min_features_split or      # not enough features to split on
        conf >= 0.95):                           # node is very homogeneous

        return LeafNode(is_classifier=is_classification, value=label, conf=conf,
                        samples=current_samples, features=current_features), current_depth

    # ------------------ INIT SPLIT FROM AXIS -------------------------
    feature_splits = get_best_splits(X, y, impurity=impurity)   # Get the best split for each feature
    f = np.argmin(feature_splits[:, 1])                           # Find the best feature to split on
    best_split_score = feature_splits[f, 1]                       # Save the score corresponding to the best split
    w, b = np.eye(1, n_features, f).squeeze(), -feature_splits[f, 0]    # Construct a (w, b) from the best split
    # X[f] <= s becomes 1. X[f] + 0. X[rest] - s <= 0

    # ------------------ OC1 PERTURBATION SEARCH -------------------------
    # This block performs the true OC1 split selection with:
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

    left_node, left_depth = build_oblique_tree_oc1(X[left], y[left], is_classification, impurity,
                                                   max_depth, min_samples_split, min_features_split,
                                                   rng=rng, n_restarts=n_restarts, bias_steps=bias_steps,
                                                   current_depth=current_depth + 1,
                                                   current_features=current_features,
                                                   current_samples=current_samples[left])

    decision_node.add_left_child(left_node)

    right_node, right_depth = build_oblique_tree_oc1(X[right], y[right], is_classification, impurity,
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
