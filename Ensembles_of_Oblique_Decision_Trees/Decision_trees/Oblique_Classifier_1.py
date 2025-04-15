# Implementation of Oblique Classifier 1 (OC1) by [Murthy et al.]

# ......Importing all the packages............................
from warnings import warn
import numpy as np
from scipy.stats import mode
from Ensembles_of_Oblique_Decision_Trees.Decision_trees.OC1_tree_structure import Tree, Node, LeafNode
from Ensembles_of_Oblique_Decision_Trees.Decision_trees import split_criteria
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier


epsilon = 1e-6


# Definition of classes provided: ObliqueClassifier1
class BaseObliqueTree(BaseEstimator):

    def __init__(self, criterion, max_depth, min_samples_split, min_features_split, random_state=None,
                 n_restarts=20, bias_steps=20):

        # Get the options for tree learning
        self.criterion = criterion                      # splitting criterion - default is 'gini-index'
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

        if self.criterion == 'gini':
            self.criterion = split_criteria.gini
        elif self.criterion == 'twoing':
            self.criterion = split_criteria.twoing
        else:
            ValueError('Unrecognized split criterion specified. Allowed split criteria are:\n'
                       '[classification] "gini": Gini impurity, "twoing": Twoing rule')

        n_samples, n_features = X.shape
        min_samples_split = self.get_min_samples_split(n_samples)
        min_features_split = self.get_min_features_split(n_features)

        self.root_node, self.learned_depth = build_oblique_tree_oc1(
            X, y, is_classifier(self), self.criterion, self.max_depth, min_samples_split,
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
                'criterion': self.criterion,
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
    def __init__(self, criterion="gini", max_depth=3, min_samples_split=2, min_features_split=1,
                 random_state=None, n_restarts=5, bias_steps=5):
        super().__init__(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split,
                         min_features_split=min_features_split, random_state=random_state,
                         n_restarts=n_restarts, bias_steps=bias_steps)


# Implements Murthy et al (1994)'s algorithm to learn an oblique decision tree via random perturbations
def build_oblique_tree_oc1(X, y, is_classification, criterion,
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
    - criterion: impurity function (e.g. Gini, Twoing)
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
    feature_splits = get_best_splits(X, y, criterion=criterion)   # Get the best split for each feature
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
        base_score = criterion(left, right)

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

                    score = criterion(left, right)

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

    left_node, left_depth = build_oblique_tree_oc1(X[left], y[left], is_classification, criterion,
                                                   max_depth, min_samples_split, min_features_split,
                                                   rng=rng, n_restarts=n_restarts, bias_steps=bias_steps,
                                                   current_depth=current_depth + 1,
                                                   current_features=current_features,
                                                   current_samples=current_samples[left])

    decision_node.add_left_child(left_node)

    right_node, right_depth = build_oblique_tree_oc1(X[right], y[right], is_classification, criterion,
                                                     max_depth, min_samples_split, min_features_split,
                                                     rng=rng, n_restarts=n_restarts, bias_steps=bias_steps,
                                                     current_depth=current_depth + 1,
                                                     current_features=current_features,
                                                     current_samples=current_samples[right])

    decision_node.add_right_child(right_node)

    return decision_node, max(left_depth, right_depth)


# Get the best splitting threshold for each feature/attribute by considering them independently of the others
def get_best_splits(X, y, criterion=split_criteria.gini):
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
            scores[i] = criterion(left, right)
            if scores[i] < best_score:
                best_score = scores[i]
                best_split = s

        all_splits[f, :] = [best_split, best_score]

    return all_splits


# # Implementation of Oblique Classifier 1 (OC1) by [Murthy et al.]
#
# # ......Importing all the packages............................
# from warnings import warn
# import numpy as np
# from scipy.stats import mode
# from Ensembles_of_Oblique_Decision_Trees.Decision_trees.OC1_tree_structure import Tree, Node, LeafNode
# from Ensembles_of_Oblique_Decision_Trees.Decision_trees import split_criteria
# from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
#
#
# epsilon = 1e-6
#
#
# # Definition of classes provided: ObliqueClassifier1
# class BaseObliqueTree(BaseEstimator):
#
#     def __init__(self, criterion, max_depth, min_samples_split, min_features_split, random_state=None):
#
#         # Get the options for tree learning
#         self.criterion = criterion                      # splitting criterion - default is 'gini-index'
#         self.max_depth = max_depth                      # maximum depth of the tree
#         self.min_samples_split = min_samples_split      # minimum number of samples needed for a split
#         self.min_features_split = min_features_split    # minimum number of features needed for a split
#         self.tree_ = None                               # Internal tree - initially set as 'None'
#         self.random_state = random_state
#         self.rng = np.random.RandomState(random_state) if random_state is not None else np.random
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
#         X = X
#         y = y
#         # Ensure that X is a 2d array of shape (n_samples, n_features)
#         if X.ndim == 1:                 # single feature from all examples
#             if len(y) > 1:
#                 X = X.reshape(-1, 1)
#             elif len(y) == 1:           # single training example
#                 X = X.reshape(1, -1)
#             else:
#                 ValueError('Invalid X and y')
#
#         if self.criterion == 'gini':
#             self.criterion = split_criteria.gini
#         elif self.criterion == 'twoing':
#             self.criterion = split_criteria.twoing
#         else:
#             ValueError('Unrecognized split criterion specified. Allowed split criteria are:\n'
#                        '[classification] "gini": Gini impurity, "twoing": Twoing rule')
#
#         n_samples, n_features = X.shape
#         min_samples_split = self.get_min_samples_split(n_samples)
#         min_features_split = self.get_min_features_split(n_features)
#
#         # Build a tree and get its root node
#         self.root_node, self.learned_depth = build_oblique_tree_oc1(X, y, is_classifier(self), self.criterion,
#                                                                     self.max_depth, min_samples_split,
#                                                                     min_features_split, rng=self.rng)
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
#                 'criterion': self.criterion, 'min_features_split': self.min_features_split}
#
#     def set_params(self, **parameters):
#         for parameter, value in parameters.items():
#             setattr(self, parameter, value)
#         return self
#
#
# # Definition of classes provided: ObliqueClassifier1
# class ObliqueClassifier1(ClassifierMixin, BaseObliqueTree):
#     def __init__(self, criterion="gini", max_depth=3, min_samples_split=2, min_features_split=1, random_state=None):
#         super().__init__(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split,
#                          min_features_split=min_features_split,
#                          random_state=random_state)
#
#
# # Implements Murthy et al (1994)'s algorithm to learn an oblique decision tree via random perturbations
# def build_oblique_tree_oc1(X, y, is_classification, criterion,
#                            max_depth, min_samples_split, min_features_split, rng=None,
#                            current_depth=0, current_features=None, current_samples=None, debug=False):
#
#     n_samples, n_features = X.shape
#
#     # Initialize
#     if current_depth == 0:
#         current_features = np.arange(n_features)
#         current_samples = np.arange(n_samples)
#
#     # Ensure y is at least a 1D array
#     y = np.atleast_1d(y)
#
#     if is_classification:
#         # If there is only one sample, use it directly.
#         if y.size == 1:
#             label = y[0]
#             conf = 1.0
#         else:
#             mode_result = mode(y, keepdims=True)  # ensures the result is an array
#             majority = mode_result.mode  # should be an array of shape (1,) now
#             count = mode_result.count
#             label = majority[0]
#             conf = count[0] / y.size
#     else:
#         std = np.std(y)
#         label = np.mean(y)
#         conf = np.sum((-std <= y) & (y <= std)) / X.shape[0]
#
#     # Check termination criteria, and return a leaf node if terminating
#     if (current_depth == max_depth or            # max depth reached
#             n_samples <= min_samples_split or    # not enough samples to split on
#             n_features <= min_features_split or  # not enough features to split on
#             conf >= 0.95):                       # node is very homogeneous
#
#         return LeafNode(is_classifier=is_classification, value=label, conf=conf,
#                         samples=current_samples, features=current_features), current_depth
#
#     # Otherwise, learn a decision node
#     feature_splits = get_best_splits(X, y, criterion=criterion)   # Get the best split for each feature
#     f = np.argmin(feature_splits[:, 1])                           # Find the best feature to split on
#     best_split_score = feature_splits[f, 1]                       # Save the score corresponding to the best split
#     w, b = np.eye(1, n_features, f).squeeze(), -feature_splits[f, 0]    # Construct a (w, b) from the best split
#                                                                   # X[f] <= s becomes 1. X[f] + 0. X[rest] - s <= 0
#
#     stagnant = 0                                                  # Used to track stagnation probability (see below)
#     for k in range(5):                                            # Randomly attempt to perturb a feature
#         m = rng.randint(0, n_features)                            # Select a random feature (weight w[m]) to update
#         idx = np.where(X[:, m] == 0)[0]
#         if len(idx) != 0:
#             X[idx, m] = epsilon
#
#         wNew = np.array(w)                                          # Initialize wNew to w
#         margin = (np.dot(X, wNew) + b)                              # Compute the signed margin of all training examples
#         u = (X[:, m]*w[m] - margin) / X[:, m]                       # Compute the residual of all training examples
#
#         possible_wm = np.convolve(np.sort(u), [0.5, 0.5])[1:-1]  # Generate a list of possible values for new w[m]
#         scores = np.empty_like(possible_wm)
#         best_wm, best_wm_score = 0, np.inf                          # Find the best value for w[m]
#         i = 0
#         for wm in possible_wm:
#             wNew[m] = wm                                                # Try (w = [w0, ..., wm, ..., wd], b) as a split
#             margin = (np.dot(X, wNew) + b)                              # Signed margin of examples using this split
#             left, right = y[margin <= 0], y[margin > 0]                 # Partition of examples using this split
#             wm_score = criterion(left, right)                           # Score of this split
#             scores[i] = wm_score
#             i += 1
#
#             if wm_score < best_wm_score:                                # Save the best split
#                 best_wm_score = wm_score
#                 best_wm = wm
#
#         # Once the best w[m] among possible u has been identified, check if its actually any good
#         if best_wm_score < best_split_score:
#             # If we have identified a split with a better score, update it
#             best_split_score = best_wm_score
#             w[m] = best_wm
#             stagnant = 0
#         elif np.abs(best_wm_score - best_split_score) < 1e-3:
#             # If we have identified a split with a similar score, update it with probability P(update) = exp(-stagnant)
#             # Stagnation prob. is the probability that the perturbation does not improve the score. To prevent the
#             # impurity from remaining stagnant for a long time, the stag. prob decreases exponentially with the number
#             # of stagnant perturbations. It is reset to 1 every time the global impurity measure is improved
#             if rng.rand() <= np.exp(-stagnant):
#                 best_split_score = best_wm_score
#                 w[m] = best_wm
#                 stagnant += 1
#
#         # If we have achieved a fantastic split, stop immediately
#         if best_split_score < 1e-3:
#             break
#
#     # .................... Validation .........................
#     idx = np.where(w == np.inf)[0]
#     if len(idx) != 0:
#         w[idx] = rng.rand(len(idx))
#     idx = np.where(w == -np.inf)[0]
#     if len(idx) != 0:
#         w[idx] = -1 * (rng.rand(len(idx)))
#     if b == np.inf:
#         b = 10
#     if b == -np.inf:
#         b = -10
#
#     # Now that a split has been found, perform a final partition
#     margin = np.dot(X, w) + b
#     left, right = margin <= 0, margin > 0
#     if len(y[left]) == 0:
#         return LeafNode(is_classifier=is_classification, value=label, conf=conf,
#                         samples=current_samples, features=current_features), current_depth
#
#     elif len(y[right]) == 0:
#         return LeafNode(is_classifier=is_classification, value=label, conf=conf,
#                         samples=current_samples, features=current_features), current_depth
#
#     else:
#
#         # Create a decision node
#         decision_node = Node(w, b, is_classifier=is_classification, value=label, conf=conf,
#                              samples=current_samples, features=current_features)
#
#         # Grow the left branch and insert it
#         left_node, left_depth = build_oblique_tree_oc1(X[left, :], y[left],
#                                                        is_classification, criterion,
#                                                        max_depth, min_samples_split, min_features_split,
#                                                        rng=rng,
#                                                        current_depth=current_depth + 1,
#                                                        current_features=current_features,
#                                                        current_samples=current_samples[left])
#
#         decision_node.add_left_child(left_node)
#
#         # Grow the right branch and insert it
#         right_node, right_depth = build_oblique_tree_oc1(X[right, :], y[right],
#                                                          is_classification, criterion,
#                                                          max_depth, min_samples_split, min_features_split,
#                                                          rng=rng,
#                                                          current_depth=current_depth + 1,
#                                                          current_features=current_features,
#                                                          current_samples=current_samples[right])
#
#         decision_node.add_right_child(right_node)
#
#         return decision_node, max(left_depth, right_depth)
#
#
# # Get the best splitting threshold for each feature/attribute by considering them independently of the others
# def get_best_splits(X, y, criterion=split_criteria.gini):
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
#         # Compute the scores
#         for i, s in enumerate(feature_splits):
#             left, right = y[X[:, f] <= s], y[X[:, f] > s]
#             scores[i] = criterion(left, right)
#             if scores[i] < best_score:
#                 best_score = scores[i]
#                 best_split = s
#
#         all_splits[f, :] = [best_split, best_score]
#
#     return all_splits
