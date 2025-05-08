# Implementation of Weighted Oblique Decision Trees by [Bin-Bin Yang et al.]
# ````Some part of the code has been shared by the author````

# .......Importing all the packages.............
import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier

# Global constants for numeric stability
EPSILON_EPSILON = 1e-220
EPSILON = 1e-50


class SplitQuestion:
    """
    Class for splitting the dataset based on an oblique hyperplane:
      dot(x[attrIDs], paras) <= threshold
    """

    def __init__(self, attrIDs=None, paras=None, threshold=0.0):
        # Avoid mutable defaults by checking None
        if attrIDs is None:
            attrIDs = [0]
        if paras is None:
            paras = [0]

        self.attrIDs = attrIDs
        self.paras = paras
        self.threshold = threshold

    def test_for_one_instance(self, x):
        """
        Returns True if dot(x[attrIDs], paras) <= threshold, False otherwise.
        """
        return np.dot(x[self.attrIDs], self.paras) <= self.threshold

    def test(self, X):
        """
        Returns a boolean array indicating which rows in X go to the left child.
        """
        return np.dot(X[:, self.attrIDs], self.paras) <= self.threshold


class Node:
    """
    A node in the Weighted Oblique Decision Tree.
    """

    def __init__(self, depth, split, sample_ids, X, Y, class_num, rng, node_seed=None):
        self.depth = depth
        self.split = split
        self.sample_ids = sample_ids
        self.X = X
        self.Y = Y
        self.class_num = class_num
        self.rng = np.random.RandomState(node_seed) if node_seed is not None else rng
        self.rng_seed = node_seed

        # Initialize attributes that will be set later
        self.is_leaf = False
        self.LChild = None
        self.RChild = None
        self.class_distribution = None

    def find_best_split(self, max_features='sqrt', max_iter=1000):
        feature_num = self.X.shape[1]
        # Determine how many features to consider
        if max_features == 'sqrt':
            subset_feature_num = int(np.sqrt(feature_num))
        elif max_features == 'all':
            subset_feature_num = feature_num
        elif max_features == 'log':
            subset_feature_num = int(np.log2(feature_num))
        elif isinstance(max_features, int):
            subset_feature_num = max_features
        elif isinstance(max_features, float):
            subset_feature_num = int(feature_num * max_features)
        else:
            subset_feature_num = feature_num

        # Choose a random subset of features
        feature_ids = range(feature_num)
        subset_feature_ids = self.rng.choice(
            list(feature_ids), size=subset_feature_num, replace=False
        ).tolist()
        # Update the split to record which features are used
        self.split.attrIDs = subset_feature_ids
        # ← Fix: Convert the list to a NumPy array for advanced indexing
        subset_feature_ids = np.array(subset_feature_ids)

        # Prepare the sub matrix of features
        sub_features_x = self.X[self.sample_ids[:, None], subset_feature_ids[None, :]]
        sub_labels = self.Y[self.sample_ids]

        def func(a):
            """
            Objective function for weighted entropy:
              w_L_sum * log2(w_L_sum) + w_R_sum * log2(w_R_sum)
              - sum(w_L_eachClass * log2(w_L_eachClass))
              - sum(w_R_eachClass * log2(w_R_eachClass))
            """
            threshold = a[0]
            paras = a[1:]
            p = sigmoid(np.dot(sub_features_x, paras) - threshold)
            w_r = p
            w_l = 1.0 - w_r

            w_r_sum = np.sum(w_r)
            w_l_sum = np.sum(w_l)

            w_r_each_class = np.array([np.sum(w_r[sub_labels == k]) for k in range(self.class_num)])
            w_l_each_class = np.array([np.sum(w_l[sub_labels == k]) for k in range(self.class_num)])

            return (w_l_sum * np.log2(w_l_sum + EPSILON_EPSILON)
                    + w_r_sum * np.log2(w_r_sum + EPSILON_EPSILON)
                    - np.sum(w_r_each_class * np.log2(w_r_each_class + EPSILON_EPSILON))
                    - np.sum(w_l_each_class * np.log2(w_l_each_class + EPSILON_EPSILON)))

        def func_gradient(a):
            """
            Gradient of the objective function with respect to 'threshold' and 'paras'.
            """
            threshold = a[0]
            paras = a[1:]

            p = sigmoid(np.dot(sub_features_x, paras) - threshold)
            w_r = p
            w_l = 1.0 - w_r

            w_r_each_class = np.array([np.sum(w_r[sub_labels == k]) for k in range(self.class_num)])
            w_l_each_class = np.array([np.sum(w_l[sub_labels == k]) for k in range(self.class_num)])

            # la ~ log-ratio term for each instance
            # slight re-interpretation of the partial derivatives in Yang et al. (2019)
            la = (np.log2(w_l_each_class[sub_labels] * w_r.sum() + EPSILON_EPSILON)
                  - np.log2(w_r_each_class[sub_labels] * w_l.sum() + EPSILON_EPSILON))
            beta = la * p * (1.0 - p)

            jac = np.zeros_like(a)
            # Partial derivative w.r.t. threshold => negative sum of beta
            jac[0] = - np.sum(beta)
            # Partial derivative w.r.t. paras => dot(sub_features_x.T, beta)
            jac[1:] = np.dot(sub_features_x.T, beta)

            return jac

        # Randomly initialize [threshold, paras...]
        initial_a = self.rng.rand(subset_feature_num + 1) - 0.5
        result = minimize(
            func,
            initial_a,
            method='L-BFGS-B',
            jac=func_gradient,
            options={'maxiter': max_iter, 'disp': False}
        )

        # Store the optimized threshold and parameters
        self.split.threshold = result.x[0]
        self.split.paras = result.x[1:]
        return 1

    def grow_stump(self):
        """
        Partitions the data into left and right sets using the learned split.
        """
        left_bool = self.split.test(self.X[self.sample_ids])
        l_sample_ids = self.sample_ids[left_bool]
        r_sample_ids = self.sample_ids[~left_bool]

        self.LChild = Node(
            self.depth + 1, SplitQuestion(), l_sample_ids,
            self.X, self.Y, self.class_num,
            rng=None,
            node_seed=hash((self.rng_seed, self.depth + 1, 0)) % (2 ** 32)
        )
        self.RChild = Node(
            self.depth + 1, SplitQuestion(), r_sample_ids,
            self.X, self.Y, self.class_num,
            rng=None,
            node_seed=hash((self.rng_seed, self.depth + 1, 1)) % (2 ** 32)
        )

        # If one side is empty, label that child as leaf using current node's distribution
        if len(l_sample_ids) == 0:
            self.LChild.is_leaf = True
            self.LChild.class_distribution = compute_class_distribution(
                self.Y[self.sample_ids], self.class_num
            )
        if len(r_sample_ids) == 0:
            self.RChild.is_leaf = True
            self.RChild.class_distribution = compute_class_distribution(
                self.Y[self.sample_ids], self.class_num
            )


class BaseObliqueTree(BaseEstimator):
    """
    Base class that implements the recursive building
    of a Weighted Oblique Decision Tree.
    """

    def __init__(self, max_depth=50, min_samples_split=2,
                 max_features='all', random_state=None, max_iter=1000):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.max_iter = max_iter

        # Use np.random.RandomState to control randomness consistently
        self.rng = np.random.RandomState(random_state)

        # Will be set during fitting
        self.X = None
        self.Y = None
        self.classNum = None
        self.sampleNum = None
        self.root_node = None
        self.leaf_num = 0
        self.tree_depth = 0

    def fit(self, X, Y):
        """
        Fit the oblique decision tree to data (X, Y).
        """
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        self.classNum = self.Y.max() + 1
        self.sampleNum = self.X.shape[0]

        root_seed = hash((self.random_state, 0)) % (2 ** 32)
        self.root_node = Node(
            depth=0,
            split=SplitQuestion(),
            sample_ids=np.arange(self.sampleNum, dtype=np.uint32),
            X=self.X,
            Y=self.Y,
            class_num=self.classNum,
            rng=None,  # will be overridden
            node_seed=root_seed
        )
        self.leaf_num = 1
        self.tree_depth = self.build_subtree(self.root_node, is_classifier(self))

    def build_subtree(self, node, is_classification):
        """
        Recursively grows the tree until stopping conditions are met.
        """
        if node.is_leaf:
            return node.depth

        # Stopping conditions
        if (node.depth >= self.max_depth or
                len(node.sample_ids) < self.min_samples_split or
                is_all_equal(self.Y[node.sample_ids])):
            node.is_leaf = True
            if is_classification:
                node.class_distribution = compute_class_distribution(
                    self.Y[node.sample_ids], self.classNum
                )
            return node.depth

        # If best split fails (returns < 0), also mark as leaf
        if node.find_best_split(self.max_features, self.max_iter) < 0:
            node.is_leaf = True
            if is_classification:
                node.class_distribution = compute_class_distribution(
                    self.Y[node.sample_ids], self.classNum
                )
            return node.depth

        # Grow the stump (create left and right children)
        node.grow_stump()
        node.is_leaf = False
        self.leaf_num += 1

        # Recurse into children, track max depth
        left_subtree_depth = self.build_subtree(node.LChild, is_classification)
        right_subtree_depth = self.build_subtree(node.RChild, is_classification)
        return max(left_subtree_depth, right_subtree_depth)

    def predict_for_one_instance(self, x):
        """
        Predict the class label for a single instance x.
        """
        present_node = self.root_node
        while not present_node.is_leaf:
            if present_node.split.test_for_one_instance(x):
                present_node = present_node.LChild
            else:
                present_node = present_node.RChild
        return np.argmax(present_node.class_distribution)

    def predict(self, X):
        """
        Predict class labels for all instances in X.
        """
        m = X.shape[0]
        y_predicted = np.zeros(m, dtype=int)
        for i in range(m):
            y_predicted[i] = self.predict_for_one_instance(X[i])
        return y_predicted

####################
# Functions
# These functions are used in the class methods above


def sigmoid(z):
    """
    Sigmoid function with safeguards against numerical overflow.
    """
    # Because large negative z can cause runtime warnings in np.exp()
    if isinstance(z, float) and z < -500:
        z = -500
    elif not isinstance(z, float):
        z[z < -500] = -500
    return 1.0 / (np.exp(-z) + 1.0)


def is_all_equal(x):
    """
    Checks if all entries in x are the same.
    """
    x_min, x_max = x.min(), x.max()
    return x_min == x_max


def compute_class_distribution(Y, class_num):
    """
    Computes the class distribution (proportion of each class)
    in the set of labels Y.
    """
    sample_num = len(Y)
    ratio_each_class = [np.sum(Y == k) / sample_num for k in range(class_num)]
    return np.array(ratio_each_class)


class WeightedObliqueDecisionTreeClassifier(ClassifierMixin, BaseObliqueTree):
    """
    A Weighted Oblique Decision Tree classifier based on:
      Yang, Bin-Bin, et al. "Weighted Oblique Decision Trees."
      AAAI Conference on Artificial Intelligence. 2019.
    """
    def __init__(self, max_depth=50, min_samples_split=2,
                 max_features='all', random_state=None, max_iter=1000):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            random_state=random_state,
            max_iter=max_iter
        )
