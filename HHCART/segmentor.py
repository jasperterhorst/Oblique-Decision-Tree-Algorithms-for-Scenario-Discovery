# segmentor.py
import numpy as np
from abc import ABC, abstractmethod
from _adopted_oblique_trees.split_criteria import gini


class SegmentorBase(ABC):
    """
    Abstract base class for generating and evaluating axis-aligned splits.
    """
    def __init__(self, min_samples_leaf=2):
        self.min_samples_leaf = min_samples_leaf

    @abstractmethod
    def _split_generator(self, X):
        """
        Yield tuples (left_indices, right_indices, split_rule).
        split_rule is a tuple (feature_index, threshold).
        """
        pass

    def __call__(self, X, y, impurity=gini):
        """
        Search over generated splits, evaluate using `impurity`,
        and return (best_impurity, best_split_rule, left_idx, right_idx).
        """
        best_imp = np.inf
        best_rule = None
        best_left = best_right = None
        for left_idx, right_idx, rule in self._split_generator(X):
            if len(left_idx) > self.min_samples_leaf and len(right_idx) > self.min_samples_leaf:
                imp = impurity(y[left_idx], y[right_idx])
                if imp < best_imp:
                    best_imp = imp
                    best_rule = rule
                    best_left, best_right = left_idx, right_idx
        return best_imp, best_rule, best_left, best_right


class MeanSegmentor(SegmentorBase):
    """
    Split each feature at its mean value.
    """
    def _split_generator(self, X):
        for i in range(X.shape[1]):
            vals = X[:, i]
            thr = np.mean(vals)
            left = np.nonzero(vals < thr)[0]
            right = np.nonzero(vals >= thr)[0]
            yield left, right, (i, thr)


class CARTSegmentor(SegmentorBase):
    """
    Evaluate all midpoints between sorted unique feature values (classic CART).
    """
    def _split_generator(self, X):
        for i in range(X.shape[1]):
            unique_vals = np.unique(X[:, i])
            for j in range(1, len(unique_vals)):
                thr = 0.5 * (unique_vals[j-1] + unique_vals[j])
                left = np.nonzero(X[:, i] < thr)[0]
                right = np.nonzero(X[:, i] >= thr)[0]
                yield left, right, (i, thr)
