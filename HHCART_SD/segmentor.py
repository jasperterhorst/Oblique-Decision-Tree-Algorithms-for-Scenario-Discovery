"""
Axis-Aligned Split Segmentors (segmentor.py)
--------------------------------------------
Provides a segmentor for generating axis-aligned binary splits following
the CART (Classification and Regression Tree) strategy. All segmentors inherit
from a common abstract base for consistent API and impurity-based evaluation.

Implements:
- CARTSegmentor: Midpoint thresholding between sorted unique values per feature.
"""

import numpy as np
from abc import ABC, abstractmethod
from _adopted_oblique_trees.split_criteria import gini


class SegmentorBase(ABC):
    """
    Abstract base class for generating and evaluating axis-aligned splits.

    Attributes:
        min_samples_leaf (int): Minimum number of samples allowed in each child node.
    """

    def __init__(self, min_samples_leaf: int = 2):
        """
        Initialise the base segmentor class.

        Args:
            min_samples_leaf (int): Minimum number of samples in each split subset.
        """
        self.min_samples_leaf = min_samples_leaf

    @abstractmethod
    def _split_generator(self, X: np.ndarray):
        """
        Generator that yields valid binary splits over the input feature matrix.

        Args:
            X (np.ndarray): Feature matrix (n_samples × n_features).

        Yields:
            Tuple[np.ndarray, np.ndarray, Tuple[int, float]]:
                - Indices of left and right splits.
                - Tuple of (feature index, split threshold).
        """
        pass

    def __call__(self, X: np.ndarray, y: np.ndarray, impurity=gini):
        """
        Search through generated splits and return the one with minimum impurity.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Binary class labels.
            impurity (callable): Impurity metric (e.g., Gini).

        Returns:
            Tuple[float, Tuple[int, float], np.ndarray, np.ndarray]:
                - Best impurity score
                - Best split rule (feature index, threshold)
                - Indices of left and right groups
        """
        best_imp = np.inf
        best_rule = None
        best_left = best_right = None

        for left_idx, right_idx, rule in self._split_generator(X):
            # Enforce minimum leaf size
            if len(left_idx) <= self.min_samples_leaf or len(right_idx) <= self.min_samples_leaf:
                continue

            imp = impurity(y[left_idx], y[right_idx])
            if imp < best_imp:
                best_imp = imp
                best_rule = rule
                best_left, best_right = left_idx, right_idx

        return best_imp, best_rule, best_left, best_right


class CARTSegmentor(SegmentorBase):
    """
    Implements classic CART splitting: all midpoints between unique feature values.

    This segmentor searches all thresholds of the form:
        threshold = 0.5 * (v[i] + v[i+1])
    where v are sorted unique values in a feature column.
    """

    def _split_generator(self, X: np.ndarray):
        """
        Generate candidate splits by computing midpoints of sorted unique values.

        Args:
            X (np.ndarray): Feature matrix.

        Yields:
            Tuple[np.ndarray, np.ndarray, Tuple[int, float]]:
                - Indices for left and right splits.
                - Rule as (feature index, threshold).
        """
        for i in range(X.shape[1]):
            unique_vals = np.unique(X[:, i])
            for j in range(1, len(unique_vals)):
                thr = 0.5 * (unique_vals[j - 1] + unique_vals[j])
                left = np.nonzero(X[:, i] < thr)[0]
                right = np.nonzero(X[:, i] >= thr)[0]
                yield left, right, (i, thr)
