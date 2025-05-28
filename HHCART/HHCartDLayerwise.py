"""
HHCartDLayerwiseClassifier: Layer-wise Oblique Decision Tree with Householder Reflections

This classifier implements the breadth-first (layer-wise) version of HHCART(D),
constructing the tree level by level. It reflects feature space using class-specific
Householder transformations and selects axis-parallel splits in the transformed space.

Features:
- Layer-wise tree building with early stopping
- Per-depth metric evaluation
- Returns interpretable DecisionTree
- No post-pruning required
- Compatible with visualization, export, and selection tools
"""

from typing import Callable
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import deque
from scipy.linalg import norm
from copy import deepcopy
from sklearn.base import BaseEstimator, ClassifierMixin

from HHCART.segmentor import CARTSegmentor
from HHCART.evaluator import evaluate_tree
from HHCART.tree import DecisionTree, DecisionNode, LeafNode


class HHCartDLayerwiseClassifier(BaseEstimator, ClassifierMixin):
    """
    Householder CART(D) Classifier (layer-wise version)

    Parameters:
        impurity (Callable): Impurity function to use (e.g., 'gini', 'entropy').
        segmentor (CARTSegmentor): Split segmentor in reflected space.
        max_depth (int): Maximum allowed tree depth.
        min_samples_split (int): Minimum samples required to consider a split.
        min_purity (float): Threshold purity above which node becomes a leaf.
        tau (float): Numerical tolerance for reflection vector.
        random_state (int or None): Random seed for reproducibility.
        debug (bool): If True, prints debug output.
    """

    def __init__(self,
                 impurity: Callable,
                 segmentor=CARTSegmentor(),
                 max_depth: int = 5,
                 min_samples_split: int = 10,
                 min_purity: float = 0.95,
                 tau: float = 1e-4,
                 random_state: int = None,
                 debug: bool = False):
        self.impurity = impurity
        self.segmentor = segmentor
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_purity = min_purity
        self.tau = tau
        self.random_state = random_state
        self.debug = debug

        self.trees_by_depth = {}
        self.metrics_by_depth = {}
        self.tree = None
        self.variable_names = None

    def _should_stop(self, y: np.ndarray, depth: int) -> bool:
        """Return True if splitting should stop at this node."""
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if len(y) < self.min_samples_split:
            return True
        _, counts = np.unique(y, return_counts=True)
        purity = np.max(counts) / len(y)
        return purity >= self.min_purity

    def _compute_reflection_and_split(self, X: np.ndarray, y: np.ndarray):
        """Find best Householder reflection and corresponding split."""
        n_features = X.shape[1]
        best_H = np.eye(n_features)
        best_impurity = float('inf')
        best_split = None

        for c in np.unique(y):
            X_class = X[y == c]
            if X_class.shape[0] <= 1:
                continue

            covariance = np.cov(X_class, rowvar=False)
            _, eigvecs = np.linalg.eigh(covariance)
            mu = eigvecs[:, -1]

            if np.allclose(mu, 0):
                continue

            deviation = np.sqrt(((np.eye(n_features) - mu) ** 2).sum(axis=1))
            if (deviation > self.tau).sum() > 0:
                i = np.argmax(deviation)
                e = np.zeros(n_features)
                e[i] = 1.0
                w = (e - mu) / norm(e - mu)
                H = np.eye(n_features) - 2 * np.outer(w, w)

                X_reflected = X @ H
                impurity, split_rule, _, _ = self.segmentor(X_reflected, y, self.impurity)

                if impurity < best_impurity:
                    best_H = H
                    best_impurity = impurity
                    best_split = split_rule

        return best_H, best_split

    def _make_leaf(self, y: np.ndarray, depth: int, node_id: int):
        label = np.bincount(y).argmax()
        purity = np.max(np.bincount(y)) / len(y)
        return LeafNode(node_id=node_id, prediction=label, depth=depth,
                        n_samples=len(y), purity=purity)

    def _make_split_node(self, X: np.ndarray, y: np.ndarray, depth: int, node_id: int):
        """Attempt to create a split node, fallback to leaf if unsuccessful."""
        H, split_rule = self._compute_reflection_and_split(X, y)
        if split_rule is None:
            return self._make_leaf(y, depth, node_id), None, None

        i, threshold = split_rule
        weights = H[:, i]
        bias = -threshold
        node = DecisionNode(node_id=node_id, weights=weights, bias=bias, depth=depth)
        return node, weights, bias

    def fit(self, X, y):
        """Train the layer-wise HHCART(D) classifier."""
        if isinstance(X, pd.DataFrame):
            self.variable_names = list(X.columns)
            X = X.values
        else:
            self.variable_names = [f"x{i}" for i in range(X.shape[1])]

        if isinstance(y, pd.Series):
            y = y.values

        node_id = 0
        queue = deque()

        root = self._make_leaf(y, depth=0, node_id=node_id)
        self.tree = DecisionTree(root)
        self.tree.variable_names = self.variable_names
        node_id += 1

        if not self._should_stop(y, depth=0):
            root, w, b = self._make_split_node(X, y, depth=0, node_id=node_id)
            if isinstance(root, DecisionNode):
                node_id += 1
                self.tree = DecisionTree(root)
                self.tree.variable_names = self.variable_names
                queue.append((root, X, y, 0, w, b))

        def freeze_bottom_nodes(tree, next_id):
            frozen_tree = deepcopy(tree)
            for node in frozen_tree.root.traverse_yield():
                if isinstance(node, DecisionNode) and len(node.children) == 0:
                    leaf = LeafNode(
                        node_id=next_id,
                        prediction=node.get_majority_class(),
                        depth=node.depth,
                        n_samples=0,
                        purity=0.0,
                    )
                    leaf.parent = node.parent
                    if node.parent:
                        node.parent.children = [leaf if c is node else c for c in node.parent.children]
                    else:
                        frozen_tree.root = leaf
                    next_id += 1
            return frozen_tree

        max_actual_depth = 0
        depth_bar = tqdm(desc="Building tree layer by layer", total=self.max_depth, unit="depth")

        while queue:
            current_depth = None
            next_layer = []

            while queue:
                parent, X_parent, y_parent, d, w, b = queue.popleft()
                current_depth = d + 1

                decision = X_parent @ w + b
                left_mask = decision < 0
                right_mask = ~left_mask

                for side_mask in [left_mask, right_mask]:
                    X_branch, y_branch = X_parent[side_mask], y_parent[side_mask]

                    if self._should_stop(y_branch, current_depth):
                        child = self._make_leaf(y_branch, current_depth, node_id)
                        parent.add_child(child)
                        node_id += 1
                    else:
                        split_node, new_w, new_b = self._make_split_node(X_branch, y_branch, current_depth, node_id)
                        if isinstance(split_node, DecisionNode):
                            parent.add_child(split_node)
                            next_layer.append((split_node, X_branch, y_branch, current_depth, new_w, new_b))
                            node_id += 1
                        else:
                            fallback = self._make_leaf(y_branch, current_depth, node_id)
                            parent.add_child(fallback)
                            node_id += 1

            if current_depth is not None:
                frozen_tree = freeze_bottom_nodes(self.tree, node_id)
                self.metrics_by_depth[current_depth] = evaluate_tree(frozen_tree, X, y)
                self.metrics_by_depth[current_depth]["depth"] = current_depth
                self.trees_by_depth[current_depth] = frozen_tree
                max_actual_depth = current_depth
                depth_bar.update(1)

            queue.extend(next_layer)

        depth_bar.close()

        # if 0 not in self.metrics_by_depth:
        #     frozen_root = freeze_bottom_nodes(self.tree, node_id)
        #     self.metrics_by_depth[0] = evaluate_tree(frozen_root, X, y)
        #     self.metrics_by_depth[0]["depth"] = 0
        #     self.trees_by_depth[0] = frozen_root

        if self.max_depth is not None and max_actual_depth < self.max_depth:
            print(f"\n⚠️  Tree stopped early at depth {max_actual_depth}, "
                  f"although max_depth was set to {self.max_depth}.")

            final_nodes = self.tree.get_leaves_at_depth(max_actual_depth)
            reasons = []
            for node in final_nodes:
                n_samples = node.n_samples
                purity = node.purity
                if n_samples < self.min_samples_split:
                    reasons.append(f"min_samples_split not met ({n_samples} < {self.min_samples_split})")
                elif purity >= self.min_purity:
                    reasons.append(f"node purity {purity:.2f} exceeded threshold {self.min_purity}")
                else:
                    reasons.append("no further gain from split")

            unique_reasons = sorted(set(reasons))
            print("   ➤ Likely reason(s):")
            for reason in unique_reasons:
                print(f"     - {reason}")

    def predict(self, X):
        """Predict labels for a batch of samples."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.array([self.tree.predict(x) for x in X])

    def score(self, X, y):
        """Compute classification accuracy."""
        return np.mean(self.predict(X) == y)

    def get_tree_by_depth(self, depth: int):
        """Return a deep copy of the tree at a specific depth."""
        return deepcopy(self.trees_by_depth[depth])

    def available_depths(self):
        return sorted(self.trees_by_depth.keys())

    def get_tree(self):
        """Return the fitted DecisionTree object."""
        return self.tree
