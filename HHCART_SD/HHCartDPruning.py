"""
HHCartDPruningClassifier (HHCartDPruning.py)
-----------------------------------------------
Implements HHCartDPruningClassifier: an oblique decision tree classifier using Householder reflections,
followed by post hoc pruning at each depth level. Integrates cleanly with scikit-learn
interfaces for model selection and evaluation.

Features:
- Full-depth oblique decision tree induction
- Depth-wise pruning with separate metric evaluation
- Compatible with scikit-learns `fit`, `predict`, and `score` APIs
- Detailed docstrings, logging, and type hints throughout
"""

from typing import Callable, Union
import numpy as np
import pandas as pd
import math
from copy import deepcopy
from tqdm import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

from HHCART_SD.segmentor import CARTSegmentor
from HHCART_SD.evaluator import evaluate_tree
from HHCART_SD.tree import DecisionTree, DecisionNode, LeafNode, TreeNode


class HHCartDPruningClassifier(BaseEstimator, ClassifierMixin):
    """
    Oblique decision tree using Householder reflections (HHCART_SD-D) with bottom-up pruning.

    After growing the tree to full depth, each depth level is pruned recursively.
    Evaluation metrics (accuracy, coverage, density, etc.) are recorded for each level.

    Args:
        impurity (Callable): Impurity function to evaluate splits.
        segmentor (Segmentor): Object that generates axis-aligned splits (default=CARTSegmentor()).
        max_depth (int): Maximum depth for initial tree construction.
        mass_min (int or float): Minimum number of samples required to attempt a split.
            - If int >= 1 → absolute number of samples
            - If float ∈ (0, 1) → fraction of total training samples
        min_purity (float): Purity threshold to stop splitting if exceeded.
        tau (float): Numerical tolerance to ignore near-zero rotations.
        debug (bool): If True, prints additional debug information.
    """
    def __init__(self,
                 impurity: Callable,
                 segmentor=CARTSegmentor(),
                 max_depth: int = 5,
                 mass_min: Union[int, float] = 2,
                 min_purity: float = 1,
                 tau: float = 0.05,
                 debug: bool = False):
        self.impurity = impurity
        self.segmentor = segmentor
        self.max_depth = max_depth
        self.mass_min = mass_min
        self.min_purity = min_purity
        self.tau = tau
        self.debug = debug

        # Validate mass_min
        if not (isinstance(mass_min, int) and mass_min >= 1) \
                and not (isinstance(mass_min, float) and 0.0 < mass_min < 1.0):
            raise ValueError("mass_min must be an int ≥ 1 or float ∈ (0, 1)")

        self.n_samples_total = None
        self.tree = None
        self.variable_names = None
        self.metrics_by_depth = {}
        self.trees_by_depth = {}

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        """
        Fit a full-depth oblique decision tree and compute metrics at all pruning levels.

        Args:
            X (np.ndarray or pd.DataFrame): Input features, shape (n_samples, n_features).
            y (np.ndarray or pd.Series): Target labels, shape (n_samples,).

        Returns:
            None
        """
        # Handle column names if input is a DataFrame
        if isinstance(X, pd.DataFrame):
            self.variable_names = list(X.columns)
            X = X.values
        else:
            self.variable_names = [f"x{i}" for i in range(X.shape[1])]

        if isinstance(y, pd.Series):
            y = y.values

        # Store total number of training samples for fraction-based mass_min
        self.n_samples_total = X.shape[0]

        # Announce start of tree building
        print(f"[INFO] Building HHCartD oblique decision tree...")

        # Estimate max nodes for progress bar
        max_nodes_progress_bar = self._calculate_max_nodes()

        # Build the full unpruned tree using progress bar
        with tqdm(total=max_nodes_progress_bar, desc="Building tree nodes") as progress_bar:
            self.tree = self._build_full_tree(X, y, depth=0, progress_bar=progress_bar)

        self.tree.variable_names = self.variable_names

        # Determine how deep the unpruned tree actually grew
        full_depth = self._get_max_depth(self.tree.root)
        if self.debug:
            print(f"[INFO] Full tree built. Max depth: {full_depth}")
            print("[DEBUG] Node depths:")
            for node in self.tree.root.traverse_yield():
                print(
                    f"  id={node.node_id}, depth={node.depth}, "
                    f"type={'Leaf' if isinstance(node, LeafNode) else 'Split'}")

        # === Prune full tree to all depths and evaluate ===
        print(f"[INFO] Pruning the full tree and evaluating metrics at each depth level; "
              f"storing pruned trees and metrics...")

        # Iteratively prune the full tree to every possible depth from 0 to full_depth
        for d in range(full_depth + 1):
            pruned_tree = self._prune_tree_to_depth(self.tree, prune_depth=d)
            max_actual_depth = self._get_max_depth(pruned_tree.root)

            # Check whether the pruning depth produced the expected result
            if self.debug:
                if d == 0:
                    print(f"[CHECK] depth={d}, should only contain root. Actual max depth: {max_actual_depth}")
                elif max_actual_depth != d:
                    print(f"[MISMATCH] Expected max depth {d}, but got {max_actual_depth}")

            # Update split and leaf statistics after pruning
            pruned_tree.refresh_metadata()
            self.trees_by_depth[d] = pruned_tree

            # Print tree structure diagnostics (optional)
            if self.debug:
                num_nodes = sum(1 for _ in pruned_tree.root.traverse_yield())
                num_leaves = sum(1 for n in pruned_tree.root.traverse_yield() if isinstance(n, LeafNode))
                print(f"[PRUNE] Depth={d} | Nodes={num_nodes} | Leaves={num_leaves}")

            try:
                # Evaluate metrics like accuracy, coverage, etc.
                metrics = evaluate_tree(pruned_tree, X, y)
                metrics["depth"] = d
                self.metrics_by_depth[d] = metrics

                # Sanity check for NaNs in metrics
                if self.debug:
                    nan_keys = [k for k, v in metrics.items() if isinstance(v, float) and np.isnan(v)]
                    if nan_keys:
                        print(f"[WARNING] Metrics contain NaNs at depth={d} in keys: {nan_keys}")
                    else:
                        print(f"[OK] Metrics saved for depth={d}: {metrics}")
            except Exception as e:
                if self.debug:
                    print(f"[ERROR] Failed to evaluate metrics at depth={d}: {e}")

    def _build_full_tree(self, X: np.ndarray, y: np.ndarray, depth: int, progress_bar=None) -> DecisionTree:
        """
        Construct the initial full-depth oblique decision tree.

        Splits are determined using Householder reflections followed by axis-aligned search
        in the transformed space.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target labels.
            depth (int): Current depth level.

        Returns:
            DecisionTree: Fully grown tree before pruning.
        """
        node_id = 0

        def build(X, y, depth) -> TreeNode:
            nonlocal node_id
            current_node_id = node_id
            node_id += 1

            if progress_bar is not None:
                progress_bar.update(1)

            if self.debug:
                print(f"[BUILD] Entering node_id={current_node_id}, depth={depth}, n_samples={len(y)}")

            if self._should_stop(y, depth):
                if self.debug:
                    print(f"[STOP] node_id={current_node_id}, depth={depth}, reason=stopping criteria met")
                return self._make_leaf(y, depth, current_node_id)

            H, rule = self._compute_reflection_and_split(X, y)
            if rule is None:
                if self.debug:
                    print(f"[STOP] node_id={current_node_id}, depth={depth}, reason=no valid split")
                return self._make_leaf(y, depth, current_node_id)

            i, thr = rule
            weights = H[:, i]
            bias = -thr

            proj = X @ weights + bias
            mask_left = proj < 0
            mask_right = ~mask_left

            y_left, y_right = y[mask_left], y[mask_right]

            node_impurity = self.impurity(y_left, y_right)

            if self.debug:
                print(
                    f"[SPLIT] node_id={current_node_id}, depth={depth}, feature_rotated_axis={i}, threshold={thr:.4f}, impurity={node_impurity:.4f}")

            node = DecisionNode(
                node_id=current_node_id,
                weights=weights,
                bias=bias,
                depth=depth,
                impurity=node_impurity
            )
            node.y = y

            node.add_child(build(X[mask_left], y_left, depth + 1))
            node.add_child(build(X[mask_right], y_right, depth + 1))
            return node

        return DecisionTree(build(X, y, 0))

    def _should_stop(self, y: np.ndarray, depth: int) -> bool:
        """
        Check whether a node should become a leaf.

        Stopping is triggered if:
        - the maximum depth is reached,
        - there are too few samples,
        - or the purity exceeds the specified threshold.

        Args:
            y (np.ndarray): Class labels reaching the node.
            depth (int): Current depth in the tree.

        Returns:
            bool: True if splitting should stop.
        """
        if self.max_depth is not None and depth >= self.max_depth:
            if self.debug:
                print(f"[STOP] Reached max_depth={self.max_depth}")
            return True

        if isinstance(self.mass_min, float):
            mass_min_required = int(np.ceil(self.mass_min * self.n_samples_total))
        else:
            mass_min_required = int(self.mass_min)

        if len(y) < mass_min_required:
            if self.debug:
                print(f"[STOP] Too few samples (n={len(y)} < mass_min={mass_min_required})")
            return True

        purity = np.max(np.bincount(y)) / len(y)
        if purity >= self.min_purity:
            if self.debug:
                print(f"[STOP] Purity threshold reached (purity={purity:.4f} >= min_purity={self.min_purity})")
            return True

        return False

    def _compute_reflection_and_split(self, X: np.ndarray, y: np.ndarray):
        """
        Applies Householder reflections using class-specific covariance directions,
        identifies the best axis-aligned split across reflected subspaces,
        and also checks the original (unreflected) space as fallback.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Labels.

        Returns:
            Tuple[np.ndarray, Optional[Tuple[int, float]]]: Best reflection matrix H and
            corresponding split rule (feature index, threshold), or None if no valid split.
        """
        d = X.shape[1]

        best_H = np.eye(d)
        best_rule = None
        best_imp = float("inf")

        imp_identity, rule_identity, _, _ = self.segmentor(X, y, self.impurity)
        if self.debug:
            print(f"[SPLIT] Identity split: impurity={imp_identity:.4f}, rule={rule_identity}")

        if rule_identity is not None:
            best_H = np.eye(d)
            best_rule = rule_identity
            best_imp = imp_identity

        for c in np.unique(y):
            Xc = X[y == c]
            if len(Xc) <= 1:
                continue
            cov = np.cov(Xc, rowvar=False)
            _, eigvecs = np.linalg.eigh(cov)
            mu = eigvecs[:, -1]
            if np.allclose(mu, 0):
                continue

            dev = np.sqrt(((np.eye(d) - mu) ** 2).sum(axis=1))
            if (dev > self.tau).any():
                i = np.argmax(dev)
                e = np.zeros(d)
                e[i] = 1
                w = (e - mu) / np.linalg.norm(e - mu)
                H = np.eye(d) - 2 * np.outer(w, w)
                imp, rule, _, _ = self.segmentor(X @ H, y, self.impurity)

                if self.debug:
                    print(f"[SPLIT] Reflection class={c}: impurity={imp:.4f}, rule={rule}")

                if rule is not None and imp < best_imp:
                    best_H = H
                    best_rule = rule
                    best_imp = imp

        return best_H, best_rule

    def _calculate_max_nodes(self) -> int:
        """
        Estimate the theoretical maximum number of nodes that the tree could build.

        The estimate is based purely on the maximum depth (`max_depth`), as this is a hard constraint.
        The number of nodes in a full binary tree of depth `max_depth` is:

            2^(max_depth + 1) - 1

        Returns:
            int: Maximum number of nodes allowed given max_depth.
        """
        # Max nodes based on max_depth (full binary tree formula)
        max_nodes_depth = 2 ** (self.max_depth + 1) - 1

        print(f"[INFO] Max number of nodes allowed by maximum depth constraint: {max_nodes_depth} "
              f"(used as progress bar target; actual number of splits unknown in advance).")

        return max_nodes_depth

    def _prune_tree_to_depth(self, tree: DecisionTree, prune_depth: int) -> DecisionTree:
        """
        Prune the tree so that its maximum depth is `prune_depth`.

        All children at depth `prune_depth` are removed, and their parents
        (at depth `prune_depth - 1`) are turned into LeafNodes.

        If `prune_depth == 0`, the entire tree is collapsed to a single root leaf
        based on the root's label distribution.

        Args:
            tree (DecisionTree): Full tree to prune.
            prune_depth (int): Maximum allowed depth (inclusive).

        Returns:
            DecisionTree: Tree pruned to desired depth.
        """
        copy = deepcopy(tree)

        if prune_depth == 0:
            # Special case: return a tree with a single root leaf
            root = copy.root
            leaf = self._make_leaf_from_node(root)
            return DecisionTree(leaf)

        root = copy.root
        for node in root.traverse_yield():
            if node.depth == prune_depth:
                # Node's children are at depth `prune_depth` → remove them
                node.children = []
                leaf = self._make_leaf_from_node(node)
                # Replace fields in-place
                node.__class__ = LeafNode
                node.prediction = leaf.prediction
                node.n_samples = leaf.n_samples
                node.purity = leaf.purity

        return copy

    @staticmethod
    def _make_leaf(y: np.ndarray, depth: int, node_id: int) -> LeafNode:
        """
        Construct a leaf node summarising the class distribution.

        The leaf stores the majority class, its purity, and the number of samples.

        Args:
            y (np.ndarray): Target labels at this node.
            depth (int): Depth of the leaf.
            node_id (int): Unique identifier for the node.

        Returns:
            LeafNode: A fully initialised leaf node.
        """
        label = int(np.bincount(y).argmax())
        purity = float(np.max(np.bincount(y)) / len(y))
        return LeafNode(node_id=node_id, prediction=label, depth=depth,
                        n_samples=len(y), purity=purity)

    @staticmethod
    def _make_leaf_from_node(node: TreeNode) -> LeafNode:
        """
        Convert a DecisionNode into a terminal LeafNode for pruning.
        Assumes the node has access to `y` for majority vote.

        Args:
            node (TreeNode): Must be a DecisionNode with `.y`.

        Returns:
            LeafNode: Leaf representation with inferred label, samples, and purity.
        """
        if isinstance(node, LeafNode):
            return node  # Already a leaf
        if not isinstance(node, DecisionNode):
            raise TypeError("Prunable node must be a DecisionNode")

        y = node.y
        label = node.get_majority_class()
        n_samples = len(y)
        purity = float(np.max(np.bincount(y)) / n_samples) if n_samples > 0 else 0.0

        return LeafNode(
            node_id=node.node_id,
            prediction=label,
            depth=node.depth,
            n_samples=n_samples,
            purity=purity
        )

    @staticmethod
    def _get_max_depth(root: TreeNode) -> int:
        """
        Compute the maximum depth of a tree.

        Traverses the entire tree to find the maximum depth value.

        Args:
            root (TreeNode): Root node of the tree.

        Returns:
            int: Maximum depth of the tree.
        """
        return max((n.depth for n in root.traverse_yield()), default=0)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class labels for input samples.

        Args:
            X (np.ndarray or pd.DataFrame): Input features, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted labels, shape (n_samples,).
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.array([self.tree.predict(x) for x in X])

    def score(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series],
              sample_weight: np.ndarray = None) -> float:
        """
        Compute the classification accuracy of the fitted decision tree.

        Args:
            X (np.ndarray or pd.DataFrame): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray or pd.Series): True labels of shape (n_samples,).
            sample_weight (np.ndarray, optional): Optional sample weights.

        Returns:
            float: Accuracy score (between 0 and 1).

        Raises:
            ValueError: If the model has not been fitted.
        """
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def get_tree_by_depth(self, depth: int) -> DecisionTree:
        """Retrieve pruned tree at specified depth."""
        return deepcopy(self.trees_by_depth[depth])

    def available_depths(self) -> list:
        """List all depths for which pruned trees are stored."""
        return sorted(self.trees_by_depth.keys())

    def get_tree(self) -> DecisionTree:
        """Return the full trained tree."""
        return self.tree

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """
        Return depth-wise evaluation metrics as a Pandas DataFrame.

        Each row corresponds to a pruning depth and includes metrics such as accuracy,
        coverage, and density.

        Returns:
            pd.DataFrame: Metrics indexed by tree depth.
        """
        return pd.DataFrame.from_dict(self.metrics_by_depth, orient="index")
