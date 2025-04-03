"""
optimizer.py

This module implements the Tree Alternating Optimization (TAO) algorithm for classification trees.
The TAO algorithm alternates over the nodes of a decision tree to update decision nodes by minimizing
a surrogate logistic loss (augmented with an L1 regularization term) and updating leaf nodes via a
majority vote. The optimizer relies on a standard tree format defined in tree.py.
"""

import numpy as np
from E_TAO_algorithm.losses import logistic_loss, logistic_loss_gradient
from E_TAO_algorithm.regularizers import l1_subgradient
from E_TAO_algorithm.utils import compute_reduced_sets
from D_choosing_oblique_decision_trees.tree import DecisionNode, LeafNode


class TAOOptimizer:
    """
    Implements the Tree Alternating Optimization (TAO) algorithm for classification trees.

    The algorithm alternates over nodes in a decision tree. At each decision node, a binary
    classification problem (the reduced problem) is solved on the “care” examples—those examples
    for which the choice of branch (left vs. right) affects the downstream loss. A surrogate
    logistic loss (plus an L1 penalty) is minimized with gradient descent.
    """
    def __init__(self, max_iterations=20, tol=1e-5, learning_rate=0.01, lambda_reg=0.01):
        """
        Initialize the TAOOptimizer.

        Args:
            max_iterations (int): Maximum iterations for TAO.
            tol (float): Tolerance for convergence based on change in objective.
            learning_rate (float): Step size for gradient-based updates.
            lambda_reg (float): L1 regularization strength.
        """
        self.max_iterations = max_iterations
        self.tol = tol
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg

    def optimize(self, tree, x, y):
        """
        Optimize the tree parameters using alternating optimization.

        Args:
            tree (DecisionTree): The decision tree (using the standard format in tree.py).
            x (np.array): Training features with shape (N, D).
            y (np.array): Training labels with shape (N,).

        Returns:
            DecisionTree: The optimized tree.
        """
        prev_obj = np.inf
        for iteration in range(self.max_iterations):
            # Compute reduced sets: mapping of each node to indices of training examples that reach it.
            reduced_sets = compute_reduced_sets(tree, x)
            # Update nodes based on their reduced sets.
            self._update_tree(tree, x, y, reduced_sets)
            # Compute the surrogate objective.
            curr_obj = self._compute_objective(tree, x, y, reduced_sets)
            # print(f"Iteration {iteration}: objective = {curr_obj:.6f}")
            if abs(prev_obj - curr_obj) < self.tol:
                print(f"Convergence reached at iteration {iteration}.")
                break
            prev_obj = curr_obj
        return tree

    def _update_tree(self, tree, x, y, reduced_sets):
        """
        Traverse the tree in level order and update each node using its reduced set.

        Args:
            tree (DecisionTree): The tree to update.
            x (np.array): Training features.
            y (np.array): Training labels.
            reduced_sets (dict): Mapping of each node to list of indices of training examples that reach it.
        """
        levels = TAOOptimizer._get_levels(tree.root)
        for level in levels:
            for node in level:
                indices = reduced_sets.get(node, [])
                if len(indices) == 0:
                    continue
                x_red = x[indices]
                y_red = y[indices]
                if isinstance(node, DecisionNode):
                    # Compute pseudo-labels and care indices.
                    pseudo_labels, care_idx = TAOOptimizer._compute_pseudo_labels(node, x_red, y_red)
                    if care_idx.size == 0:
                        # No care examples; skip update.
                        continue
                    x_care = x_red[care_idx]
                    # Compute prediction scores: s = x_care dot weights + bias.
                    s = x_care.dot(node.weights) + node.bias
                    # Compute gradient of logistic loss.
                    grad_loss = logistic_loss_gradient(s, pseudo_labels)
                    grad_w = x_care.T.dot(grad_loss) / care_idx.size
                    grad_b = np.mean(grad_loss)
                    # Incorporate L1 subgradient.
                    grad_w += l1_subgradient(node.weights, lambda_reg=self.lambda_reg)
                    # Update parameters.
                    node.weights -= self.learning_rate * grad_w
                    node.bias -= self.learning_rate * grad_b
                elif isinstance(node, LeafNode):
                    # Update leaf prediction via a majority vote.
                    counts = np.bincount(y_red.astype(int))
                    node.prediction = np.argmax(counts)

    @staticmethod
    def _compute_pseudo_labels(node, x_red, y_red):
        """
        Compute pseudo-labels for a decision node using a refined strategy.

        For each sample in x_red, simulate sending it down both left and right subtrees.
        If both branches yield the same predicted label, mark the example as “don't care” (exclude it).
        Otherwise, compute the 0/1 loss for left and right; assign pseudo-label 0 if left branch loss is lower
        or equal, and 1 if right branch loss is lower.

        Args:
            node (DecisionNode): The decision node to update.
            x_red (np.array): Training examples reaching the node.
            y_red (np.array): True labels for these examples.

        Returns:
            tuple: (pseudo_labels, care_indices) where pseudo_labels is a np.array of labels (0 or 1)
            for the care examples, and care_indices is a np.array of indices indicating which examples are care points.
        """
        pseudo_labels_list = []
        care_indices_list = []
        for i, sample in enumerate(x_red):
            true_label = y_red[i]
            left_pred = TAOOptimizer._predict_from_node(node.children[0], sample)
            right_pred = TAOOptimizer._predict_from_node(node.children[1], sample)
            # If both branches yield the same prediction, the decision does not affect loss.
            if left_pred == right_pred:
                continue  # Exclude don't-care points.
            loss_left = 0 if left_pred == true_label else 1
            loss_right = 0 if right_pred == true_label else 1
            pseudo_labels_list.append(0 if loss_left <= loss_right else 1)
            care_indices_list.append(i)
        if care_indices_list:
            return np.array(pseudo_labels_list), np.array(care_indices_list)
        else:
            return np.array([]), np.array([])

    @staticmethod
    def _predict_from_node(node, x):
        """
        Recursively predict the output starting from the given node.
        If the node is a leaf, return its prediction.
        If the node is a decision node, use its decision function to select the appropriate branch.

        Args:
            node: The starting node.
            x (np.array): A single input instance.

        Returns:
            The predicted label from the subtree rooted at the given node.
        """
        if node.is_leaf():
            return node.prediction
        if isinstance(node, DecisionNode):
            branch = node.decision(x)
            if len(node.children) < 2:
                raise ValueError(f"DecisionNode {node.node_id} missing children.")
            return TAOOptimizer._predict_from_node(node.children[branch], x)
        return None

    def _compute_objective(self, tree, x, y, reduced_sets):
        """
        Compute a surrogate objective for the tree.

        For each decision node, the objective is defined as the average logistic loss on its care subset
        plus the L1 penalty on its weight.

        Args:
            tree: The decision tree.
            x (np.array): Training features.
            y (np.array): Training labels.
            reduced_sets (dict): Mapping of each node to a list of indices of training samples that reach it.

        Returns:
            float: The overall surrogate objective.
        """
        total_loss = 0.0
        levels = TAOOptimizer._get_levels(tree.root)
        for level in levels:
            for node in level:
                if isinstance(node, DecisionNode):
                    indices = reduced_sets.get(node, [])
                    if len(indices) == 0:
                        continue
                    x_red = x[indices]
                    # Compute pseudo-labels and care indices.
                    pseudo_labels, care_idx = TAOOptimizer._compute_pseudo_labels(node, x_red, y[indices])
                    if care_idx.size == 0:
                        continue
                    x_care = x_red[care_idx]
                    s = x_care.dot(node.weights) + node.bias
                    loss = np.mean(logistic_loss(s, pseudo_labels))
                    reg = self.lambda_reg * np.sum(np.abs(node.weights))
                    total_loss += loss + reg
        return total_loss

    @staticmethod
    def _get_levels(root):
        """
        Return the tree nodes level by level (BFS order) as a list of lists.

        Args:
            root: The root node of the tree.

        Returns:
            list: A list of lists where each inner list represents one level of the tree.
        """
        levels = []
        current = [root]
        while current:
            levels.append(current)
            next_level = []
            for node in current:
                next_level.extend(node.children)
            current = next_level
        return levels[::-1]
