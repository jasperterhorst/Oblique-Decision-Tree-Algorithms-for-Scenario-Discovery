"""
tree.py

This module defines the core classes used to represent decision trees. It provides a unified
structure to represent various oblique decision tree models (e.g., OC1, CART-LC), enabling
interoperable analysis, visualization, and later optimization via the Tree Alternating
Optimization (TAO) algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt


class TreeNode:
    def __init__(self, node_id, depth=0):
        self.node_id = node_id
        self.depth = depth
        self.parent = None
        self.children = []

    def is_leaf(self):
        return len(self.children) == 0

    def add_child(self, child):
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    def __repr__(self):
        return f"TreeNode(id={self.node_id}, depth={self.depth})"


class DecisionNode(TreeNode):
    def __init__(self, node_id, weights=None, bias=0.0, depth=0):
        super().__init__(node_id, depth)
        self.weights = weights if weights is not None else np.array([])
        self.bias = bias

    def decision(self, x):
        return 1 if np.dot(x, self.weights) + self.bias >= 0 else 0

    def __repr__(self):
        return (f"DecisionNode(id={self.node_id}, depth={self.depth}, "
                f"weights={self.weights}, bias={self.bias})")


class LeafNode(TreeNode):
    def __init__(self, node_id, prediction=None, depth=0):
        super().__init__(node_id, depth)
        self.prediction = prediction

    def __repr__(self):
        return f"LeafNode(id={self.node_id}, depth={self.depth}, prediction={self.prediction})"


class DecisionTree:
    def __init__(self, root: TreeNode):
        self.root = root
        self.max_depth = self._compute_max_depth(root)
        self.num_splits = self.count_nodes(DecisionNode)
        self.num_leaves = self.count_nodes(LeafNode)
        self.num_nodes = self.count_nodes()

    def _compute_max_depth(self, node):
        if node.is_leaf():
            return node.depth
        return max(self._compute_max_depth(child) for child in node.children)

    def predict(self, x):
        node = self.root
        while not node.is_leaf():
            if isinstance(node, DecisionNode):
                direction = node.decision(x)
                if len(node.children) < 2:
                    raise ValueError(
                        f"DecisionNode {node.node_id} does not have two children."
                    )
                node = node.children[direction]
            else:
                break
        return node.prediction if isinstance(node, LeafNode) else None

    def traverse(self, action=lambda node: print(node), node=None):
        node = node or self.root
        action(node)
        for child in node.children:
            self.traverse(action, child)

    def count_nodes(self, node_type=None):
        count = 0

        def _count(node):
            nonlocal count
            if node_type is None or isinstance(node, node_type):
                count += 1

        self.traverse(_count)
        return count

    def _build_tree_str(self, node, indent=""):
        tree_str = indent + repr(node) + "\n"
        for child in node.children:
            tree_str += self._build_tree_str(child, indent + "    ")
        return tree_str

    def plot(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis("off")
        tree_str = self._build_tree_str(self.root)
        ax.text(
            0.05,
            0.95,
            tree_str,
            fontsize=10,
            family="monospace",
            verticalalignment="top",
            transform=ax.transAxes,
        )
        plt.show()
