"""
Core module for representing decision trees.

This module implements the basic data structures used to model decision trees.
"""

import numpy as np
import matplotlib.pyplot as plt


class TreeNode:
    """
    Represents a node in a decision tree.
    """

    def __init__(self, node_id, depth=0):
        self.node_id = node_id
        self.depth = depth
        self.parent = None
        self.children = []

    def is_leaf(self):
        """Return True if the node has no children."""
        return len(self.children) == 0

    def add_child(self, child):
        """
        Add a child node.

        The child's parent is set to this node and its depth is updated.
        """
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    def traverse_yield(self):
        """
        Yield all nodes in the subtree rooted at this node (depth-first).
        """
        yield self
        for child in self.children:
            yield from child.traverse_yield()

    def __repr__(self):
        return f"TreeNode(id={self.node_id}, depth={self.depth})"


class DecisionNode(TreeNode):
    """
    A decision node that computes an outcome based on input features.
    """

    def __init__(self, node_id, weights=None, bias=0.0, depth=0):
        super().__init__(node_id, depth)
        self.weights = weights if weights is not None else np.array([])
        self.bias = bias
        self.y = None  # to be optionally assigned at construction or fitting

    def decision(self, x):
        """
        Compute the decision outcome for an input vector.

        Returns:
            1 if the weighted sum plus bias is non-negative; otherwise 0.
        """
        return 1 if np.dot(x, self.weights) + self.bias >= 0 else 0

    def get_majority_class(self):
        """
        Return the most frequent class label in the associated samples.
        If `self.y` is not set, fallback to class 0.
        """
        if isinstance(self.y, np.ndarray) and len(self.y) > 0:
            return int(np.bincount(self.y).argmax())
        return 0

    def __repr__(self):
        return (f"DecisionNode(id={self.node_id}, depth={self.depth}, "
                f"weights={self.weights}, bias={self.bias})")


class LeafNode(TreeNode):
    """
    A leaf node that holds a prediction.
    """

    def __init__(self, node_id, prediction=None, depth=0, n_samples=None, purity=None):
        super().__init__(node_id, depth)
        self.prediction = prediction
        self.n_samples = n_samples
        self.purity = purity

    def traverse_yield(self):
        yield self

    def __repr__(self):
        return (f"LeafNode(id={self.node_id}, depth={self.depth}, "
                f"prediction={self.prediction}, n_samples={self.n_samples}, purity={self.purity:.2f})")


class DecisionTree:
    """
    A decision tree model built from TreeNode objects.
    """

    def __init__(self, root: TreeNode):
        self.root = root
        self.max_depth = self._compute_max_depth(root)
        self.num_splits = self.count_nodes(DecisionNode)
        self.num_leaves = self.count_nodes(LeafNode)
        self.num_nodes = self.count_nodes()
        self.variable_names = None

    def _compute_max_depth(self, node):
        if node.is_leaf():
            return node.depth
        return max(self._compute_max_depth(child) for child in node.children)

    def predict(self, x):
        """
        Predict the output for a given input vector.

        Returns:
            The prediction from the reached leaf node.
        """
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

        if isinstance(node, LeafNode):
            return int(node.prediction)  # ensure int
        else:
            raise ValueError("Tree traversal ended in a non-leaf node. Tree may be malformed.")

    def traverse(self, action=lambda node: print(node), node=None):
        """
        Traverse the tree, applying an action to each node.
        """
        node = node or self.root
        action(node)
        for child in node.children:
            self.traverse(action, child)

    def print_structure(self):
        self._print_node(self.root, indent="")

    def _print_node(self, node, indent=""):
        if isinstance(node, DecisionNode):
            # Format weights as readable string (e.g., "0.7*x1 - 0.4*x2")
            terms = [f"{w:+.2f}*x{i}" for i, w in enumerate(node.weights) if abs(w) > 1e-6]
            condition = " ".join(terms) + f" + {node.bias:+.2f} >= 0"
            print(f"{indent}[Node id={node.node_id}, depth={node.depth}] (split: {condition})")

            # Recurse on children
            for i, child in enumerate(node.children):
                branch = "├── " if i == 0 else "└── "
                self._print_node(child, indent + branch)
        elif isinstance(node, LeafNode):
            print(f"{indent}[Node id={node.node_id}, depth={node.depth}] "
                  f"(leaf: prediction={node.prediction}, purity={node.purity:.2f}, samples={node.n_samples})")

    def get_leaves_at_depth(self, depth):
        """
        Return a list of all LeafNode instances at the specified depth.
        """
        leaves = []

        def visit(node):
            if isinstance(node, LeafNode) and node.depth == depth:
                leaves.append(node)

        self.traverse(visit)
        return leaves

    def count_nodes(self, node_type=None):
        """
        Count the total number of nodes in the tree.
        If a node_type is provided, count only nodes of that type.
        """
        count = 0

        def _count(node):
            nonlocal count
            if node_type is None or isinstance(node, node_type):
                count += 1

        self.traverse(_count)
        return count

    def _build_tree_str(self, node, indent=""):
        """
        Recursively build a string representation of the tree.
        """
        tree_str = indent + repr(node) + "\n"
        for child in node.children:
            tree_str += self._build_tree_str(child, indent + "    ")
        return tree_str

    def plot(self):
        """
        Plot a textual representation of the tree using matplotlib.
        """
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
