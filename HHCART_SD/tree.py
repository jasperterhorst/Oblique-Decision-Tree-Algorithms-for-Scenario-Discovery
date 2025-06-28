"""
Decision Tree Structure (tree.py)
---------------------------------
Defines the core data structures for decision trees used in HHCART_SD.
Includes:
- TreeNode: Generic node base class
- DecisionNode: Internal split node with weights and bias
- LeafNode: Terminal node with prediction
- DecisionTree: Full tree interface with traversal and prediction

Each node type supports traversal and introspection.
"""

import numpy as np
import matplotlib.pyplot as plt


class TreeNode:
    """
    Base class for all nodes in a decision tree.

    Attributes:
        node_id (int): Unique identifier for the node.
        depth (int): Depth of the node in the tree.
        parent (TreeNode): Parent node reference.
        children (list): List of child nodes.
    """

    def __init__(self, node_id: int, depth: int = 0):
        self.node_id = node_id
        self.depth = depth
        self.parent = None
        self.children = []

    def is_leaf(self) -> bool:
        """
        Check if this node is a leaf node.

        Returns:
            bool: True if the node has no children.
        """
        return len(self.children) == 0

    def add_child(self, child: "TreeNode") -> None:
        """
        Add a child to this node and set its parent and depth.

        Args:
            child (TreeNode): Node to attach as a child.
        """
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    def traverse_yield(self):
        """
        Yield all nodes in the subtree rooted at this node.

        Yields:
            TreeNode: Each node in depth-first traversal order.
        """
        yield self
        for child in self.children:
            yield from child.traverse_yield()

    def __repr__(self) -> str:
        return f"TreeNode(id={self.node_id}, depth={self.depth})"


class DecisionNode(TreeNode):
    """
    A decision node that splits input space using a linear function.

    Attributes:
        weights (np.ndarray): Weight vector for split direction.
        bias (float): Bias term for the decision threshold.
        y (np.ndarray or None): Labels at this node (used for majority voting).
        impurity (float or None): Split impurity at this node (e.g., Gini impurity).
    """

    def __init__(self, node_id: int, weights: np.ndarray = None, bias: float = 0.0, depth: int = 0,
                 impurity: float = None):
        super().__init__(node_id, depth)
        self.weights = weights if weights is not None else np.array([])
        self.bias = bias
        self.y = None
        self.impurity = impurity
        self.is_axis_aligned = False
        self.split_metadata = {}

    def decision(self, x: np.ndarray) -> int:
        """
        Apply the decision function to a feature vector.

        Args:
            x (np.ndarray): Feature vector.

        Returns:
            int: 1 if x satisfies the split condition; else 0.
        """
        return int(np.dot(x, self.weights) + self.bias >= 0)

    def get_majority_class(self) -> int:
        """
        Determine the majority class among samples reaching this node.

        Returns:
            int: Most frequent class label, or 0 if undefined.
        """
        if isinstance(self.y, np.ndarray) and len(self.y) > 0:
            return int(np.bincount(self.y).argmax())
        return 0

    def __repr__(self) -> str:
        split_type = "axis-aligned" if self.is_axis_aligned else "oblique"
        return (
            f"DecisionNode(id={self.node_id}, depth={self.depth}, "
            f"impurity={self.impurity:.4f}, weights={self.weights}, bias={self.bias}, {split_type})"
        )


class LeafNode(TreeNode):
    """
    A terminal leaf node that holds a prediction.

    Attributes:
        prediction (int): Output value for samples reaching this node.
        n_samples (int): Number of samples routed to this node.
        purity (float): Fraction of dominant class in this node.
    """

    def __init__(
        self,
        node_id: int,
        prediction: int = None,
        depth: int = 0,
        n_samples: int = None,
        purity: float = None
    ):
        super().__init__(node_id, depth)
        self.prediction = prediction
        self.n_samples = n_samples
        self.purity = purity

    def traverse_yield(self):
        """
        Yield only this node, as it has no children.

        Yields:
            LeafNode: The node itself.
        """
        yield self

    def __repr__(self) -> str:
        return (
            f"LeafNode(id={self.node_id}, depth={self.depth}, "
            f"prediction={self.prediction}, n_samples={self.n_samples}, purity={self.purity:.2f})"
        )


class DecisionTree:
    """
    A complete decision tree composed of TreeNode instances.

    Attributes:
        root (TreeNode): Root of the tree.
        max_depth (int): Maximum depth observed in the tree.
        num_splits (int): Number of internal decision nodes.
        num_leaves (int): Number of terminal leaf nodes.
        num_nodes (int): Total number of nodes in the tree.
    """

    def __init__(self, root: TreeNode):
        self.root = root
        self.max_depth = self._compute_max_depth(root)
        self.num_splits = self.count_nodes(DecisionNode)
        self.num_leaves = self.count_nodes(LeafNode)
        self.num_nodes = self.count_nodes()
        self.variable_names = None

    def _compute_max_depth(self, node: TreeNode) -> int:
        """
        Recursively compute maximum tree depth.

        Args:
            node (TreeNode): Root or subtree node.

        Returns:
            int: Maximum depth from this node.
        """
        if node.is_leaf():
            return node.depth
        return max(self._compute_max_depth(child) for child in node.children)

    def predict(self, x: np.ndarray) -> int:
        """
        Predict the output for a given input vector.

        Args:
            x (np.ndarray): Input feature vector.

        Returns:
            int: Predicted class label.

        Raises:
            ValueError: If traversal ends in a malformed node.
        """
        node = self.root
        while not node.is_leaf():
            if isinstance(node, DecisionNode):
                direction = node.decision(x)
                if len(node.children) != 2:
                    raise ValueError(
                        f"DecisionNode {node.node_id} must have exactly two children."
                    )
                node = node.children[direction]
            else:
                break

        if isinstance(node, LeafNode):
            return int(node.prediction)
        raise ValueError("Traversal ended in a non-leaf node. Check tree structure.")

    def traverse(self, action=lambda node: print(node), node: TreeNode = None) -> None:
        """
        Apply an action to all nodes in the tree.

        Args:
            action (callable): Function to apply to each node.
            node (TreeNode, optional): Starting node. Defaults to root.
        """
        node = node or self.root
        action(node)
        for child in node.children:
            self.traverse(action, child)

    def print_structure(self) -> None:
        """
        Print a textual structure of the tree.
        """
        self._print_node(self.root)

    def _print_node(self, node: TreeNode, indent: str = "") -> None:
        """
        Print a single node and recursively print its children.

        Args:
            node (TreeNode): Node to print.
            indent (str): Indentation string.
        """
        if isinstance(node, DecisionNode):
            terms = [f"{w:+.2f}*x{i}" for i, w in enumerate(np.ravel(node.weights)) if abs(float(w)) > 1e-6]
            condition = " ".join(terms) + f" + {node.bias:+.2f} >= 0"
            split_type = "axis-aligned" if node.is_axis_aligned else "oblique"
            print(f"{indent}[Node id={node.node_id}, depth={node.depth}, impurity={node.impurity:.4f}, "
                  f"{split_type}] (split: {condition})")
            for i, child in enumerate(node.children):
                branch = "├── " if i == 0 else "└── "
                self._print_node(child, indent + branch)
        elif isinstance(node, LeafNode):
            print(f"{indent}[Node id={node.node_id}, depth={node.depth}] "
                  f"(leaf: prediction={node.prediction}, purity={node.purity:.2f}, samples={node.n_samples})")

    def get_leaves_at_depth(self, depth: int) -> list:
        """
        Get all leaf nodes at a given tree depth.

        Args:
            depth (int): Depth to inspect.

        Returns:
            list: List of LeafNode instances at the specified depth.
        """
        leaves = []

        def visit(node):
            if isinstance(node, LeafNode) and node.depth == depth:
                leaves.append(node)

        self.traverse(visit)
        return leaves

    def count_nodes(self, node_type: type = None) -> int:
        """
        Count nodes of a specific type (or all nodes if type is None).

        Args:
            node_type (type, optional): Type to count (e.g., LeafNode).

        Returns:
            int: Number of matching nodes.
        """
        count = 0

        def count_fn(node):
            nonlocal count
            if node_type is None or isinstance(node, node_type):
                count += 1

        self.traverse(count_fn)
        return count

    def refresh_metadata(self) -> None:
        """
        Recalculate and update node statistics for the tree.

        This method updates the following attributes:
        - `num_splits`: number of internal decision nodes
        - `num_leaves`: number of leaf nodes
        - `num_nodes`: total number of nodes (internal + leaf)

        It should be called after structural changes to the tree (e.g., pruning)
        to ensure all metadata reflects the current state.

        Returns:
            None
        """
        self.num_splits = self.count_nodes(DecisionNode)
        self.num_leaves = self.count_nodes(LeafNode)
        self.num_nodes = self.count_nodes()

    def _build_tree_str(self, node: TreeNode, indent: str = "") -> str:
        """
        Construct a string representation of the tree.

        Args:
            node (TreeNode): Node to include.
            indent (str): Indentation level.

        Returns:
            str: Tree as string.
        """
        tree_str = indent + repr(node) + "\n"
        for child in node.children:
            tree_str += self._build_tree_str(child, indent + "    ")
        return tree_str

    def plot(self) -> None:
        """
        Plot the structure of the tree as a text diagram using matplotlib.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis("off")
        tree_str = self._build_tree_str(self.root)
        ax.text(
            0.05, 0.95, tree_str,
            fontsize=10,
            family="monospace",
            verticalalignment="top",
            transform=ax.transAxes
        )
        plt.show()
