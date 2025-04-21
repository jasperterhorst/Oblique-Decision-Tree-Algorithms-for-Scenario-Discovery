from C_oblique_decision_trees.core.tree import DecisionTree, DecisionNode, LeafNode
import numpy as np


def convert_ridge_cart(model):
    """
    Convert a trained RidgeCARTClassifier into a standardized DecisionTree.

    Parameters:
        model: A CARTClassifier with _root as root node,
               and internal nodes with _weights, _left_child, _right_child,
               and leaves with a label.

    Returns:
        DecisionTree: the unified tree object.
    """
    node_counter = [0]

    def recurse(node, depth):
        node_id = node_counter[0]
        node_counter[0] += 1

        if node.is_leaf:
            return LeafNode(node_id=node_id, prediction=node.label, depth=depth)

        weights = node._weights[:-1]
        bias = -node._weights[-1]
        active_features = np.where(weights != 0)[0]

        decision_node = DecisionNode(
            node_id=node_id,
            weights=weights,
            bias=bias,
            depth=depth
        )

        decision_node.add_child(recurse(node.left_child, depth + 1))
        decision_node.add_child(recurse(node.right_child, depth + 1))
        return decision_node

    root = recurse(model.root, 0)
    return DecisionTree(root=root)
