# cart_converter.py

from C_oblique_decision_tree_benchmark.core.tree import DecisionTree, DecisionNode, LeafNode


def convert_cart(model):
    """
    Convert a trained CARTClassifier into a standardized DecisionTree.

    Parameters:
        model: A CARTClassifier with _root as root node,
               and internal nodes with _weights, _left_child, _right_child,
               and leaves with a label.

    Returns:
        DecisionTree: the unified tree object.
    """
    def recurse(node, node_id=0, depth=0):
        if node.is_leaf:
            return LeafNode(node_id=node_id, prediction=node.label, depth=depth)

        weights = node._weights[:-1]
        bias = -node._weights[-1]
        internal = DecisionNode(node_id=node_id, weights=weights, bias=bias, depth=depth)

        left = recurse(node._left_child, 2 * node_id + 1, depth + 1)
        right = recurse(node._right_child, 2 * node_id + 2, depth + 1)

        internal.add_child(left)
        internal.add_child(right)
        return internal

    return DecisionTree(recurse(model._root))
