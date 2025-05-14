"""
Converter module for HHCart models.

Converts a trained HHCartClassifier model into a standardized DecisionTree.
"""

from C_oblique_decision_tree_benchmark.core.tree import DecisionTree, DecisionNode, LeafNode


def convert_hhcart(model):
    """
    Convert a trained HHCartClassifier model into a standardized DecisionTree.

    Parameters:
        model: A HHCartClassifier model with attributes:
               - _root: The root node.
               - For non-leaf nodes, _weights, _left_child, _right_child.
               - For leaf nodes, a label attribute.

    Returns:
        DecisionTree: The standardized decision tree.
    """
    def recurse(node, node_id=0, depth=0):
        if node.is_leaf:
            return LeafNode(node_id=node_id, prediction=node.label, depth=depth)
        else:
            weights = node._weights[:-1]
            bias = -node._weights[-1]
            decision_node = DecisionNode(node_id=node_id, weights=weights, bias=bias, depth=depth)

            left_child = recurse(node._left_child, node_id=2 * node_id + 1, depth=depth + 1)
            right_child = recurse(node._right_child, node_id=2 * node_id + 2, depth=depth + 1)

            decision_node.add_child(left_child)
            decision_node.add_child(right_child)
            return decision_node

    root = recurse(model._root)
    return DecisionTree(root=root)
