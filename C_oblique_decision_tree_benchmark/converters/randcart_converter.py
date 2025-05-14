"""
Converter module for RandCART models.

Converts a trained RandCART model into a standardized DecisionTree.
"""

from C_oblique_decision_tree_benchmark.core.tree import DecisionTree, DecisionNode, LeafNode


def convert_randcart(model):
    """
    Convert a trained RandCART model into a standardized DecisionTree.

    Parameters:
        model: A RandCART model with attributes:
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

            left = recurse(node._left_child, 2 * node_id + 1, depth + 1)
            right = recurse(node._right_child, 2 * node_id + 2, depth + 1)
            decision_node.add_child(left)
            decision_node.add_child(right)
            return decision_node

    return DecisionTree(recurse(model._root))
