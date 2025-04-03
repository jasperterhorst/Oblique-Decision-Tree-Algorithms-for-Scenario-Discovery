"""
Converter module for CO2 models.

Converts a trained CO2 model into a standardized DecisionTree.
"""

from D_oblique_decision_trees.core.tree import DecisionTree, DecisionNode, LeafNode


def convert_co2(model):
    """
    Convert a trained CO2 model into a standardized DecisionTree.

    Parameters:
        model: A CO2 model with attributes:
               - _root: The root node of the model.
               - For non-leaf nodes, attributes _weights, _bias, _left_child, _right_child.
               - For leaf nodes, a label attribute.

    Returns:
        DecisionTree: The standardized decision tree.
    """
    def recurse(node, node_id=0, depth=0):
        if node.is_leaf:
            return LeafNode(node_id=node_id, prediction=node.label, depth=depth)
        else:
            weights = node._weights
            bias = node._bias
            decision_node = DecisionNode(node_id=node_id, weights=weights, bias=bias, depth=depth)

            left = recurse(node._left_child, 2 * node_id + 1, depth + 1)
            right = recurse(node._right_child, 2 * node_id + 2, depth + 1)
            decision_node.add_child(left)
            decision_node.add_child(right)
            return decision_node

    return DecisionTree(recurse(model._root))
