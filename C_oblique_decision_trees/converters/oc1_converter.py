"""
Converter module for OC1 models.

Converts a trained OC1 model into a standardized DecisionTree.
"""

from C_oblique_decision_trees.core.tree import DecisionTree, DecisionNode, LeafNode


def convert_oc1(model):
    """
    Convert a trained OC1 model into a standardized DecisionTree.

    Parameters:
        model: An OC1 model with an attribute 'tree_' containing a 'root_node'.
               Non-leaf nodes should have attributes: is_leaf(), w, b, left_child, right_child.
               Leaf nodes should have a 'value' attribute.

    Returns:
        DecisionTree: The standardized decision tree.
    """
    def recurse(node, node_id=0, depth=0):
        if node.is_leaf():
            return LeafNode(node_id=node_id, prediction=node.value, depth=depth)
        else:
            weights = node.w
            bias = node.b
            decision_node = DecisionNode(node_id=node_id, weights=weights, bias=bias, depth=depth)

            left = recurse(node.left_child, 2 * node_id + 1, depth + 1)
            right = recurse(node.right_child, 2 * node_id + 2, depth + 1)
            decision_node.add_child(left)
            decision_node.add_child(right)
            return decision_node

    return DecisionTree(recurse(model.tree_.root_node))
