"""
Converter module for Weighted Oblique Decision Tree (WODT) models.

Converts a trained WODT model into a standardized DecisionTree.
"""

from C_oblique_decision_trees.core.tree import DecisionTree, DecisionNode, LeafNode
import numpy as np


def convert_wodt(model):
    """
    Convert a trained WODT model into a standardized DecisionTree.

    Parameters:
        model: A WODT model created by 'Ensembles_of_Oblique_Decision_Trees with a 'root_node' attribute.
               Non-leaf nodes should have a 'split' object with attributes 'paras' (weights)
               and 'threshold' (bias). Children are accessed via 'LChild' and 'RChild'.
               Leaf nodes may include a 'class_distribution' attribute.

    Returns:
        DecisionTree: The standardized decision tree.
    """
    def recurse(node, node_id=0, depth=0):
        if node.is_leaf:
            prediction = int(np.argmax(node.class_distribution)) if hasattr(node, "class_distribution") else 0
            return LeafNode(node_id=node_id, prediction=prediction, depth=depth)

        weights = np.array(node.split.paras)
        bias = -node.split.threshold
        decision_node = DecisionNode(node_id=node_id, weights=weights, bias=bias, depth=depth)
        decision_node.add_child(recurse(node.LChild, 2 * node_id + 1, depth + 1))
        decision_node.add_child(recurse(node.RChild, 2 * node_id + 2, depth + 1))
        return decision_node

    return DecisionTree(recurse(model.root_node))
