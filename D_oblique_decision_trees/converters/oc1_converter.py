# D_oblique_decision_trees/converters/oc1_converter.py
from D_oblique_decision_trees.tree import DecisionTree, DecisionNode, LeafNode


def convert_oc1(model):
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
