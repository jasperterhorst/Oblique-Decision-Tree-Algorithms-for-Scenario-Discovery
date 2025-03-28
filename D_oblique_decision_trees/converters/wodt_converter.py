# # D_oblique_decision_trees/converters/wodt_converter.py
#
# from D_oblique_decision_trees.tree import DecisionTree, DecisionNode, LeafNode
# import numpy as np
#
#
# def convert_wodt(model):
#     def recurse(node, node_id=0, depth=0):
#         if getattr(node, "is_leaf", False):
#             # If class_distribution is available, take the argmax as prediction
#             if hasattr(node, "class_distribution"):
#                 prediction = int(np.argmax(node.class_distribution))
#             else:
#                 prediction = 0  # fallback default
#             return LeafNode(node_id=node_id, prediction=prediction, depth=depth)
#         else:
#             # Extract weight vector and bias from the SplitQuestion object
#             split_obj = getattr(node, "split", None)
#             if split_obj is None or not hasattr(split_obj, "paras") or not hasattr(split_obj, "threshold"):
#                 raise ValueError("Node does not contain valid split parameters.")
#
#             weights = np.array(split_obj.paras)
#             bias = split_obj.threshold
#             decision_node = DecisionNode(node_id=node_id, weights=weights, bias=bias, depth=depth)
#
#             # Recurse
#             left = recurse(getattr(node, "LChild", None), 2 * node_id + 1, depth + 1)
#             right = recurse(getattr(node, "RChild", None), 2 * node_id + 2, depth + 1)
#             decision_node.add_child(left)
#             decision_node.add_child(right)
#             return decision_node
#
#     root = getattr(model, "root_node", None)
#     if root is None:
#         raise ValueError("Model does not have a root_node attribute.")
#
#     return DecisionTree(recurse(root))


from D_oblique_decision_trees.tree import DecisionTree, DecisionNode, LeafNode
import numpy as np

def convert_wodt(model):
    def recurse(node, node_id=0, depth=0):
        if getattr(node, "is_leaf", False):
            if hasattr(node, "class_distribution"):
                prediction = int(np.argmax(node.class_distribution))
            else:
                prediction = 0
            return LeafNode(node_id=node_id, prediction=prediction, depth=depth)
        else:
            split_obj = getattr(node, "split", None)
            if split_obj is None or not hasattr(split_obj, "paras") or not hasattr(split_obj, "threshold"):
                raise ValueError("Node does not contain valid split parameters.")

            weights = np.array(split_obj.paras)
            bias = -split_obj.threshold
            # bias = split_obj.threshold + np.dot(weights, np.array([0.5] * len(weights)))  # <-- key fix

            decision_node = DecisionNode(node_id=node_id, weights=weights, bias=bias, depth=depth)
            left = recurse(getattr(node, "LChild", None), 2 * node_id + 1, depth + 1)
            right = recurse(getattr(node, "RChild", None), 2 * node_id + 2, depth + 1)
            decision_node.add_child(left)
            decision_node.add_child(right)
            return decision_node

    root = getattr(model, "root_node", None)
    if root is None:
        raise ValueError("Model does not have a root_node attribute.")

    return DecisionTree(recurse(root))
