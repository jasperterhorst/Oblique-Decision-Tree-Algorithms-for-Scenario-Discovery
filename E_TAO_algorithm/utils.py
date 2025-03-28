"""
utils.py

This module provides utility functions for the TAO package. The functions in this module
allow for computing the reduced set of training examples that reach each node in a decision tree,
pruning dead nodes (nodes that receive no examples), and recursively traversing a tree to perform
an action on each node.
"""


def compute_reduced_sets(tree, x):
    """
    Compute the reduced set for each node in a tree.

    The reduced set for a node is defined as the list of indices of the training examples in x
    that reach that node.

    Parameters:
        tree (DecisionTree): The decision tree.
        x (np.array): Training features with shape (N, D).

    Returns:
        dict: A dictionary mapping each node (key) to a list of example indices.
    """
    reduced_sets = {}

    def traverse_sample(data_item, index_val, node):
        # Add the current index to the node's reduced set.
        if node in reduced_sets:
            reduced_sets[node].append(index_val)
        else:
            reduced_sets[node] = [index_val]
        # If the node is not a leaf, propagate the sample to the appropriate child.
        if not node.is_leaf():
            branch = node.decision(data_item)
            # Only propagate if the node has at least two children.
            if len(node.children) >= 2:
                traverse_sample(data_item, index_val, node.children[branch])

    for idx, sample in enumerate(x):
        traverse_sample(sample, idx, tree.root)

    return reduced_sets


def prune_dead_nodes(tree, reduced_sets):
    """
    Prune dead nodes from a tree.

    A dead node is one receiving no training examples. This function recursively removes
    any branch where a node's reduced set is empty.

    Parameters:
        tree (DecisionTree): The decision tree.
        reduced_sets (dict): The reduced sets for each node as computed by compute_reduced_sets().

    Returns:
        DecisionTree: The pruned tree.
    """
    def prune_node(node):
        if not node.is_leaf():
            pruned_children = []
            for child in node.children:
                if child in reduced_sets and len(reduced_sets[child]) > 0:
                    prune_node(child)
                    pruned_children.append(child)
            node.children = pruned_children

    prune_node(tree.root)
    return tree


def traverse_tree(tree, action):
    """
    Recursively traverse a tree and apply a given action to each node.

    Parameters:
        tree (DecisionTree): The decision tree.
        action (function): A function that accepts a node as its argument.
    """
    def _traverse(node):
        action(node)
        for child in node.children:
            _traverse(child)
    _traverse(tree.root)
