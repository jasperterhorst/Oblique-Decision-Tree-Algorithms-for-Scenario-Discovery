"""
Core package for C_oblique_decision_tree_benchmark.

Exports the main tree structures for building and working with decision trees.
"""

from .tree import TreeNode, DecisionNode, LeafNode, DecisionTree

__all__ = ["TreeNode", "DecisionNode", "LeafNode", "DecisionTree"]
