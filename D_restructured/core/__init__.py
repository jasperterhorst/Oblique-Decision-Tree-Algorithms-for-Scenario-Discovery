"""
Core package for D_oblique_decision_trees.

Exports the main tree structures for building and working with decision trees.
"""

from .tree import TreeNode, DecisionNode, LeafNode, DecisionTree

__all__ = ["TreeNode", "DecisionNode", "LeafNode", "DecisionTree"]
