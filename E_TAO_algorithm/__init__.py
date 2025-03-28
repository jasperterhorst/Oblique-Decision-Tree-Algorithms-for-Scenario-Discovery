"""
E_TAO_package: A TAO implementation for optimizing oblique decision trees
with constant leaves using a 0/1 classification loss (via a logistic surrogate)
and L1 regularization to induce sparsity and pruning.
"""

__version__ = "0.1.0"

from .tree import TreeNode, DecisionNode, LeafNode, DecisionTree
from .optimizer import TAOOptimizer
from .losses import zero_one_loss, logistic_loss, logistic_loss_gradient
from .regularizers import l1_regularizer, l1_subgradient, L1Regularizer
from .utils import compute_reduced_sets, prune_dead_nodes, traverse_tree
