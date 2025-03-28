"""
regularizers.py

This module provides an implementation of the L1 regularizer used in the TAO optimization
process. The L1 regularizer enforces sparsity in the weight vectors of decision nodes,
which is critical for node pruning in TAO. Both a function-based implementation and a
class-based implementation of the L1 regularizer are provided.
"""

import numpy as np


def l1_regularizer(weights, lambda_reg=0.01):
    """
    Compute the L1 regularization cost for a given weight vector.

    The L1 penalty is defined as:
        lambda_reg * sum(|weights|)

    Args:
        weights (np.array): Weight vector (or any numpy array of parameters).
        lambda_reg (float): Regularization strength.

    Returns:
        float: The L1 penalty.
    """
    return lambda_reg * np.sum(np.abs(weights))


def l1_subgradient(weights, lambda_reg=0.01, threshold=1e-8):
    """
    Compute the subgradient of the L1 regularizer for a given weight vector.

    L1 regularization is non-differentiable at zero. The subgradient is defined as:
        grad_i = lambda_reg * sign(w_i) if |w_i| >= threshold,
        grad_i = 0 if |w_i| < threshold.

    The threshold is used to avoid instability near zero.

    Args:
        weights (np.array): Weight vector.
        lambda_reg (float): Regularization strength.
        threshold (float): A small value below which a weight is considered zero.

    Returns:
        np.array: The subgradient vector of the same shape as weights.
    """
    grad = lambda_reg * np.sign(weights)
    grad[np.abs(weights) < threshold] = 0.0
    return grad


class L1Regularizer:
    """
    Class-based implementation of the L1 regularizer.

    This regularizer promotes sparsity in weight vectors, which is critical in the TAO algorithm
    since zero weights lead to node pruning.
    """
    def __init__(self, lambda_reg=0.01, threshold=1e-8):
        """
        Initialize the L1Regularizer.

        Args:
            lambda_reg (float): Regularization strength.
            threshold (float): A small value below which weights are considered zero.
        """
        self.lambda_reg = lambda_reg
        self.threshold = threshold

    def penalty(self, weights):
        """
        Compute the L1 penalty for the given weights.

        Args:
            weights (np.array): Weight vector.

        Returns:
            float: The L1 penalty.
        """
        return self.lambda_reg * np.sum(np.abs(weights))

    def gradient(self, weights):
        """
        Compute the subgradient of the L1 penalty for the given weights.

        Args:
            weights (np.array): Weight vector.

        Returns:
            np.array: The subgradient.
        """
        grad = self.lambda_reg * np.sign(weights)
        grad[np.abs(weights) < self.threshold] = 0.0
        return grad
