"""
losses.py

This module provides loss functions for binary classification. It includes the
0/1 loss, which directly measures misclassification error, and the logistic loss,
which is a smooth surrogate used for optimization. Also, the module
provides a function to compute the gradient of the logistic loss.
"""

import numpy as np


def zero_one_loss(y_true, y_pred):
    """
    Compute the 0/1 classification loss.

    The 0/1 loss measures the fraction of misclassified examples.

    Parameters:
        y_true (np.array): True labels (e.g., integers 0 or 1).
        y_pred (np.array): Predicted labels.

    Returns:
        float: The average 0/1 loss.
    """
    return np.mean(y_true != y_pred)


def logistic_loss(s, y):
    """
    Compute the logistic loss for binary classification.

    The logistic loss is a smooth surrogate for the 0/1 loss. It is defined as:
        L(s, y) = log(1 + exp(-y' × s))
    where y' = 2 × y – 1 transforms labels {0, 1} to {-1, +1}.

    Parameters:
        s (np.array): Prediction scores (linear outputs) from the model.
        y (np.array): True binary labels (0 or 1).

    Returns:
        np.array: The logistic loss for each instance.
    """
    y_prime = 2 * y - 1
    return np.log(1 + np.exp(-y_prime * s))


def logistic_loss_gradient(s, y):
    """
    Compute the gradient of the logistic loss with respect to s.

    The gradient is given by:
        dL/ds = -y' * (1 – sigma(s))
    where y' = 2 × y – 1 and sigma(s) = 1 / (1 + exp(-s)).

    Parameters:
        s (np.array): Prediction scores.
        y (np.array): True labels in {0, 1}.

    Returns:
        np.array: The gradient of the logistic loss with respect to s.
    """
    y_prime = 2 * y - 1
    sigma = 1 / (1 + np.exp(-s))
    return -y_prime * (1 - sigma)
