"""
evaluation

This module provides a framework for evaluating oblique decision trees and traditional scenario discovery methods.
It includes standardized metrics, evaluation routines, and benchmark execution across datasets and algorithms.
"""

from .metrics import (
    compute_accuracy,
    compute_coverage,
    compute_density,
    compute_f_score,
    interpretability_score,
    sparsity,
    compute_convergence_length,
    compute_training_time,
    seed_stability,
)

from .evaluator import evaluate_tree
from .benchmark_runner import DepthSweepRunner
