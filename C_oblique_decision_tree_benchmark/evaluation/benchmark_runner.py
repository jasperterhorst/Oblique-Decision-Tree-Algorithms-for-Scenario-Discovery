"""
Benchmark Runner for Decision Tree Depth Sweeping.

Defines the DepthSweepRunner class to benchmark various decision tree models
across different depths and datasets.
"""

import time
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

from C_oblique_decision_tree_benchmark.converters.dispatcher import convert_tree

from C_oblique_decision_tree_benchmark.evaluation import evaluate_tree, save_depth_sweep_df, save_trees_dict

from _adopted_oblique_trees.HouseHolder_CART import HHCartAClassifier, HHCartDClassifier
from _adopted_oblique_trees.RandCART import RandCARTClassifier
from _adopted_oblique_trees.CO2 import CO2Classifier
from _adopted_oblique_trees.Modified_Oblique_Classifier_1 import ModifiedObliqueClassifier1
from _adopted_oblique_trees.WODT import WeightedObliqueDecisionTreeClassifier
from _adopted_oblique_trees.RidgeCART import RidgeCARTClassifier
from _adopted_oblique_trees.CART import CARTClassifier
from _adopted_oblique_trees.segmentor import CARTSegmentor
from _adopted_oblique_trees.split_criteria import gini

from src.config.settings import DEFAULT_VARIABLE_SEEDS


class DepthSweepRunner:
    """
    Runs depth sweep experiments to benchmark decision tree models.
    Grows trees progressively from depth 0 to the maximum depth and evaluates them.
    """

    def __init__(self, datasets, max_depth=10):
        """
        Initialize the depth sweep runner.

        Parameters:
            datasets (dict or list): A dict mapping dataset names to (X, y) tuples
                                      or a list of (X, y) tuples.
            max_depth (int): Maximum tree depth to test.
        """
        if isinstance(datasets, dict):
            self.datasets = list(datasets.items())
        else:
            self.datasets = [("dataset_" + str(i), ds) for i, ds in enumerate(datasets)]

        self.max_depth = max_depth
        self.results = []

    @staticmethod
    def build_registry(
            random_state=None, impurity=gini, segmentor=CARTSegmentor(), n_restarts=20, bias_steps=20,
            tau=0.05, nu=1.0, eta=0.01, tol=1e-3, n_rotations=1, max_features='all', min_samples_split=2,
            max_iter_global=1000000, max_iter_wodt=None, max_iter_co2=None,
    ):
        """
        Constructs a registry of oblique decision tree classifiers, each wrapped in a lambda
        that accepts tree depth and returns an initialized model. This setup supports
        benchmarking various models in a depth sweep.

        Parameters:

        # General Controls
            random_state (int or None):
                Seed for reproducibility. Use None for stochastic behaviour.

            min_samples_split (int, default=2):
                Minimum number of samples required to perform a split. Must be ≥ 2.

        # Iteration Control
        max_iter_global (int, default=1000):
            Default max optimisation iterations used by CO2 and WODT unless overridden.

        # Split Quality & Search Strategies
            impurity (callable):
                Function to evaluate split quality (e.g., gini, entropy).
                Applies to: HHCART_SD variants, CO2, RandCART, RidgeCART, CART.

            segmentor (object):
                Defines axis-aligned split search (e.g., CARTSegmentor, MeanSegmentor).
                Applies to: HHCART_SD variants, CO2, RandCART, RidgeCART.

        # MOC1 Hyperparameters
            n_restarts (int, default=20):
                Number of random restarts (≥ 1) to escape local minima.

        # HHCART_SD (A and D) Controls
            tau (float, default=1e-6):
                Reflection tolerance. Small positive value to skip near-axis eigenvectors.
                Applies to: HHCART_SD(A) and HHCART_SD(D).

        # CO2-Specific Hyperparameters
            max_iter_co2 (int, optional):
                Overrides global max_iter for CO2. If None, uses `max_iter_global`.

            nu (float, default=1.0):
                Positive smoothing parameter for surrogate loss.

            eta (float, default=0.01):
                Learning rate > 0 for gradient descent.

            tol (float, default=1e-3):
                Convergence tolerance > 0 for stopping criterion.

        # RandCART Controls
            n_rotations (int, default=1):
                Number of random rotations ≥ 1 applied before splitting.

        # WODT Controls
            max_iter_wodt (int, optional):
                Overrides global max_iter for WODT. If None, uses `max_iter_global`.

            max_features (int, float, or 'all', default='all'):
                Number of features to consider at each split.
                - If int: absolute count.
                - If float: fraction (0 < value ≤ 1).
                - If 'all': use all features.

        Returns:
            dict: A model registry mapping string identifiers (e.g., 'co2', 'sparse_oc1')
                  to functions that take depth as input and return initialized classifier instances.
        """
        def make(cls, **kwargs):
            return lambda depth: cls(max_depth=depth, random_state=random_state, min_samples_split=min_samples_split,
                                     **kwargs)

        return {
            "hhcart_a": make(HHCartAClassifier, impurity=impurity, segmentor=segmentor, tau=tau),
            "hhcart_d": make(HHCartDClassifier, impurity=impurity, segmentor=segmentor, tau=tau),
            "randcart": make(RandCARTClassifier, impurity=impurity, segmentor=CARTSegmentor(), n_rotations=n_rotations),
            "wodt": make(WeightedObliqueDecisionTreeClassifier, max_features=max_features,
                         max_iter=max_iter_wodt if max_iter_wodt is not None else max_iter_global,),
            "cart": make(CARTClassifier, impurity=impurity),
            "ridge_cart": make(RidgeCARTClassifier, impurity=impurity, segmentor=segmentor),
            "moc1": make(ModifiedObliqueClassifier1, n_restarts=n_restarts, bias_steps=bias_steps),
            "co2": make(CO2Classifier, impurity=impurity, segmentor=segmentor,
                        max_iter_per_node=max_iter_co2 if max_iter_co2 is not None else max_iter_global,
                        nu=nu, eta=eta, tol=tol),
        }

    def run(self, auto_export=True, filename="result.csv", tree_dict_filename="result.pkl",
            n_seeds=1, fixed_seed=None, return_trees=True, registry=None, save_tree_dict=True, batch_mode=False,
            output_subfolder=None):
        """
        Run depth sweeps, evaluating all models and saving results.

        Args:
            auto_export (bool): Save CSV after run.
            filename (str): CSV filename for results.
            tree_dict_filename (str): Filename for tree dict.
            n_seeds (int): Number of random seeds to use.
            fixed_seed (int): Fixed seed (overrides n_seeds).
            return_trees (bool): Whether to return the trained trees.
            registry (dict): Custom model registry.
            save_tree_dict (bool): Whether to persist the trees.
            batch_mode (bool): If True, save outputs to batch results folder.
            output_subfolder (str): Subdirectory to store outputs.
        Returns:
            (DataFrame, dict): Results and trained trees (if return_trees).
        """

        seeds = [fixed_seed] if fixed_seed is not None else DEFAULT_VARIABLE_SEEDS[:n_seeds]

        run_type = "batch" if batch_mode else "single"

        if registry is None:
            registry = self.build_registry(random_state=fixed_seed or 42)

        trees_dict = {}
        total_loops = len(seeds) * len(registry) * len(self.datasets) * (self.max_depth + 1)
        pbar = tqdm(total=total_loops, desc="Depth Sweeping")

        for seed in seeds:
            np.random.seed(seed)
            random.seed(seed)

            for model_name, constructor in registry.items():
                for dataset_name, (X, y) in self.datasets:
                    # Ensure X is a NumPy array
                    X_np = X.values if isinstance(X, pd.DataFrame) else X

                    # Step 1: Get true maximum depth after fitting full tree once
                    model_full = constructor(self.max_depth)
                    model_full.fit(X_np, y)
                    full_tree = convert_tree(model_full, model_type=model_name)
                    true_max_depth = full_tree.max_depth

                    # Step 2: Loop through depths from 0 to true_max_depth
                    for depth in range(0, true_max_depth + 1):
                        model_at_depth = constructor(depth)
                        fit_start = time.perf_counter()
                        model_at_depth.fit(X_np, y)
                        fit_end = time.perf_counter()

                        tree_at_depth = convert_tree(model_at_depth, model_type=model_name)
                        metrics = evaluate_tree(tree_at_depth, X_np, y)  # Use X_np here

                        # Parse shape and label noise
                        shape_type = dataset_name.split("_label_noise")[0]
                        try:
                            label_noise_str = dataset_name.split("_label_noise_")[1].split("_")[0]
                            label_noise = float(label_noise_str) / 100  # Assuming naming like label_noise_003
                        except IndexError:
                            label_noise = 0.0  # Default if not present

                        metrics.update({
                            "shape": shape_type,
                            "label_noise": label_noise,
                            "n_samples": X.shape[0],
                            "data_dim": X.shape[1],
                            "seed": seed,
                            "algorithm": model_name,
                            "depth": depth,
                            "runtime": fit_end - fit_start,
                        })

                        trees_dict.setdefault((model_name, dataset_name, seed), {})[depth] = tree_at_depth
                        self.results.append(metrics)
                        pbar.update(1)

        df = pd.DataFrame(self.results)

        # Standard output ordering
        if auto_export:
            base_cols = [
                "shape", "label_noise", "n_samples",
                "seed", "data_dim", "algorithm", "depth",
                "accuracy", "coverage", "density", "f_score",
                "gini_coverage_all_leaves", "gini_density_all_leaves",
                "splits", "leaves",
                "total_active_feature_count", "avg_active_feature_count",
                "runtime"
            ]

            df = df[[col for col in base_cols if col in df.columns]]

            save_depth_sweep_df(df, filename=filename, run_type=run_type, subdir=output_subfolder)

        if return_trees and save_tree_dict:
            if not tree_dict_filename:
                tree_dict_filename = filename.replace(".csv", ".pkl")
            save_trees_dict(trees_dict, filename=tree_dict_filename, run_type=run_type, subdir=output_subfolder)

        return (df, trees_dict) if return_trees else df
