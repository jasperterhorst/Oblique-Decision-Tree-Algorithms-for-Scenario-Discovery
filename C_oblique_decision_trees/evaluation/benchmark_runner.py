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

from C_oblique_decision_trees.converters.dispatcher import convert_tree

from C_oblique_decision_trees.evaluation import evaluate_tree, save_depth_sweep_df, save_trees_dict

from _adopted_oblique_trees.HouseHolder_CART import HHCartAClassifier, HHCartDClassifier
from _adopted_oblique_trees.RandCART import RandCARTClassifier
from _adopted_oblique_trees.CO2 import CO2Classifier
from _adopted_oblique_trees.Oblique_Classifier_1 import ObliqueClassifier1
from _adopted_oblique_trees.WODT import WeightedObliqueDecisionTreeClassifier
from _adopted_oblique_trees.RidgeCART import RidgeCARTClassifier
from _adopted_oblique_trees.CART import CARTClassifier

from _adopted_oblique_trees.segmentor import CARTSegmentor, MeanSegmentor
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
            random_state=None,
            impurity=gini,
            segmentor=CARTSegmentor(),
            n_restarts=20,
            bias_steps=20,
            max_iter_per_node=10,
            tau=1e-6,
            nu=1.0,
            eta=0.01,
            tol=1e-3,
            n_rotations=1,
            max_features='all',
            min_features_split=1,
            min_samples_split=2,
    ):
        """
        Constructs a registry of oblique decision tree classifiers, each wrapped in a lambda
        that accepts tree depth and returns an initialized model. This setup is used for
        benchmarking various models in a depth sweep.

        Parameters:
            random_state (int): Controls reproducibility. Passed to all models that support it.

            impurity (callable): Impurity function used to evaluate splits (e.g., Gini, entropy).
                Used by CO2, HHCART (A and D), and RandCART. Default is the standard Gini index.

            segmentor (object): Object that implements the split search strategy.
                Passed to HHCART (for axis-aligned splits in reflected space), CO2 (to define initial splits),
                and RandCART (for axis-aligned splits in rotated space).
                Typically a CART-based splitter or a MeanSegmentor.

            n_restarts (int): Number of random restarts for hyperplane optimization in OC1.
                Controls how many times the algorithm attempts to escape local minima.

            bias_steps (int): Number of bias perturbation steps used in OC1's local search process.

            tau (float): Tolerance for convergence of HHCART classifiers.

            tol (float): Tolerance for convergence of CO2 classifiers.

            max_iter_per_node (int): Maximum number of iterations for hyperplane optimization in CO2.

            nu (float): Smoothing constant for the convex-concave surrogate loss in CO2.
                Influences how strongly the surrogate bounds the true classification error.

            eta (float): Learning rate for CO2’s stochastic gradient descent.
                Determines the step size for updating hyperplane parameters.

            n_rotations (int): Number of random rotations for the data in RandCART.

            max_features (int, float, or 'all'): Used by WODT to control the number of features
                randomly selected at each split. Supports both absolute counts and fractional values.

            min_features_split (int): Minimum number of non-zero coefficients required in an OC1 split.
                Enforces sparsity by ensuring splits involve at least a certain number of input features.

            min_samples_split (int): Minimum number of samples required to split an internal node.

        Returns:
            dict: A model registry mapping string identifiers (e.g., 'co2', 'oc1') to functions
                  that take depth as input and return initialized classifier instances.
        """

        def make(cls, **kwargs):
            return lambda depth: cls(max_depth=depth, random_state=random_state, **kwargs)

        return {
            "hhcart_a": make(HHCartAClassifier, impurity=impurity, segmentor=segmentor,
                             tau=tau, min_samples_split=min_samples_split),
            "hhcart_d": make(HHCartDClassifier, impurity=impurity, segmentor=segmentor,
                             tau=tau, min_samples_split=min_samples_split),
            "randcart": make(RandCARTClassifier, impurity=impurity, segmentor=CARTSegmentor(),
                             min_samples_split=min_samples_split, n_rotations=n_rotations),
            "oc1": make(ObliqueClassifier1, n_restarts=n_restarts, bias_steps=bias_steps,
                        min_features_split=min_features_split),
            "wodt": make(WeightedObliqueDecisionTreeClassifier, max_features=max_features,
                         min_samples_split=min_samples_split),
            "co2": make(CO2Classifier, impurity=impurity, segmentor=segmentor,
                        max_iter_per_node=max_iter_per_node, nu=nu, eta=eta,
                        min_samples_split=min_samples_split, tol=tol),
            "cart": make(CARTClassifier, impurity=impurity, min_samples_split=min_samples_split),
            "ridge_cart": make(RidgeCARTClassifier, impurity=impurity, segmentor=segmentor,
                               min_samples_split=min_samples_split),
        }

    def run(self, auto_export=True, filename="result.csv", tree_dict_filename="result.pkl",
            n_seeds=1, fixed_seed=None, return_trees=True, registry=None, save_tree_dict=True, batch_mode=False,
            output_subfolder=None, n_restarts=10, bias_steps=10):
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
            n_restarts (int): Number of restarts of hyperplanes in OC1.
            bias_steps (int): Number changes in the bias for OC1.

        Returns:
            (DataFrame, dict): Results and trained trees (if return_trees).
        """

        seeds = [fixed_seed] if fixed_seed is not None else DEFAULT_VARIABLE_SEEDS[:n_seeds]

        run_type = "batch" if batch_mode else "single"

        if registry is None:
            registry = self.build_registry(random_state=fixed_seed or 42, n_restarts=n_restarts, bias_steps=bias_steps)

        trees_dict = {}
        total_loops = len(seeds) * len(registry) * len(self.datasets) * (self.max_depth + 1)
        pbar = tqdm(total=total_loops, desc="Depth Sweeping")

        for seed in seeds:
            np.random.seed(seed)
            random.seed(seed)

            for model_name, constructor in registry.items():
                for dataset_name, (X, y) in self.datasets:
                    # Step 1: Get true maximum depth after fitting full tree once
                    model_full = constructor(self.max_depth)
                    model_full.fit(X, y)
                    full_tree = convert_tree(model_full, model_type=model_name)
                    true_max_depth = full_tree.max_depth

                    # Step 2: Loop through depths from 0 to true_max_depth
                    for depth in range(0, true_max_depth + 1):
                        model_at_depth = constructor(depth)
                        fit_start = time.perf_counter()
                        model_at_depth.fit(X, y)
                        fit_end = time.perf_counter()

                        tree_at_depth = convert_tree(model_at_depth, model_type=model_name)
                        metrics = evaluate_tree(tree_at_depth, X, y)

                        metrics["runtime"] = fit_end - fit_start
                        metrics.update({
                            "seed": seed,
                            "dataset": dataset_name,
                            "data_dim": X.shape[1],
                            "algorithm": model_name,
                            "depth": depth
                        })

                        trees_dict.setdefault((model_name, dataset_name, seed), {})[depth] = tree_at_depth
                        self.results.append(metrics)
                        pbar.update(1)

        df = pd.DataFrame(self.results)

        # Standard output ordering
        if auto_export:

            desired_cols = [
                "seed", "dataset", "data_dim", "algorithm", "depth",
                "accuracy", "coverage", "density", "f_score",
                "gini_coverage_all_leaves", "gini_density_all_leaves",
                "splits", "leaves",
                "total_active_feature_count", "avg_active_feature_count",
                "runtime"
            ]

            # desired_cols = [
            #     "seed", "dataset", "data_dim", "algorithm", "depth",
            #     "accuracy", "coverage", "density", "f_score",
            #     gini_coverage_all_leaves", "gini_density_all_leaves",
            #     "splits", "leaves", "avg_active_feature_count",
            #     "feature_utilisation_ratio", "tree_level_sparsity_index",
            #     "composite_interpretability_score", "run_time"
            # ]

            df = df[[col for col in desired_cols if col in df.columns]]
            save_depth_sweep_df(df, filename=filename, run_type=run_type, subdir=output_subfolder)

        if return_trees and save_tree_dict:
            if not tree_dict_filename:
                tree_dict_filename = filename.replace(".csv", ".pkl")
            save_trees_dict(trees_dict, filename=tree_dict_filename, run_type=run_type, subdir=output_subfolder)

        return (df, trees_dict) if return_trees else df
