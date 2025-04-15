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
from C_oblique_decision_trees.evaluation.evaluator import evaluate_tree
from C_oblique_decision_trees.evaluation.io_utils import save_depth_sweep_df, save_trees_dict
from Ensembles_of_Oblique_Decision_Trees.Decision_trees.HouseHolder_CART import HHCartClassifier
from Ensembles_of_Oblique_Decision_Trees.Decision_trees.RandCART import RandCARTClassifier
from Ensembles_of_Oblique_Decision_Trees.Decision_trees.Oblique_Classifier_1 import ObliqueClassifier1
from Ensembles_of_Oblique_Decision_Trees.Decision_trees.WODT import WeightedObliqueDecisionTreeClassifier
from Ensembles_of_Oblique_Decision_Trees.Decision_trees.segmentor import MeanSegmentor
from Ensembles_of_Oblique_Decision_Trees.Decision_trees.split_criteria import gini
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
    def build_registry(random_state=None, impurity=gini, segmentor=MeanSegmentor()):
        """
        Returns a registry of constructors that take depth and return models.
        Supports passing impurity function and segmentor instance.
        """

        def make(cls, **kwargs):
            return lambda depth: cls(max_depth=depth, random_state=random_state, **kwargs)

        return {
            "hhcart": make(HHCartClassifier, impurity=impurity, segmentor=segmentor),
            "randcart": make(RandCARTClassifier, impurity=impurity, segmentor=segmentor),
            "oc1": make(ObliqueClassifier1),
            "wodt": make(WeightedObliqueDecisionTreeClassifier, max_features='all'),
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
                    model = constructor(self.max_depth)
                    model.fit(X, y)

                    full_tree = convert_tree(model, model_type=model_name)
                    true_max_depth = full_tree.max_depth

                    for depth in range(0, min(self.max_depth, true_max_depth) + 1):
                        model_at_depth = constructor(depth)

                        # FIT
                        fit_start = time.perf_counter()
                        model_at_depth.fit(X, y)
                        fit_end = time.perf_counter()

                        # CONVERT
                        tree_at_depth = convert_tree(model_at_depth, model_type=model_name)

                        # EVALUATE
                        metrics = evaluate_tree(tree_at_depth, X, y)

                        # Timing results
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
                "gini_coverage_class_1_leaves", "gini_density_class_1_leaves",
                "splits", "leaves", "runtime"
            ]

            # desired_cols = [
            #     "seed", "dataset", "data_dim", "algorithm", "depth",
            #     "accuracy", "coverage", "density", "f_score",
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
