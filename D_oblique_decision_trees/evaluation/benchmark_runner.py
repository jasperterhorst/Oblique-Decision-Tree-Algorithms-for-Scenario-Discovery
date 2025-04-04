"""
Benchmark Runner for Decision Tree Depth Sweeping.

Defines the DepthSweepRunner class to benchmark various decision tree models
across different depths and datasets.
"""

import os
import time
import pickle
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

from D_oblique_decision_trees.converters.dispatcher import convert_tree
from D_oblique_decision_trees.evaluation.evaluator import evaluate_tree
from Ensembles_of_Oblique_Decision_Trees.Decision_trees.HouseHolder_CART import HHCartClassifier
from Ensembles_of_Oblique_Decision_Trees.Decision_trees.RandCART import RandCARTClassifier
from Ensembles_of_Oblique_Decision_Trees.Decision_trees.Oblique_Classifier_1 import ObliqueClassifier1
from Ensembles_of_Oblique_Decision_Trees.Decision_trees.WODT import WeightedObliqueDecisionTreeClassifier
from Ensembles_of_Oblique_Decision_Trees.Decision_trees.segmentor import MeanSegmentor
from Ensembles_of_Oblique_Decision_Trees.Decision_trees.split_criteria import gini
from src.config.settings import DEFAULT_VARIABLE_SEEDS
from src.config.paths import DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR


class DepthSweepRunner:
    """
    Runs depth sweep experiments to benchmark decision tree models.
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
    def _build_registry_for_depth(depth):
        """
        Build a registry of model constructors for a given tree depth.

        Returns:
            dict: Mapping of model type strings with functions instantiating models.
        """
        return {
            "hhcart": lambda: HHCartClassifier(impurity=gini, segmentor=MeanSegmentor(), max_depth=depth),
            "randcart": lambda: RandCARTClassifier(impurity=gini, segmentor=MeanSegmentor(), max_depth=depth),
            "oc1": lambda: ObliqueClassifier1(max_depth=depth, min_samples_split=2),
            "wodt": lambda: WeightedObliqueDecisionTreeClassifier(max_depth=depth, min_samples_split=2,
                                                                  max_features='all'),
        }

    def run(self, auto_export=True, filename="depth_sweep_result.csv", n_seeds=1, fixed_seed=None):
        """
        Execute depth sweep experiments over the specified datasets, model types, and random seeds.

        Parameters:
            auto_export (bool): If True, the results will be reordered, printed, and saved.
            filename (str): The filename for saving the CSV results.
            n_seeds (int): Number of seeds to vary over (taken from settings.vary_seeds)

        Returns:
            pd.DataFrame: A DataFrame containing the benchmark results.
        """
        seeds = [fixed_seed] if fixed_seed is not None else DEFAULT_VARIABLE_SEEDS[:n_seeds]
        total_loops = len(self.datasets) * (self.max_depth + 1) * len(seeds) * 4

        save_base = os.path.join(DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR, "saved_trees")
        os.makedirs(save_base, exist_ok=True)

        pbar = tqdm(total=total_loops, desc="Depth Sweeping")

        for seed in seeds:
            np.random.seed(seed)  # for other algorithms
            random.seed(seed)  # for WODT
            for depth in range(0, self.max_depth + 1):
                registry = self._build_registry_for_depth(depth)
                for model_name, constructor in registry.items():
                    for dataset_name, (X, y) in self.datasets:
                        model = constructor()
                        start_t = time.time()
                        model.fit(X, y)
                        end_t = time.time()

                        tree = convert_tree(model, model_type=model_name)
                        metrics = evaluate_tree(tree, X, y, training_time=(end_t - start_t))
                        metrics["seed"] = seed
                        metrics["dataset"] = dataset_name
                        metrics["data_dim"] = X.shape[1]
                        metrics["algorithm"] = model_name
                        metrics["depth"] = depth

                        tree_dir = os.path.join(save_base, dataset_name, model_name)
                        os.makedirs(tree_dir, exist_ok=True)
                        tree_path = os.path.join(tree_dir, f"seed_{seed}_depth_{depth}.pkl")
                        with open(tree_path, "wb") as f:
                            pickle.dump(tree, f)

                        self.results.append(metrics)
                        pbar.update(1)

        df = pd.DataFrame(self.results)

        if auto_export:
            # Reorder columns
            desired_cols = [
                "seed", "dataset", "data_dim", "algorithm", "depth",
                "accuracy", "coverage", "density", "f_score",
                "splits", "leaves", "avg_active_feature_count",
                "feature_utilisation_ratio", "tree_level_sparsity_index",
                "composite_interpretability_score", "training_time"
            ]
            existing = [col for col in desired_cols if col in df.columns]
            df = df[existing]

            existing = [col for col in desired_cols if col in df.columns]
            df = df[existing]
            print("\n===== Depth Sweep Results =====\n")
            print(df)

            path = os.path.join(DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR, filename)
            df.to_csv(path, index=False)
            print(f"Saved results to: {path}")

        return df
