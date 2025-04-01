import os
import time
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from D_oblique_decision_trees.converters.dispatcher import convert_tree
from D_oblique_decision_trees.evaluation.evaluator import evaluate_tree
from Ensembles_of_Oblique_Decision_Trees.Decision_trees.HouseHolder_CART import HHCartClassifier
from Ensembles_of_Oblique_Decision_Trees.Decision_trees.RandCART import RandCARTClassifier
from Ensembles_of_Oblique_Decision_Trees.Decision_trees.Oblique_Classifier_1 import ObliqueClassifier1
from Ensembles_of_Oblique_Decision_Trees.Decision_trees.WODT import WeightedObliqueDecisionTreeClassifier
from Ensembles_of_Oblique_Decision_Trees.Decision_trees.segmentor import MeanSegmentor
from Ensembles_of_Oblique_Decision_Trees.Decision_trees.split_criteria import gini
from src.config.plot_settings import beautify_plot


class DepthSweepRunner:
    def __init__(self, datasets, max_depth=10):
        if isinstance(datasets, dict):
            self.datasets = list(datasets.items())
        else:
            self.datasets = [("dataset_" + str(i), ds) for i, ds in enumerate(datasets)]

        self.max_depth = max_depth
        self.results = []

    @staticmethod
    def _build_registry_for_depth(depth):
        return {
            "hhcart": lambda: HHCartClassifier(impurity=gini, segmentor=MeanSegmentor(), max_depth=depth),
            "randcart": lambda: RandCARTClassifier(impurity=gini, segmentor=MeanSegmentor(), max_depth=depth),
            "oc1": lambda: ObliqueClassifier1(max_depth=depth, min_samples_split=2),
            "wodt": lambda: WeightedObliqueDecisionTreeClassifier(max_depth=depth, min_samples_split=2,
                                                                  max_features='all'),
        }

    def run(self):
        total_loops = len(self.datasets) * self.max_depth * 4
        save_base = os.path.join("..", "_data", "depth_sweep_batch_results", "saved_trees")
        os.makedirs(save_base, exist_ok=True)

        with tqdm(total=total_loops, desc="Depth Sweeping") as pbar:
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
                        metrics["dataset"] = dataset_name
                        metrics["data_dim"] = X.shape[1]
                        metrics["algorithm"] = model_name
                        metrics["depth"] = depth

                        tree_dir = os.path.join(save_base, dataset_name, model_name)
                        os.makedirs(tree_dir, exist_ok=True)
                        tree_path = os.path.join(tree_dir, f"depth_{depth}.pkl")
                        with open(tree_path, "wb") as f:
                            pickle.dump(tree, f)

                        self.results.append(metrics)
                        pbar.update(1)

        return pd.DataFrame(self.results)

    @staticmethod
    def reorder_and_print(df):
        desired_cols = [
            "dataset", "data_dim", "algorithm", "depth",
            "accuracy", "coverage", "density", "f_score",
            "splits", "leaves", "avg_active_feature_count",
            "feature_utilisation_ratio", "tree_level_sparsity_index",
            "composite_interpretability_score", "training_time"
        ]
        existing = [c for c in desired_cols if c in df.columns]
        df = df[existing]

        print("\n===== Depth Sweep Results =====\n")
        print(df)
        return df

    @staticmethod
    def plot_results(df, metric="accuracy", title=None, xlabel="X", ylabel="Y",
                     x_lim=None, y_lim=None, save_name=None):
        if metric not in df.columns:
            print(f"Metric '{metric}' not in columns: {list(df.columns)}")
            return

        # Set global font settings for this plot
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']

        save_dir = os.path.join("..", "_data", "depth_sweep_batch_results")
        os.makedirs(save_dir, exist_ok=True)
        if save_name is None:
            save_name = f"{metric}_vs_depth_per_dataset.pdf"
        save_path = os.path.join(save_dir, save_name)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(data=df, x="depth", y=metric, hue="algorithm", style="dataset", ax=ax)

        # Apply x and y limits if provided
        if x_lim is not None:
            ax.set_xlim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)

        # Set default title if none provided.
        if title is None:
            title = f"{metric.capitalize()} vs. Depth (per dataset)"

        beautify_plot(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, save_path=save_path)

    @staticmethod
    def plot_aggregated_metric(df, metric="accuracy", title=None, xlabel="X", ylabel="Y",
                               x_lim=None, y_lim=None, save_name=None):
        if metric not in df.columns:
            print(f"Metric '{metric}' not found in columns: {list(df.columns)}")
            return

        agg_df = df.groupby(["algorithm", "depth"], as_index=False)[metric].mean()

        # Set global font settings for this plot.
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']

        save_dir = os.path.join("..", "_data", "depth_sweep_batch_results")
        os.makedirs(save_dir, exist_ok=True)
        if save_name is None:
            save_name = f"{metric}_vs_depth_mean.pdf"
        save_path = os.path.join(save_dir, save_name)

        fig, ax = plt.subplots(figsize=(8, 5))
        for algo in agg_df["algorithm"].unique():
            sub = agg_df[agg_df["algorithm"] == algo]
            ax.plot(sub["depth"], sub[metric], label=algo)

        # Apply x and y limits if provided
        if x_lim is not None:
            ax.set_xlim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)

        # Set default title if none provided.
        if title is None:
            title = f"Mean {metric.capitalize()} vs. Depth (averaged over datasets)"

        beautify_plot(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, save_path=save_path)

    @staticmethod
    def save_results(df, filename="depth_sweep_result.csv"):
        save_dir = os.path.join("..", "_data", "depth_sweep_batch_results")
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        df.to_csv(path, index=False)
        print(f"Saved results to: {path}")

    @staticmethod
    def load_results(filename):
        return pd.read_csv(filename)
