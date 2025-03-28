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

class DepthSweepRunner:
    def __init__(self, datasets, max_depth=10):
        if isinstance(datasets, dict):
            self.datasets = list(datasets.items())
        else:
            self.datasets = [("dataset_" + str(i), ds) for i, ds in enumerate(datasets)]

        self.max_depth = max_depth
        self.results = []

    def _build_registry_for_depth(self, depth):
        return {
            "hhcart": lambda: HHCartClassifier(impurity=gini, segmentor=MeanSegmentor(), max_depth=depth),
            "randcart": lambda: RandCARTClassifier(impurity=gini, segmentor=MeanSegmentor(), max_depth=depth),
            "oc1": lambda: ObliqueClassifier1(max_depth=depth, min_samples_split=2),
            "wodt": lambda: WeightedObliqueDecisionTreeClassifier(max_depth=depth, min_samples_split=2, max_features='all'),
        }

    def run(self):
        total_loops = len(self.datasets) * self.max_depth * 4  # 4 models by default
        os.makedirs("saved_trees", exist_ok=True)

        with tqdm(total=total_loops, desc="Depth Sweeping") as pbar:
            for depth in range(1, self.max_depth + 1):
                registry = self._build_registry_for_depth(depth)
                for model_name, constructor in registry.items():
                    for dataset_name, (X, y) in self.datasets:
                        model = constructor()
                        start_t = time.time()
                        model.fit(X, y)
                        end_t = time.time()

                        tree = convert_tree(model, model_type=model_name)
                        metrics = evaluate_tree(tree, X, y, training_time=(end_t - start_t))
                        metrics["algorithm"] = model_name
                        metrics["dataset"] = dataset_name
                        metrics["depth"] = depth

                        tree_dir = os.path.join("saved_trees", dataset_name, model_name)
                        os.makedirs(tree_dir, exist_ok=True)
                        tree_path = os.path.join(tree_dir, f"depth_{depth}.pkl")
                        with open(tree_path, "wb") as f:
                            pickle.dump(tree, f)

                        self.results.append(metrics)
                        pbar.update(1)

        return pd.DataFrame(self.results)

    def reorder_and_print(self, df):
        desired_cols = [
            "dataset", "algorithm", "depth",
            "accuracy", "coverage", "density", "f_score",
            "splits", "leaves", "interpretability", "sparsity",
            "training_time"
        ]
        existing = [c for c in desired_cols if c in df.columns]
        df = df[existing]

        print("\n===== Depth Sweep Results =====\n")
        print(df)
        return df

    def plot_results(self, df, metric="accuracy"):
        if metric not in df.columns:
            print(f"Metric '{metric}' not in columns: {list(df.columns)}")
            return

        plt.figure(figsize=(8, 5))
        sns.lineplot(data=df, x="depth", y=metric, hue="algorithm", style="dataset")
        plt.title(f"{metric.capitalize()} vs. Depth (per dataset)")
        plt.ylim(0, 1)
        plt.show()

    def plot_aggregated_metric(self, df, metric="accuracy"):
        if metric not in df.columns:
            print(f"Metric '{metric}' not found in columns: {list(df.columns)}")
            return

        agg_df = df.groupby(["algorithm", "depth"], as_index=False)[metric].mean()

        plt.figure(figsize=(8, 5))
        for algo in agg_df["algorithm"].unique():
            sub = agg_df[agg_df["algorithm"] == algo]
            plt.plot(sub["depth"], sub[metric], label=algo)
        plt.title(f"Mean {metric.capitalize()} vs. Depth (averaged over datasets)")
        plt.xlabel("Depth")
        plt.ylabel(f"Mean {metric}")
        plt.ylim(0, 1)
        plt.legend()
        plt.show()

    def save_results(self, df, filename="depth_sweep_results.csv"):
        df.to_csv(filename, index=False)

    def load_results(self, filename):
        return pd.read_csv(filename)



# import os
# import time
# import pickle
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm
#
# from D_oblique_decision_trees.converters.dispatcher import convert_tree
# from D_oblique_decision_trees.evaluation.evaluator import evaluate_tree
# from Ensembles_of_Oblique_Decision_Trees.Decision_trees.HouseHolder_CART import HHCartClassifier
# from Ensembles_of_Oblique_Decision_Trees.Decision_trees.RandCART import RandCARTClassifier
# from Ensembles_of_Oblique_Decision_Trees.Decision_trees.Oblique_Classifier_1 import ObliqueClassifier1
# from Ensembles_of_Oblique_Decision_Trees.Decision_trees.WODT import WeightedObliqueDecisionTreeClassifier
# from Ensembles_of_Oblique_Decision_Trees.Decision_trees.segmentor import MeanSegmentor
# from Ensembles_of_Oblique_Decision_Trees.Decision_trees.split_criteria import gini
#
# class DepthSweepRunner:
#     """
#     DepthSweepRunner that does NOT stop if no new splits appear; it always
#     runs depth=1..max_depth for each (model, dataset).
#
#     Provides:
#       - run() -> DataFrame
#       - reorder_and_print(df) -> reorders columns
#       - plot_results(df, metric='accuracy') -> direct plot per dataset
#       - plot_aggregated_metric(df, metric='accuracy') -> averages metric across datasets
#       - save_results / load_results
#     """
#
#     def __init__(self, datasets, max_depth=10, model_registry=None):
#         """
#         :param datasets: dict {dataset_name: (X,y)} or list of (X,y)
#         :param max_depth: maximum depth to attempt
#         :param model_registry: optional dict {model_name: lambda -> model}
#                If None, uses an internal default registry
#         """
#         if isinstance(datasets, dict):
#             self.datasets = list(datasets.items())
#         else:
#             self.datasets = [("dataset_" + str(i), ds) for i, ds in enumerate(datasets)]
#
#         self.max_depth = max_depth
#         self.results = []
#
#         if model_registry is None:
#             self.model_registry = self._default_registry()
#         else:
#             self.model_registry = model_registry
#
#     def _default_registry(self):
#         return {
#             "hhcart": lambda: HHCartClassifier(impurity=gini, segmentor=MeanSegmentor()),
#             "randcart": lambda: RandCARTClassifier(impurity=gini, segmentor=MeanSegmentor()),
#             "oc1": lambda: ObliqueClassifier1(min_samples_split=2),
#             "wodt": lambda: WeightedObliqueDecisionTreeClassifier(min_samples_split=2, max_features='all'),
#         }
#
#     def run(self):
#         total_loops = len(self.model_registry) * len(self.datasets) * self.max_depth
#         os.makedirs("saved_trees", exist_ok=True)
#
#         with tqdm(total=total_loops, desc="Depth Sweeping") as pbar:
#             for model_name, constructor in self.model_registry.items():
#                 for dataset_name, (X, y) in self.datasets:
#                     for depth in range(1, self.max_depth + 1):
#                         model = constructor()
#                         model.max_depth = depth
#
#                         start_t = time.time()
#                         model.fit(X, y)
#                         end_t = time.time()
#
#                         tree = convert_tree(model, model_type=model_name)
#                         metrics = evaluate_tree(tree, X, y, training_time=(end_t - start_t))
#                         metrics["algorithm"] = model_name
#                         metrics["dataset"] = dataset_name
#                         metrics["depth"] = depth
#
#                         # Save tree to disk
#                         tree_dir = os.path.join("saved_trees", dataset_name, model_name)
#                         os.makedirs(tree_dir, exist_ok=True)
#                         tree_path = os.path.join(tree_dir, f"depth_{depth}.pkl")
#                         with open(tree_path, "wb") as f:
#                             pickle.dump(tree, f)
#
#                         self.results.append(metrics)
#                         pbar.update(1)
#
#         return pd.DataFrame(self.results)
#
#     def reorder_and_print(self, df):
#         desired_cols = [
#             "dataset", "algorithm", "depth",
#             "accuracy", "coverage", "density", "f_score",
#             "splits", "leaves", "interpretability", "sparsity",
#             "training_time"
#         ]
#         existing = [c for c in desired_cols if c in df.columns]
#         df = df[existing]
#
#         print("\n===== Depth Sweep Results =====\n")
#         print(df)
#         return df
#
#     def plot_results(self, df, metric="accuracy"):
#         if metric not in df.columns:
#             print(f"Metric '{metric}' not in columns: {list(df.columns)}")
#             return
#
#         plt.figure(figsize=(8, 5))
#         sns.lineplot(data=df, x="depth", y=metric, hue="algorithm", style="dataset")
#         plt.title(f"{metric.capitalize()} vs. Depth (per dataset)")
#         plt.ylim(0, 1)
#         plt.show()
#
#     def plot_aggregated_metric(self, df, metric="accuracy"):
#         if metric not in df.columns:
#             print(f"Metric '{metric}' not found in columns: {list(df.columns)}")
#             return
#
#         agg_df = df.groupby(["algorithm", "depth"], as_index=False)[metric].mean()
#
#         plt.figure(figsize=(8, 5))
#         for algo in agg_df["algorithm"].unique():
#             sub = agg_df[agg_df["algorithm"] == algo]
#             plt.plot(sub["depth"], sub[metric], label=algo)
#         plt.title(f"Mean {metric.capitalize()} vs. Depth (averaged over datasets)")
#         plt.xlabel("Depth")
#         plt.ylabel(f"Mean {metric}")
#         plt.ylim(0, 1)
#         plt.legend()
#         plt.show()
#
#     def save_results(self, df, filename="depth_sweep_results.csv"):
#         df.to_csv(filename, index=False)
#
#     def load_results(self, filename):
#         return pd.read_csv(filename)



# class BenchmarkRunner:
#     def __init__(self, trees, datasets):
#         self.trees = trees  # dict of {algorithm_name: DecisionTree}
#         self.datasets = datasets  # list of (X, y) tuples
#
#     def run(self):
#         results = []
#         for name, tree in self.trees.items():
#             for i, (X, y) in enumerate(self.datasets):
#                 metrics = evaluate_tree(tree, X, y)
#                 metrics['algorithm'] = name
#                 metrics['dataset_id'] = i
#                 results.append(metrics)
#         return pd.DataFrame(results)
