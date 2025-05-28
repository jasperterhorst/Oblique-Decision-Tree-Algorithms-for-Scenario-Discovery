"""
HHCART Public Interface (core.py)
---------------------------------
Defines the high-level `HHCartD` class, providing a user-friendly API for training
Householder-reflected oblique decision trees over top-K feature subsets.

Example usage:

    from HHCART import HHCartD
    hh = HHCartD(X_df, y_array, max_depth=6, feature_selector="mutual_info")
    hh.build_tree(k_range=[1, 2, 3])
    hh.plot_tradeoff()
    hh.select(depth=3, k=2)
    hh.inspect()

Supports integration with custom feature selectors, metric logging, and dynamic plotting.
"""

import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from HHCART.HHCartDLayerwise import HHCartDLayerwiseClassifier
from HHCART.split_criteria import gini
from HHCART.feature_selectors import FEATURE_SELECTOR_REGISTRY
from HHCART.visualisation import bind_plotting_methods


class HHCartD:
    def __init__(self, X, y, *, max_depth=6, min_samples_split=10, min_purity=0.95,
                 tau=1e-4, random_state=None, feature_selector=None):
        self.X = X
        self.y = y
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_purity = min_purity
        self.tau = tau
        self.random_state = random_state

        if isinstance(feature_selector, str):
            if feature_selector in FEATURE_SELECTOR_REGISTRY:
                self.feature_selector = FEATURE_SELECTOR_REGISTRY[feature_selector]
            else:
                raise ValueError(f"Unknown feature selector name: {feature_selector}")
        else:
            self.feature_selector = feature_selector

        self.trees_by_depth = {}      # {(k, depth): tree}
        self.metrics_by_depth = {}    # {(k, depth): metrics dict}
        self.metrics_df = None
        self.selected_depth = None
        self.selected_k = None

        bind_plotting_methods(self)

    def build_tree(self, *, k_range=None):
        if self.feature_selector and k_range:
            print("🔍 Computing feature scores using selector...")
            scores = np.asarray(self.feature_selector(self.X, self.y))
            sorted_idx = np.argsort(scores)[::-1]
            rows = []

            for k in k_range:
                print(f"\n🌲 Building tree with top-{k} features:")
                selected_features = self.X.columns[sorted_idx[:k]]
                print("   ➤ Selected features:", list(selected_features))

                X_k = self.X[selected_features]
                model = HHCartDLayerwiseClassifier(
                    impurity=gini,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_purity=self.min_purity,
                    tau=self.tau,
                    random_state=self.random_state,
                )
                model.fit(X_k, self.y)

                for d, metrics in model.metrics_by_depth.items():
                    if d == 0:
                        continue  # Skip trivial depth
                    self.trees_by_depth[(k, d)] = model.trees_by_depth[d]
                    m = metrics.copy()
                    m.update({"k": k, "depth": d, "n_features": k, "n_samples": self.X.shape[0]})
                    self.metrics_by_depth[(k, d)] = m
                    rows.append(m)

                print(f"✅ Done: Built tree up to depth {max(model.metrics_by_depth.keys())}")
                final_metrics = model.metrics_by_depth[max(model.metrics_by_depth.keys())]
                print(f"   Accuracy: {final_metrics['accuracy']:.3f}, "
                      f"Coverage: {final_metrics['coverage']:.3f}, "
                      f"Density: {final_metrics['density']:.3f}")

            self.metrics_df = pd.DataFrame(rows)

        else:
            print("⚠️ No feature selection specified — using all features.")
            model = HHCartDLayerwiseClassifier(
                impurity=gini,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_purity=self.min_purity,
                tau=self.tau,
                random_state=self.random_state,
            )

            model.fit(self.X, self.y)

            # with tqdm(total=self.max_depth, desc="Training (all features)", unit="depth") as pbar:
            #     model.fit(self.X, self.y)
            #     for _ in model.metrics_by_depth:
            #         pbar.update(1)

            rows = []
            for d, metrics in model.metrics_by_depth.items():
                if d == 0:
                    continue
                self.trees_by_depth[(None, d)] = model.trees_by_depth[d]
                m = metrics.copy()
                m.update({"k": None, "depth": d, "n_features": self.X.shape[1], "n_samples": self.X.shape[0]})
                self.metrics_by_depth[(None, d)] = m
                rows.append(m)

            self.metrics_df = pd.DataFrame(rows)
            print(f"✅ Tree built to depth {max(model.metrics_by_depth.keys())}")

    def select(self, depth, k=None):
        self.selected_depth = depth
        self.selected_k = k

    def get_selected_tree(self):
        key = (self.selected_k, self.selected_depth)
        if key not in self.trees_by_depth:
            raise ValueError(f"Tree for k={self.selected_k}, depth={self.selected_depth} not found.")
        return self.trees_by_depth[key]

    def feature_selector_top_k(self, k):
        if not self.feature_selector:
            raise ValueError("Feature selector is not defined.")
        scores = np.asarray(self.feature_selector(self.X, self.y))
        top_k = np.argsort(scores)[::-1][:k]
        return self.X.columns[top_k].tolist()

    def print_tree(self, depth, k=None):
        key = (k, depth)
        if key not in self.trees_by_depth:
            raise ValueError(f"No tree found for k={k}, depth={depth}")
        print(f"Tree structure for k={k}, depth={depth}:")
        self.trees_by_depth[key].print_structure()

    def inspect(self):
        self.get_selected_tree().print_structure()

    def save(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        metrics_path = os.path.join(folder_path, "metrics.csv")
        trees_path = os.path.join(folder_path, "trees.pkl")

        self.metrics_df.to_csv(metrics_path, index=False)
        with open(trees_path, "wb") as f:
            pickle.dump(self.trees_by_depth, f)

        print(f"✅ Saved metrics to {metrics_path}")
        print(f"✅ Saved trees to {trees_path}")
