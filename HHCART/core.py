"""
HHCART Public Interface (core.py)
---------------------------------
Provides a clean API for training and inspecting Householder-reflected
oblique decision trees (HHCART-D) using the full input feature space.

Main features:
- Progressive tree building across depths
- Depth-based selection for evaluation
- Visualisation hooks for trade-offs and splits

Example:
    hh = HHCartD(X_df, y_array, max_depth=6)
    hh.build_tree()
    hh.select(depth=3)
    hh.inspect()
    hh.plot_tradeoff()
"""

import time
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime

from HHCART.HHCartDPruning import HHCartDPruningClassifier
from HHCART.split_criteria import gini
from HHCART.visualisation import bind_plotting_methods
from HHCART.io.save_load import save_full_model


class HHCartD:
    """
    High-level interface for building HHCART-D decision trees with full feature sets.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y,
        *,
        max_depth: int = 6,
        min_samples_split: int = 2,
        min_purity: float = 1,
        tau: float = 0.05,
        random_state: int = None,
        save_dir: Optional[Path] = None,
        debug: bool = False,
    ):
        """
        Initialise the HHCART-D model wrapper.

        Args:
            X (pd.DataFrame): Feature matrix (rows = samples, columns = features).
            y (pd.Series or np.ndarray): Binary target labels.
            max_depth (int): Maximum tree depth to explore.
            min_samples_split (int): Minimum samples to allow further splitting.
            min_purity (float): Minimum class purity to accept a node as pure.
            tau (float): Tolerance for numerical stability in reflection steps.
            random_state (int, optional): Seed for reproducible splits.
            debug (bool, optional): Whether to enable debugging tree build process.
        """
        if not isinstance(X, pd.DataFrame):
            print(f"[⚠] Expected X as pd.DataFrame, got {type(X)}")
        self.X = X
        self.y = y
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_purity = min_purity
        self.tau = tau
        self.random_state = random_state
        self.save_dir: Optional[Path] = None
        self.debug = debug

        self.model_type = "hhcart_d"  # used in folder name

        self.trees_by_depth = {}      # Stores trained trees at each depth
        self.metrics_by_depth = {}    # Stores dict of evaluation metrics per depth
        self.metrics_df = None        # Combined metrics table (used for plotting)
        self.selected_depth = None    # User-specified depth for inspection

        # Attach plotting tools dynamically (e.g., .plot_tradeoff)
        bind_plotting_methods(self)

    def build_tree(self, model_title: Optional[str] = None) -> None:
        """
        Train HHCART-D trees and save the result under data/model_title/.

        Args:
            model_title (str, optional): Folder name to save the model. If None, auto-generated.
        """
        # Clear any previously stored trees or metrics
        self.trees_by_depth.clear()
        self.metrics_by_depth.clear()
        self.metrics_df = None
        self.selected_depth = None

        # Set up the model and build the tree to max depth
        model = HHCartDPruningClassifier(
            impurity=gini,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_purity=self.min_purity,
            tau=self.tau,
            debug=self.debug,
        )
        start_time = time.perf_counter()
        model.fit(self.X, self.y)
        end_time = time.perf_counter()
        train_duration = end_time - start_time

        # Store each depth's tree and metrics
        rows = []
        for depth, metrics in model.metrics_by_depth.items():
            self.trees_by_depth[depth] = model.trees_by_depth[depth]

            m = metrics.copy()
            m.update({
                "depth": depth,
                "n_features": self.X.shape[1],
                "n_samples": self.X.shape[0],
                "runtime": train_duration
            })
            self.metrics_by_depth[depth] = m
            rows.append(m)

        self.metrics_df = pd.DataFrame(rows)

        # === Save under /data/model_title ===
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        final_title = model_title or f"{self.model_type}_depth_{self.max_depth}_{timestamp}"
        self.save_dir = save_full_model(self, name=final_title)

        print(f"[✓] Tree of depth {max(self.trees_by_depth.keys())} built and saved to: {self.save_dir}")

    def available_depths(self) -> list[int]:
        """
        Return a list of all available tree depths built in this model.

        Returns:
            list[int]: Sorted list of trained depths.
        """
        return sorted(self.trees_by_depth.keys())

    def get_tree_by_depth(self, depth: int):
        """
        Return the decision tree pruned to a specific depth.

        Args:
            depth (int): Tree depth to retrieve.

        Returns:
            DecisionTree: Pruned decision tree.

        Raises:
            ValueError: If no tree exists at the requested depth.
        """
        if depth not in self.trees_by_depth:
            raise ValueError(f"[❌] No tree found at depth={depth}.")
        return self.trees_by_depth[depth]

    @property
    def coverage_by_depth(self) -> dict[int, float]:
        return dict(zip(self.metrics_df["depth"], self.metrics_df["coverage"]))

    @property
    def density_by_depth(self) -> dict[int, float]:
        return dict(zip(self.metrics_df["depth"], self.metrics_df["density"]))

    def select(self, depth: int) -> None:
        """
        Set a specific tree depth for downstream inspection or plotting.

        Args:
            depth (int): Tree depth to activate.

        Raises:
            ValueError: If no tree exists at the requested depth.
        """
        if depth not in self.trees_by_depth:
            raise ValueError(f"[❌] No tree exists at depth={depth}. Did you call .build_tree()?")

        self.selected_depth = depth
        print(f"[✓] Selected tree at depth {depth}.")

    def get_selected_tree(self):
        """
        Return the currently selected tree object.

        Returns:
            DecisionTree: A trained decision tree at selected depth.

        Raises:
            ValueError: If no depth was selected or if the tree is missing.
        """
        if self.selected_depth is None:
            raise ValueError("[❌] No depth selected. Use .select(depth) first.")
        if self.selected_depth not in self.trees_by_depth:
            raise ValueError(f"[❌] No tree found at depth={self.selected_depth}.")
        return self.trees_by_depth[self.selected_depth]

    def print_tree(self, depth: int) -> None:
        """
        Print the structure of the decision tree at a given depth.

        Args:
            depth (int): Tree depth to print.

        Raises:
            ValueError: If no such tree has been trained.
        """
        if depth not in self.trees_by_depth:
            raise ValueError(f"[❌] No tree found at depth={depth}.")
        print(f"\n[🌳] Tree structure at depth {depth}:\n")
        self.trees_by_depth[depth].print_structure()

    def inspect(self) -> None:
        """
        Print the structure of the currently selected tree.

        Raises:
            ValueError: If no tree is currently selected.
        """
        print(f"\n[🔍] Inspecting tree at selected depth {self.selected_depth}...\n")
        self.get_selected_tree().print_structure()
