"""
HHCART_SD Public Interface (core.py)
---------------------------------
Provides a clean API for training and inspecting Householder-reflected
oblique decision trees (HHCART_SD-D) using the full input feature space.

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

from HHCART_SD.HHCartDPruning import HHCartDPruningClassifier
from HHCART_SD.split_criteria import gini
from HHCART_SD.visualisation import bind_plotting_methods
from HHCART_SD.io.save_load import save_full_model


class HHCartD:
    """
    HHCART_SD-D Interface: Oblique Decision Trees for Scenario Discovery
    ===============================================================

    `HHCartD` is the high-level interface for building and analyzing
    oblique decision trees using Householder reflections. This class
    manages the training, evaluation, selection, and visualization of
    scenario discovery trees over varying depths.

    Key Features:
    -------------
    - Builds oblique decision trees that split the input space using linear hyperplanes.
    - Supports training trees to different depths and selecting the optimal depth.
    - Includes built-in metrics: accuracy, coverage, density, and purity.
    - Provides rich visualization tools to interpret trees and model behavior.

    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix (samples × features).
    y : np.ndarray or pd.Series
        Binary labels (0 or 1) for each sample.
    max_depth : int, optional (default=6)
        Maximum tree depth to train.
    min_samples_split : int, optional (default=2)
        Minimum samples required to consider a split.
    min_purity : float, optional (default=1.0)
        Purity threshold to stop splitting.
    tau : float, optional (default=0.05)
        Tolerance for numerical stability in split calculation.
    random_state : int, optional
        Seed for reproducible results.
    save_dir : pathlib.Path, optional
        Directory to save trained models.
    debug : bool, optional
        Enables verbose debugging during tree construction.

    Visualization Methods (auto-attached):
    --------------------------------------
    These methods are dynamically attached at runtime via `bind_plotting_methods()`.
    Your IDE may not recognize them statically, but they are available on all HHCartD instances.

    - plot_tradeoff_path(): Coverage vs. Density tradeoff path.
    - plot_clipped_boundaries(): Visualize oblique splits.
    - plot_tree_structure(depth=3): Tree structure at a specific depth.
    - plot_metrics_over_depth(): Accuracy, coverage, and density over depths.
    - plot_node_size_distribution(): Node sample sizes per depth.
    - plot_oblique_regions(): Valid polygonal decision regions.

    Example:
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> X = pd.DataFrame(np.random.rand(100, 5))
    >>> y = np.random.randint(0, 2, size=100)
    >>> hh = HHCartD(X, y, max_depth=6)
    >>> hh.build_tree()
    >>> hh.select(depth=3)
    >>> hh.inspect()
    >>> hh.plot_tree_structure(depth=3)
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

    def build_tree(self, model_title: Optional[str] = None, save: bool = True) -> None:
        """
        Train HHCART_SD-D trees and save the result under data/model_title/.

        Args:
            model_title (str, optional): Folder name to save the model. If None, auto-generated.
            save (bool): Whether to save the model to disk. Defaults to True.
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

        # === Optionally save under /data/model_title ===
        if save:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            final_title = model_title or f"{self.model_type}_depth_{self.max_depth}_{timestamp}"
            self.save_dir = save_full_model(self, name=final_title)
            print(f"[✓] Tree of depth {max(self.trees_by_depth.keys())} built and saved to: {self.save_dir}")
        else:
            self.save_dir = None
            print(f"[✓] Tree of depth {max(self.trees_by_depth.keys())} built (not saved).")

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
