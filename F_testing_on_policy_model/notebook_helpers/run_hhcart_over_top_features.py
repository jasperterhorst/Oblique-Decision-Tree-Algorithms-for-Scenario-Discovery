"""
Run HHCartD Over Top Features (run_hhcart_top_features.py)
----------------------------------------------------------
Provides a utility function to run HHCartD on varying numbers of top-ranked features,
automatically save the trained models, and generate all standard plots for analysis.

Features:
- Select top-N features based on feature_scores.
- Build HHCartD trees for each N and save with traceable model title.
- Generate and save: tree structure plots, metrics plots, tradeoff plots, node size plots.
- Uses short feature name representation to keep file names readable.

Usage Example:
--------------
run_hhcart_over_top_features(X, y, feature_scores, top_n_range=(2, 9), ...)
"""

import pandas as pd
import numpy as np
from typing import Union, Tuple

from HHCART_SD import HHCartD


def run_hhcart_over_top_features(
    X_full: pd.DataFrame,
    y: Union[np.ndarray, pd.Series],
    feature_scores: pd.DataFrame,
    top_n_range: Union[list, Tuple[int, int]],
    *,
    max_depth: int,
    mass_min: Union[int, float],
    min_purity: float,
    appendix: str = "",
    debug: bool = False
) -> None:
    """
    Run HHCartD on varying numbers of top-ranked features and save models and plots.

    For each N in top_n_range:
    - Builds a HHCartD tree using the top N features.
    - Saves the trained model with a clear model title.
    - Generates and saves standard plots (tree structure, metrics, tradeoff, node size).
    - Prints progress and result summary to console.

    Args:
        X_full (pd.DataFrame):
            Full input feature matrix.
        y (np.ndarray or pd.Series):
            Target labels (binary).
        feature_scores (pd.DataFrame):
            Output from get_ex_feature_scores(); features must be sorted by importance.
        top_n_range (list[int] or tuple[int, int]):
            List of N values to test, or (min, max) range inclusive.
        max_depth (int):
            Maximum tree depth for HHCartD.
        mass_min (int or float):
            mass_min parameter for HHCartD.
        min_purity (float):
            min_purity parameter for HHCartD.
        appendix (str, optional):
            Optional appendix to append to model titles. Default is "".
        debug (bool, optional):
            Whether to enable debug prints. Default is False.

    Returns:
        None

    Side effects:
        - Saves trained models and plots to disk.
        - Prints progress and result summary to console.
    """
    # Normalise top_n_range → convert tuple to list if needed
    if isinstance(top_n_range, tuple):
        if len(top_n_range) != 2:
            raise ValueError("top_n_range as tuple must be (min_N, max_N)")
        min_N, max_N = top_n_range
        top_n_list = list(range(min_N, max_N + 1))
    else:
        top_n_list = list(top_n_range)

    # Extract ordered list of top features
    top_features_ordered = feature_scores.index.tolist()

    # Run HHCartD for each N in top_n_list
    for n_feats in top_n_list:
        # Select top N features
        selected_features = top_features_ordered[:n_feats]

        # Subset X
        X_selected = X_full[selected_features].copy()

        # Prepare short feature string (first 3 letters of each feature)
        feature_name_str = "_".join([feat[:3] for feat in selected_features])

        # Prepare model title
        model_title = (
            f"hhcart_top{n_feats}f_"
            f"dep_{max_depth}_mass_{str(mass_min).replace('.', '_')}"
            f"_pur_{str(min_purity).replace('.', '_')}_"
            f"{feature_name_str}"
            f"{appendix}"
        )

        # Log start of run
        print(f"\n[RUN] Building HHCartD with top {n_feats} features: {selected_features}")
        print(f"[INFO] Model title: {model_title}")

        # Instantiate HHCartD
        hh = HHCartD(
            X_selected,
            y,
            min_purity=min_purity,
            mass_min=mass_min,
            max_depth=max_depth,
            debug=debug
        )

        # Build and save model
        hh.build_tree(model_title)

        # Generate and save standard plots
        # -- Tree structure plots for each depth
        for depth in hh.available_depths():
            hh.select(depth=depth)
            hh.plot_tree_structure(depth=depth, save=True)

        # -- Metrics plots
        hh.plot_metrics_vs_structure(save=True)
        hh.plot_metrics_vs_structure(save=True, x_axis="class1_leaf_count")

        # -- Tradeoff plots
        hh.plot_tradeoff_path(save=True)
        hh.plot_tradeoff_path(save=True, color_by="class1_leaf_count")

        # -- Node size plot
        hh.plot_node_size_distribution(save=True)

        # Log end of run
        print(f"\n[DONE] Finished run with top {n_feats} features.")
        print(f"[RESULT] Depths built: {hh.available_depths()}")
