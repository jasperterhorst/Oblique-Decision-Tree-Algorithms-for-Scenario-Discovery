"""
Manual implementation of the Patient Rule Induction Method (PRIM) for scenario discovery.
Includes peeling logic and performance metric calculation (coverage and density).
"""

import numpy as np
import pandas as pd
from typing import Tuple
from A_scenario_methods_demo.utils import get_points_in_box


def manual_prim(data: pd.DataFrame, target: np.ndarray, peel_alpha: float = 0.05, mass_min: float = 0.05):
    """
    Perform PRIM on 2D _data with binary targets (0 or 1).

    Args:
        data: DataFrame with columns 'x' and 'y'.
        target: Array of binary target labels.
        peel_alpha: Fraction of samples to peel at each step.
        mass_min: Minimum fraction of samples required in a box.

    Returns:
        boxes_history: List of boxes generated during peeling.
        final_box: The last (best) box discovered.
    """
    total_samples = len(data)
    min_points = max(1, int(total_samples * mass_min))

    box = {
        "x": (data["x"].min(), data["x"].max()),
        "y": (data["y"].min(), data["y"].max())
    }

    boxes_history = [box.copy()]
    current_idx = data.index.values

    def mean_target(idx: np.ndarray) -> float:
        return target[idx].mean() if len(idx) > 0 else 0.0

    current_mean = mean_target(current_idx)
    improvement = True

    while improvement and len(current_idx) > min_points:
        best_improvement = 0.0
        best_box = None
        best_idx = None

        for dim in ["x", "y"]:
            for side in ["min", "max"]:
                values = data.loc[current_idx, dim].values
                sorted_vals = np.sort(values)
                n_peel = max(int(peel_alpha * len(values)), 1)

                new_box = box.copy()
                if side == "min":
                    new_box[dim] = (sorted_vals[n_peel], box[dim][1])
                else:
                    new_box[dim] = (box[dim][0], sorted_vals[-n_peel - 1])

                idx_new = get_points_in_box(new_box, data)

                if len(idx_new) < min_points:
                    continue

                new_mean = mean_target(idx_new)
                improvement_val = new_mean - current_mean

                if improvement_val > best_improvement:
                    best_improvement = improvement_val
                    best_box = new_box.copy()
                    best_idx = idx_new

        if best_box is not None and best_improvement > 0:
            box = best_box.copy()
            current_idx = best_idx
            current_mean = mean_target(current_idx)
            boxes_history.append(box.copy())
        else:
            improvement = False

    return boxes_history, box


def compute_box_metrics(box: dict, data: pd.DataFrame, target: np.ndarray) -> Tuple[float, float]:
    """
    Compute coverage and density for a given box.

    Args:
        box: Dictionary with 'x' and 'y' bounds.
        data: DataFrame with columns 'x' and 'y'.
        target: Binary array of labels.

    Returns:
        coverage: Fraction of all positives inside the box.
        density: Fraction of points in the box that are positive.
    """
    idx_box = get_points_in_box(box, data)

    positives_in_box = np.sum(target[idx_box] == 1)
    num_in_box = len(idx_box)
    total_positives = np.sum(target == 1)

    density = positives_in_box / num_in_box if num_in_box > 0 else 0.0
    coverage = positives_in_box / total_positives if total_positives > 0 else 0.0

    return coverage, density
