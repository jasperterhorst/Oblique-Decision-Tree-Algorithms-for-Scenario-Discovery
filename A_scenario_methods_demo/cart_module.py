"""
CART algorithm wrapper for scenario discovery.
Extracts axis-aligned boxes with coverage and density metrics.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from A_scenario_methods_demo.utils import get_points_in_box
from src.config import DEFAULT_VARIABLE_SEEDS


def get_cart_boxes(tree: DecisionTreeClassifier, x_bounds: tuple, y_bounds: tuple) -> list:
    """
    Recursively extract leaf regions (boxes) from a fitted decision tree.

    Args:
        tree: A fitted DecisionTreeClassifier.
        x_bounds: (min, max) bounds for the x-axis.
        y_bounds: (min, max) bounds for the y-axis.

    Returns:
        A list of box dictionaries with keys 'x', 'y', and 'node'.
    """
    tree_ = tree.tree_
    boxes = []

    def recurse(node: int, bounds: dict) -> None:
        if tree_.children_left[node] == -1 and tree_.children_right[node] == -1:
            box = bounds.copy()
            box["node"] = node
            boxes.append(box)
            return

        feature = tree_.feature[node]
        threshold = tree_.threshold[node]

        if feature == 0:  # x-axis
            left = bounds.copy()
            left["x"] = (bounds["x"][0], min(bounds["x"][1], threshold))
            recurse(tree_.children_left[node], left)

            right = bounds.copy()
            right["x"] = (max(bounds["x"][0], threshold), bounds["x"][1])
            recurse(tree_.children_right[node], right)

        elif feature == 1:  # y-axis
            lower = bounds.copy()
            lower["y"] = (bounds["y"][0], min(bounds["y"][1], threshold))
            recurse(tree_.children_left[node], lower)

            upper = bounds.copy()
            upper["y"] = (max(bounds["y"][0], threshold), bounds["y"][1])
            recurse(tree_.children_right[node], upper)

    recurse(0, {"x": x_bounds, "y": y_bounds})
    return boxes


def run_cart(data: pd.DataFrame, y: np.ndarray, mass_min: float = 0.05):
    """
    Train a CART model and extract decision boxes and performance metrics.

    Args:
        data: DataFrame with 'x' and 'y' columns.
        y: Binary outcome labels (0 = not of interest, 1 = of interest).
        mass_min: Minimum fraction of samples per leaf node.

    Returns:
        Tuple:
            - boxes: List of box dicts (with node and bounds).
            - coverage_list: Per-box coverage values.
            - density_list: Per-box density values.
            - classifications: Predicted class per box.
            - clf: Fitted DecisionTreeClassifier.
    """
    X = data[["x", "y"]].values
    n_samples = len(data)
    min_samples_leaf = max(int(n_samples * mass_min), 1)

    clf = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, random_state=DEFAULT_VARIABLE_SEEDS[0])
    clf.fit(X, y)

    x_bounds = (data["x"].min(), data["x"].max())
    y_bounds = (data["y"].min(), data["y"].max())
    boxes = get_cart_boxes(clf, x_bounds, y_bounds)

    total_positive = np.sum(y == 1)
    coverage_list, density_list, classifications = [], [], []

    for box in boxes:
        node = box["node"]
        pred_class = int(np.argmax(clf.tree_.value[node][0]))
        classifications.append(pred_class)

        idx_box = get_points_in_box(box, data)
        num_in_box = len(idx_box)

        positives = np.sum(y[idx_box] == 1)
        density = positives / num_in_box if num_in_box > 0 else 0
        coverage = positives / total_positive if total_positive > 0 else 0

        density_list.append(density)
        coverage_list.append(coverage)

    return boxes, coverage_list, density_list, classifications, clf
