"""
HHCART Feature Selection Methods
--------------------------------
Defines feature scoring functions for selecting top-K features.
Each function must follow the signature:
    def selector(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        returns a score per feature (higher is better)
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, f_classif
from scipy.stats import pointbiserialr


def mutual_info_score(X: pd.DataFrame, y: np.ndarray):
    return mutual_info_classif(X, y, discrete_features='auto')


def anova_score(X: pd.DataFrame, y: np.ndarray):
    scores, _ = f_classif(X, y)
    return scores


def point_biserial_score(X: pd.DataFrame, y: np.ndarray):
    return np.array([abs(pointbiserialr(X[col], y).correlation) for col in X.columns])


FEATURE_SELECTOR_REGISTRY = {
    "mutual_info": mutual_info_score,
    "anova": anova_score,
    "point_biserial": point_biserial_score
}
