"""
Local PCA preprocess for PCA-PRIM, returns mean and std for correct inverse transform.
"""

import numpy as np
import pandas as pd
from ema_workbench.analysis.prim_util import rotate_subset


def pca_preprocess(experiments, y, subsets=None, exclude=set()):
    """
    Perform PCA to preprocess experiments before running PRIM.

    This version also returns X_mean and X_std for proper inverse rotation.

    Parameters
    ----------
    experiments : DataFrame
    y : ndarray
    subsets : dict, optional
    exclude : set, optional

    Returns
    -------
    rotated_experiments : DataFrame
    rotation_matrix : DataFrame
    X_mean : ndarray
    X_std : ndarray
    """
    # experiments to rotate
    x = experiments.drop(exclude, axis=1)

    if not x.select_dtypes(exclude=np.number).empty:
        raise RuntimeError("X includes non numeric columns")
    if not set(np.unique(y)) == {0, 1}:
        raise RuntimeError(
            f"y should only contain 0s and 1s, currently y contains {set(np.unique(y))}."
        )

    if not subsets:
        subsets = {"r": x.columns.values.tolist()}

    new_columns = []
    new_dtypes = []
    for key, value in subsets.items():
        subset_cols = [f"{key}_{i}" for i in range(len(value))]
        new_columns.extend(subset_cols)
        new_dtypes.extend((float,) * len(value))

    rotated_experiments = pd.DataFrame(index=experiments.index.values)
    for name, dtype in zip(new_columns, new_dtypes):
        rotated_experiments[name] = pd.Series(dtype=dtype)

    for entry in exclude:
        rotated_experiments[entry] = experiments[entry]

    rotation_matrix = np.zeros((x.shape[1],) * 2)
    column_names = []
    row_names = []

    j = 0
    for key, value in subsets.items():
        x_subset = x[value]

        # Compute mean and std for proper inverse
        X_mean = x_subset.mean(axis=0).values
        X_std = x_subset.std(axis=0).values
        X_std[X_std == 0] = 1

        # Standardise
        x_subset_std = (x_subset - X_mean) / X_std

        # Rotate
        subset_rotmat, subset_experiments = rotate_subset(x_subset_std, y)

        rotation_matrix[j : j + len(value), j : j + len(value)] = subset_rotmat
        row_names.extend(value)
        j += len(value)

        for i in range(len(value)):
            name = f"{key}_{i}"
            rotated_experiments[name] = subset_experiments[:, i]
            column_names.append(name)

    rotation_matrix = pd.DataFrame(rotation_matrix, index=row_names, columns=column_names)

    # Now compute global X_mean and X_std for all features
    X_mean_full = x.mean(axis=0).values
    X_std_full = x.std(axis=0).values
    X_std_full[X_std_full == 0] = 1

    return rotated_experiments, rotation_matrix, X_mean_full, X_std_full
