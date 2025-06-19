import numpy as np
import pandas as pd
from typing import List
from .box_utils import compute_corners_from_box_limits


def extract_pca_prim_boxes(prim_obj, selected_indices: List[int],
                           rotation_matrix: np.ndarray,
                           X_mean: np.ndarray,
                           X_std: np.ndarray,
                           rotation_feature_names: List[str]) -> pd.DataFrame:
    """
    Extract selected boxes from a trained PCA-PRIM object, returning coverage,
    density, mass, and polygon corner coordinates in original feature space.

    Parameters
    ----------
    prim_obj : Prim
    selected_indices : List[int]
    rotation_matrix : np.ndarray
    X_mean : np.ndarray
    X_std : np.ndarray
    rotation_feature_names : List[str]

    Returns
    -------
    boxes_df : pd.DataFrame
    """
    records = []

    for idx in selected_indices:

        stats = prim_obj.peeling_trajectory.loc[idx]
        box_lim = prim_obj.box_lims[idx]

        corners_rotated = compute_corners_from_box_limits(box_lim, features_to_include=rotation_feature_names)

        # Correct inverse transform: first rotate back, then de-standardise
        X_std_recovered = corners_rotated @ rotation_matrix.T

        corners_original_df = (corners_rotated @ rotation_matrix.T * X_std) + X_mean
        corners_original_np = corners_original_df.to_numpy()

        coords = [tuple(row) for row in corners_original_np]

        for i, coord in enumerate(coords):
            for j, val in enumerate(coord):
                if not np.issubdtype(type(val), np.number):
                    print(f"\n[ERROR] Non-numeric value detected in coord[{i}][{j}]: {val} (type: {type(val)})")
                    print(f"Full coord: {coord}")
                    raise ValueError(
                        f"Non-numeric coordinate detected: {coord}. This box likely contains a categorical restriction.")

        record = {
            "box_id": idx,
            "coverage": stats["coverage"],
            "density": stats["density"],
            "mass": stats["mass"],
            "coords": coords,
            "box_lim": box_lim,
        }
        records.append(record)

    boxes_df = pd.DataFrame.from_records(records)
    return boxes_df
