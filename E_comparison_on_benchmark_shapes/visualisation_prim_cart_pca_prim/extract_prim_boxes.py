"""
Extract boxes from a trained PRIM object and convert box limits to corner coordinates.

This module provides a function to extract explicitly selected PRIM boxes,
compute relevant statistics, and convert the hyperrectangular box limits into
corner coordinates suitable for visualization or further analysis.

The output DataFrame includes:
- box_id: Index of the box in the PRIM peeling trajectory
- coverage, density, mass: performance metrics
- coords: list of tuples representing polygon corner coordinates in original feature space
"""

import pandas as pd
from typing import List, Tuple


def box_limits_to_corners(box_lim: pd.DataFrame) -> List[Tuple[float, ...]]:
    """
    Convert box limits (min/max per feature) into corner coordinates.

    Parameters
    ----------
    box_lim : pd.DataFrame
        DataFrame with two rows (min and max), columns are feature names

    Returns
    -------
    corners : list of tuples
        List of coordinate tuples representing each corner of the box.
        For 2D, returns 4 corners; for 3D, 8 corners, etc.
    """
    feature_names = box_lim.columns.tolist()
    mins = box_lim.iloc[0].values
    maxs = box_lim.iloc[1].values

    n_dims = len(feature_names)
    corners = []

    # Generate all corner combinations by binary counting (0=min, 1=max)
    for i in range(2**n_dims):
        corner = []
        for dim in range(n_dims):
            # Check bit dim of i
            if (i >> dim) & 1:
                corner.append(maxs[dim])
            else:
                corner.append(mins[dim])
        corners.append(tuple(corner))

    return corners


def extract_prim_boxes(prim_obj, selected_indices: List[int]) -> pd.DataFrame:
    """
    Extract selected boxes from a trained PRIM object, returning coverage,
    density, mass, and polygon corner coordinates.

    Parameters
    ----------
    prim_obj : Prim
        A fitted PRIM object with peeling_trajectory and box_lims attributes.
    selected_indices : List[int]
        List of box indices in the peeling trajectory to extract.

    Returns
    -------
    boxes_df : pd.DataFrame
        DataFrame with columns ['box_id', 'coverage', 'density', 'mass', 'coords'].
        'coords' contains a list of tuples with corner coordinates for each box.
    """
    records = []

    for idx in selected_indices:
        stats = prim_obj.peeling_trajectory.loc[idx]
        box_lim = prim_obj.box_lims[idx]

        # Convert box limits to corner coordinates
        coords = box_limits_to_corners(box_lim)

        record = {
            "box_id": idx,
            "coverage": stats["coverage"],
            "density": stats["density"],
            "mass": stats["mass"],
            "coords": coords,
        }
        records.append(record)

    boxes_df = pd.DataFrame.from_records(records)
    return boxes_df
