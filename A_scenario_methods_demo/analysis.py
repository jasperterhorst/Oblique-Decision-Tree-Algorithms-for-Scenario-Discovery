"""
Module for running the analysis pipeline.
"""

import numpy as np
import pandas as pd
from A_scenario_methods_demo.prim_module import manual_prim, compute_box_metrics
from A_scenario_methods_demo.pca_rotation_module import apply_pca, rotate_data
from A_scenario_methods_demo.cart_module import run_cart


def run_analysis(
    num_dots: int,
    quad_coords: list,
    frac_inside: float,
    frac_outside: float,
    peel_frac: float,
    prim_mass_min: float,
    cart_mass_min: float,
) -> dict:
    """
    Generate _data, label points based on a quadrilateral region, and run PRIM, PCA-PRIM,
    and CART analyses.

    Parameters:
        num_dots (int): Number of random samples.
        quad_coords (list): List of 8 floats defining the 4 vertices of the quadrilateral.
        frac_inside (float): Probability to label a point as 1 inside the quadrilateral.
        frac_outside (float): Probability to label a point as 1 outside the quadrilateral.
        peel_frac (float): Fraction for peeling in the PRIM algorithm.
        prim_mass_min (float): Minimum fraction of total samples required in a box.
        cart_mass_min (float): Minimum fraction required for the CART algorithm.

    Returns:
        dict: Dictionary containing samples, labels, and analysis results.
    """
    quadrilateral = [
        (quad_coords[i], quad_coords[i + 1])
        for i in range(0, len(quad_coords), 2)
    ]
    np.random.seed(12)
    samples = np.random.rand(num_dots, 2)

    def in_quadrilateral(point, a, b, c, d):
        def area(x, y, z):
            return abs(
                (x[0] * (y[1] - z[1]) +
                 y[0] * (z[1] - x[1]) +
                 z[0] * (x[1] - y[1])) / 2.0
            )

        total_area = area(a, b, c) + area(a, c, d)
        pt_area = (
                area(point, a, b) +
                area(point, b, c) +
                area(point, c, d) +
                area(point, d, a)
        )
        return abs(pt_area - total_area) < 1e-6

    y_labels = np.empty(num_dots, dtype=int)
    for i, sample_pt in enumerate(samples):
        if in_quadrilateral(sample_pt, *quadrilateral):
            y_labels[i] = np.random.choice([0, 1], p=[1 - frac_inside, frac_inside])
        else:
            y_labels[i] = np.random.choice([0, 1], p=[1 - frac_outside, frac_outside])

    df_samples = pd.DataFrame(samples, columns=["x", "y"])

    # PRIM Analysis on original _data
    boxes_history, _ = manual_prim(
        df_samples, y_labels, peel_alpha=peel_frac, mass_min=prim_mass_min
    )
    prim_orig_cov = []
    prim_orig_dens = []
    for box in boxes_history:
        cov, dens = compute_box_metrics(box, df_samples, y_labels)
        prim_orig_cov.append(cov)
        prim_orig_dens.append(dens)

    # PCA-PRIM Analysis
    mu, V = apply_pca(samples, y_labels, vulnerable_value=1)
    X_rot = rotate_data(samples, mu, V)
    df_rot = pd.DataFrame(X_rot, columns=["x", "y"])
    quadrilateral_rot = [
        np.dot((np.array(vertex) - mu), V) for vertex in quadrilateral
    ]
    boxes_history_rot, _ = manual_prim(
        df_rot, y_labels, peel_alpha=peel_frac, mass_min=prim_mass_min
    )
    pcaprim_cov = []
    pcaprim_dens = []
    for box in boxes_history_rot:
        cov, dens = compute_box_metrics(box, df_rot, y_labels)
        pcaprim_cov.append(cov)
        pcaprim_dens.append(dens)

    # CART Analysis
    cart_boxes, cart_cov, cart_dens, cart_class, _ = run_cart(
        df_samples, y_labels, mass_min=cart_mass_min
    )

    return {
        "samples": samples,
        "y_labels": y_labels,
        "quadrilateral": quadrilateral,
        "boxes_history": boxes_history,
        "prim_orig_cov": prim_orig_cov,
        "prim_orig_dens": prim_orig_dens,
        "X_rot": X_rot,
        "quadrilateral_rot": quadrilateral_rot,
        "boxes_history_rot": boxes_history_rot,
        "pcaprim_cov": pcaprim_cov,
        "pcaprim_dens": pcaprim_dens,
        "mu": mu,
        "V": V,
        "cart_boxes": cart_boxes,
        "cart_cov": cart_cov,
        "cart_dens": cart_dens,
        "cart_class": cart_class,
    }
