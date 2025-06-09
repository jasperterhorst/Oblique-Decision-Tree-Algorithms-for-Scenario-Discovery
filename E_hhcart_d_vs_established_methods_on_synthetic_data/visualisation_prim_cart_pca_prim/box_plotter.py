"""
Box Plotter utilities for visualising scenario discovery boxes.

Provides a unified `plot_boxes()` function to visualise:
- PRIM boxes (selected by user)
- PCA-PRIM boxes (selected by user, rotated back)
- CART boxes predicting class 1

Supports passing:
- single box object (Prim, PrimBox, CART)
- tuple of PrimBox objects → plotted together in one plot

All plotting is polygon-based, using the corner coordinates extracted.
"""

import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.config.colors_and_plot_styles import (PRIMARY_LIGHT, PRIMARY_DARK,
                                               SECONDARY_LIGHT, SECONDARY_MIDDLE, SECONDARY_DARK)
from src.config.plot_settings import apply_global_plot_settings, beautify_plot

from .extract_cart_boxes import extract_cart_boxes
from .extract_prim_boxes import extract_prim_boxes
from .extract_pca_prim_boxes import extract_pca_prim_boxes
from .box_utils import order_polygon_corners_2d


def plot_boxes(obj=None, boxes=None, X=None, y=None, selected_indices=None,
               rotation_matrix=None, X_mean=None, X_std=None,
               rotation_feature_names=None,
               labels=None, title=None, save_path=None):
    """
    Unified box plotter for PRIM, PCA-PRIM and CART.
    """
    # --- Sanity checks ---
    if X is None:
        raise ValueError("You must provide X (for plotting background points).")

    feature_names = list(X.columns)
    if len(feature_names) != 2:
        raise ValueError("plot_boxes() currently supports only 2D plotting (len(X.columns) == 2).")

    feature_names = list(X.columns)
    var1 = feature_names[0]
    var2 = feature_names[1]

    if boxes is not None:
        if not isinstance(boxes, tuple):
            raise ValueError("boxes must be a tuple of PrimBox objects.")
        if not all(any(name in str(type(box)) for name in ["PrimBox"]) for box in boxes):
            raise ValueError("All elements in boxes must be PrimBox objects.")
        if selected_indices is None or len(selected_indices) != len(boxes):
            raise ValueError("You must provide selected_indices matching the number of boxes.")

        multi_boxes_df = []
        source_labels = []
        for i, (box_obj, box_idx) in enumerate(zip(boxes, selected_indices)):
            if rotation_matrix is None:
                box_df = extract_prim_boxes(box_obj, [box_idx])
                source_algorithm = "PRIM"
            else:
                inferred_rotation_feature_names = rotation_feature_names
                if inferred_rotation_feature_names is None:
                    inferred_rotation_feature_names = list(box_obj.box_lims[0].columns)

                box_df = extract_pca_prim_boxes(
                    box_obj, [box_idx], rotation_matrix, X_mean, X_std, inferred_rotation_feature_names
                )
                source_algorithm = "PCA-PRIM"

            label = labels[i] if labels is not None else f"Box {i+1}"
            box_df["source"] = label
            box_df["source_algorithm"] = source_algorithm
            box_df["box_display_id"] = i + 1
            multi_boxes_df.append(box_df)
            source_labels.append(label)

        boxes_df = pd.concat(multi_boxes_df, ignore_index=True)

    else:
        if obj is None:
            raise ValueError("You must provide either obj or boxes.")

        source_labels = []

        if obj.__class__.__name__ == "CART":
            boxes_df = extract_cart_boxes(obj)
            boxes_df["source"] = "CART"
            boxes_df["source_algorithm"] = "CART"
            boxes_df["box_display_id"] = boxes_df["box_id"] + 1  # add this line
            source_labels = ["CART"]

        elif any(name in str(type(obj)) for name in ["Prim", "PrimBox"]):
            if rotation_matrix is None:
                if selected_indices is None:
                    raise ValueError("For PRIM, you must provide selected_indices.")
                boxes_df = extract_prim_boxes(obj, selected_indices)
                boxes_df["source"] = "PRIM"
                boxes_df["source_algorithm"] = "PRIM"
                source_labels = ["PRIM"]
            else:
                if selected_indices is None:
                    raise ValueError("For PCA-PRIM, you must provide selected_indices.")
                inferred_rotation_feature_names = rotation_feature_names
                if inferred_rotation_feature_names is None:
                    inferred_rotation_feature_names = list(obj.box_lims[0].columns)

                boxes_df = extract_pca_prim_boxes(
                    obj, selected_indices, rotation_matrix, X_mean, X_std, inferred_rotation_feature_names
                )
                boxes_df["source"] = "PCA-PRIM"
                boxes_df["source_algorithm"] = "PCA-PRIM"
                source_labels = ["PCA-PRIM"]

        else:
            raise TypeError(f"Unsupported object type: {type(obj)}")

    # --- Plotting ---
    apply_global_plot_settings()
    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    # Plot background points
    X_plot = X[feature_names]
    ax.scatter(
        X_plot[y == 0].iloc[:, 0],
        X_plot[y == 0].iloc[:, 1],
        c=PRIMARY_LIGHT,
        edgecolor='none',
        alpha=0.7,
        label="Class 0"
    )
    ax.scatter(
        X_plot[y == 1].iloc[:, 0],
        X_plot[y == 1].iloc[:, 1],
        c=PRIMARY_DARK,
        edgecolor='none',
        alpha=0.7,
        label="Class 1"
    )

    # Plot boxes
    ax = _plot_boxes_2d(boxes_df, ax=ax, source_labels=source_labels)

    # --- Add combined info + Rule Set ---
    N_total = len(X)
    N_class1_total = np.sum(y == 1)
    class1_in_box_total = 0
    points_in_box_total = 0

    textstr = ""

    for label in source_labels:
        rows = boxes_df[boxes_df["source"] == label]

        label_class1_in_box = 0
        label_points_in_box = 0

        for i, row in rows.iterrows():
            box_id = row["box_id"]
            box_class1_in_box = row["coverage"] * N_class1_total
            box_points_in_box = row["mass"] * N_total

            label_class1_in_box += box_class1_in_box
            label_points_in_box += box_points_in_box

            box_display_id = row["box_display_id"]
            textstr += f"Box {box_display_id}:\n"
            textstr += f"    Coverage: {row['coverage']:.2f}\n"
            textstr += f"    Density:  {row['density']:.2f}\n"
            textstr += f"    Rule Set:\n"

            if row["source_algorithm"] == "PRIM" or row["source_algorithm"] == "CART":
                xs = [c[0] for c in row["coords"]]
                ys = [c[1] for c in row["coords"]]
                xmin = min(xs)
                xmax = max(xs)
                ymin = min(ys)
                ymax = max(ys)
                textstr += f"      {var1} in [{xmin:.2f}, {xmax:.2f}]\n"
                textstr += f"      {var2} in [{ymin:.2f}, {ymax:.2f}]\n"

            elif row["source_algorithm"] == "PCA-PRIM":
                # Safe W conversion
                W = np.array(rotation_matrix.T)
                mu = X_mean
                sigma = X_std

                if boxes is not None:
                    box_lims = row["box_lim"]
                else:
                    box_lims = obj.box_lims[0]

                inferred_rotation_feature_names = rotation_feature_names
                if inferred_rotation_feature_names is None:
                    inferred_rotation_feature_names = list(box_lims.columns)

                for j, r_feature in enumerate(inferred_rotation_feature_names):
                    l_j = box_lims.iloc[0, j]
                    u_j = box_lims.iloc[1, j]

                    # Compute a_i for each feature
                    a1 = W[0, j] / sigma[0]
                    a2 = W[1, j] / sigma[1]

                    # Compute offset term
                    offset = (a1 * mu[0] + a2 * mu[1])

                    # Compute full thresholds
                    b_lower = l_j + offset
                    b_upper = u_j + offset

                    # Now print in original space:
                    textstr += f"      {a1:.2f} * {var1} + {a2:.2f} * {var2} ≥ {b_lower:.2f}\n"
                    textstr += f"      {a1:.2f} * {var1} + {a2:.2f} * {var2} ≤ {b_upper:.2f}\n"

            textstr += "\n"

        label_combined_coverage = label_class1_in_box / N_class1_total
        label_combined_density = label_class1_in_box / label_points_in_box

        class1_in_box_total += label_class1_in_box
        points_in_box_total += label_points_in_box

    combined_coverage = class1_in_box_total / N_class1_total
    combined_density = class1_in_box_total / points_in_box_total

    textstr += f"Combined:\n"
    textstr += f"  Coverage: {combined_coverage:.2f}\n"
    textstr += f"  Density:  {combined_density:.2f}\n"

    ax.text(
        1.02, 0.98, textstr, transform=ax.transAxes,
        fontsize=12, verticalalignment='top', horizontalalignment='left',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='round'),
    )

    beautify_plot(
        ax,
        title=title or "Scenario Discovery Boxes",
        xlabel=feature_names[0],
        ylabel=feature_names[1],
        save_path=save_path
    )

    fig.subplots_adjust(top=0.85)

    return ax


def _plot_boxes_2d(boxes_df, ax, source_labels):
    # Prepare initial fixed colors
    fixed_colors = [SECONDARY_MIDDLE, SECONDARY_LIGHT, SECONDARY_DARK]

    # If we have more labels than fixed colors, get remaining colors from tab10
    remaining_n = max(0, len(source_labels) - len(fixed_colors))
    if remaining_n > 0:
        fallback_cmap = plt.cm.get_cmap("tab10", remaining_n)
        fallback_colors = [mcolors.to_hex(fallback_cmap(i)) for i in range(remaining_n)]
    else:
        fallback_colors = []

    # Combine fixed colors and fallback colors
    full_color_list = fixed_colors[:len(source_labels)] + fallback_colors

    # Map source_labels to colors
    source_color_map = {
        label: full_color_list[i] for i, label in enumerate(source_labels)
    }

    for label in source_labels:
        rows = boxes_df[boxes_df["source"] == label]
        for _, row in rows.iterrows():
            coords = row["coords"]
            ordered_coords = order_polygon_corners_2d(coords)
            polygon = np.array([[point[0], point[1]] for point in ordered_coords])

            patch = patches.Polygon(
                polygon,
                closed=True,
                facecolor='none',
                edgecolor=source_color_map[label],
                linewidth=2,
                alpha=1
            )
            ax.add_patch(patch)

            xs = polygon[:, 0]
            ys = polygon[:, 1]
            x_center = (xs.min() + xs.max()) / 2
            y_center = (ys.min() + ys.max()) / 2

            if row["source_algorithm"] == "CART":
                ax.text(
                    x_center,
                    y_center,
                    str(row["box_id"]),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=source_color_map[label],
                    weight="bold",
                    bbox=dict(
                        facecolor='white',
                        edgecolor='none',
                        alpha=0.6,
                        boxstyle='circle,pad=0.2'
                    )
                )

            # ax.text(
            #     x_center,
            #     y_center,
            #     str(row["box_id"]),
            #     ha="center",
            #     va="center",
            #     fontsize=12,
            #     color=source_color_map[label],
            #     weight="bold",
            #     bbox=dict(
            #         facecolor='white',
            #         edgecolor='none',
            #         alpha=0.6,
            #         boxstyle='circle,pad=0.2'
            #     )
            # )

    handles = [
        patches.Patch(color=source_color_map[label], label=label)
        for label in source_labels
    ]
    ax.legend(handles=handles, loc="best")

    return ax
