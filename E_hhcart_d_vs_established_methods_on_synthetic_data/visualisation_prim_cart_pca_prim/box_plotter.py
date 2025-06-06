"""
Box Plotter utilities for visualising scenario discovery boxes.

Provides a unified `plot_boxes()` function to visualise:
- PRIM boxes (selected by user)
- PCA-PRIM boxes (selected by user, rotated back)
- CART boxes predicting class 1

All plotting is polygon-based, using the corner coordinates extracted.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.config.colors_and_plot_styles import PRIMARY_LIGHT, PRIMARY_DARK, SECONDARY_DARK
from src.config.plot_settings import apply_global_plot_settings, beautify_plot

from .extract_cart_boxes import extract_cart_boxes
from .extract_prim_boxes import extract_prim_boxes
from .extract_pca_prim_boxes import extract_pca_prim_boxes
from .box_utils import order_polygon_corners_2d


def plot_boxes(obj, X=None, y=None, selected_indices=None,
               rotation_matrix=None, X_mean=None, X_std=None,
               feature_names=None, rotation_feature_names=None,
               title=None, save_path=None):
    """
    Unified box plotter for PRIM, PCA-PRIM and CART.

    Parameters
    ----------
    obj : Prim | CART | Prim (for PCA-PRIM)
        The fitted model object.
    X : pd.DataFrame
        The original input features. Required to plot background points.
    y : np.array
        The target labels (0/1). Required to plot background points.
    selected_indices : list of int, optional
        Required for PRIM and PCA-PRIM. Indices of the boxes to extract.
    rotation_matrix : np.ndarray, optional
        Required for PCA-PRIM. Rotation matrix used during preprocessing.
    X_mean : np.ndarray, optional
        Required for PCA-PRIM. Mean of the original features used during preprocessing.
    X_std : np.ndarray, optional
        Required for PCA-PRIM. Standard deviation of the original features used during preprocessing.
    feature_names : list of str
        Names of the features to plot background points (must be 2D for plotting).
    rotation_feature_names : list of str, optional
        Names of the rotated features to use when extracting PCA-PRIM boxes.
        If not provided, falls back to feature_names (for backward compatibility).
    title : str, optional
        Title of the plot.
    save_path : str, optional
        Path to save the figure (PDF).

    Returns
    -------
    matplotlib Axes
        The axes with the plotted boxes.
    """
    # Dispatch based on object type
    if obj.__class__.__name__ == "CART":
        boxes_df = extract_cart_boxes(obj)

    elif any(name in str(type(obj)) for name in ["Prim", "PrimBox"]):
        if rotation_matrix is None:
            # Standard PRIM
            if selected_indices is None:
                raise ValueError("For PRIM, you must provide selected_indices.")
            boxes_df = extract_prim_boxes(obj, selected_indices)
        else:
            # PCA-PRIM
            if selected_indices is None:
                raise ValueError("For PCA-PRIM, you must provide selected_indices.")
            if rotation_feature_names is None:
                rotation_feature_names = feature_names  # fallback for backward compatibility
            boxes_df = extract_pca_prim_boxes(obj, selected_indices, rotation_matrix,
                                              X_mean, X_std, rotation_feature_names)

    else:
        raise TypeError(f"Unsupported object type: {type(obj)}")

    # Sanity check
    if feature_names is None:
        raise ValueError("You must provide feature_names for plotting.")

    if len(feature_names) != 2:
        raise ValueError("plot_boxes() currently supports only 2D plotting (len(feature_names) == 2).")

    # Apply global plot settings
    apply_global_plot_settings()

    # Prepare axis
    fig, ax = plt.subplots(figsize=(5, 5))

    # --- Plot background points first ---
    if X is not None and y is not None:
        X_plot = X[feature_names]

        # Plot class 0
        ax.scatter(
            X_plot[y == 0].iloc[:, 0],
            X_plot[y == 0].iloc[:, 1],
            c=PRIMARY_LIGHT,
            edgecolor='none',
            alpha=0.7,
            label="Class 0"
        )
        # Plot class 1
        ax.scatter(
            X_plot[y == 1].iloc[:, 0],
            X_plot[y == 1].iloc[:, 1],
            c=PRIMARY_DARK,
            edgecolor='none',
            alpha=0.7,
            label="Class 1"
        )

    # --- Plot the boxes ---
    ax = _plot_boxes_2d(boxes_df, feature_names, ax=ax)

    # Beautify
    beautify_plot(
        ax,
        title=title or "Scenario Discovery Boxes",
        xlabel=feature_names[0],
        ylabel=feature_names[1],
        save_path=save_path
    )

    return ax


def _plot_boxes_2d(boxes_df, feature_names, ax):
    """
    Internal function to plot boxes as polygons in 2D.

    Parameters
    ----------
    boxes_df : pd.DataFrame
        DataFrame returned by an extract_*_boxes() function.
    feature_names : list of str
        Names of the two features to plot.
    ax : matplotlib Axes
        Axes to plot on.

    Returns
    -------
    matplotlib Axes
    """
    for _, row in boxes_df.iterrows():
        coords = row["coords"]  # list of tuples

        # Reorder the corners to form a convex polygon
        ordered_coords = order_polygon_corners_2d(coords)

        # Convert to numpy array
        polygon = np.array([[point[0], point[1]] for point in ordered_coords])

        # Plot polygon
        patch = patches.Polygon(
            polygon,
            closed=True,
            facecolor="none",
            edgecolor=SECONDARY_DARK,
            linewidth=2,
            alpha=1
        )
        ax.add_patch(patch)

        # Compute true box center for label
        xs = polygon[:, 0]
        ys = polygon[:, 1]
        x_center = (xs.min() + xs.max()) / 2
        y_center = (ys.min() + ys.max()) / 2

        # Plot box ID at center
        ax.text(
            x_center,
            y_center,
            str(row["box_id"]),
            ha="center",
            va="center",
            fontsize=12,
            color=SECONDARY_DARK,
            weight="bold",
            bbox=dict(
                facecolor='lightgreen',
                edgecolor='none',
                alpha=0.4,
                boxstyle='circle,pad=0.2'
            )
        )

    return ax
