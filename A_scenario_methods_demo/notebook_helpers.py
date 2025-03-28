"""
Helper functions for the visualisation notebook.

This module offloads update and save functions so that the notebook remains concise.
It saves all outputs—including individual evolution boxes for each step of the PRIM and PCA–PRIM algorithms—
into dedicated folders.
"""

import os
from matplotlib.colors import to_rgba
import pandas as pd
from IPython.display import display
from matplotlib.patches import Rectangle

from A_scenario_methods_demo.analysis import run_analysis
from A_scenario_methods_demo.plotting.base_plots import plot_base, generic_plot
from A_scenario_methods_demo.plotting.prim_plots import (
    plot_original_data,
    plot_spatial_evolution,
    plot_peeling_trajectory,
    plot_rotated_data,
    plot_rotated_with_boxes,
    plot_original_with_boxes,
    plot_overlayed_peeling_trajectories,
    plot_peeling_trajectory_with_constraint_colors,
)
from A_scenario_methods_demo.plotting.cart_plots import plot_cart_spatial_evolution
from A_scenario_methods_demo.utils import interpolate_color
from A_scenario_methods_demo.pca_rotation_module import transform_box_to_original
from src.config.paths import SCENARIO_METHODS_DEMO_OUTPUTS_DIR
from src.config.colors import SECONDARY_LIGHT, SECONDARY_DARK

# Define save paths using the configuration
SAVE_PATHS = {
    "original_data": os.path.join(SCENARIO_METHODS_DEMO_OUTPUTS_DIR, "data_plot.pdf"),
    "prim_spatial": os.path.join(SCENARIO_METHODS_DEMO_OUTPUTS_DIR, "PRIM", "prim_box_evolution.pdf"),
    "prim_peeling": os.path.join(SCENARIO_METHODS_DEMO_OUTPUTS_DIR, "PRIM", "prim_peeling_trajectory.pdf"),
    "pcaprim_rotated": os.path.join(SCENARIO_METHODS_DEMO_OUTPUTS_DIR, "PCA_PRIM", "pcaprim_rotated_data.pdf"),
    "pcaprim_rotated_peeling": os.path.join(
        SCENARIO_METHODS_DEMO_OUTPUTS_DIR, "PCA_PRIM", "pcaprim_rotated_box_evolution.pdf"
    ),
    "pcaprim_spatial_orig": os.path.join(
        SCENARIO_METHODS_DEMO_OUTPUTS_DIR, "PCA_PRIM", "pcaprim_original_box_evolution.pdf"
    ),
    "pcaprim_peeling": os.path.join(
        SCENARIO_METHODS_DEMO_OUTPUTS_DIR, "PCA_PRIM", "pcaprim_peeling_trajectory.pdf"
    ),
    "pcaprim_spatial_rotated": os.path.join(
        SCENARIO_METHODS_DEMO_OUTPUTS_DIR, "PCA_PRIM", "pcaprim_spatial_evolution_rotated.pdf"
    ),
    "overlay": os.path.join(SCENARIO_METHODS_DEMO_OUTPUTS_DIR, "peeling_trajectory_prim_vs_pca_prim.pdf"),
    "cart": os.path.join(SCENARIO_METHODS_DEMO_OUTPUTS_DIR, "CART", "cart_plot.pdf"),
    "prim_peeling_constraints": os.path.join(
        SCENARIO_METHODS_DEMO_OUTPUTS_DIR, "PRIM", "peeling_trajectory_with_constraints.pdf"
    ),
}

# Define separate directories for evolution boxes
PRIM_EVOLUTION_DIR = os.path.join(SCENARIO_METHODS_DEMO_OUTPUTS_DIR, "PRIM", "evolution")
os.makedirs(PRIM_EVOLUTION_DIR, exist_ok=True)
PCA_PRIM_EVOLUTION_DIR = os.path.join(SCENARIO_METHODS_DEMO_OUTPUTS_DIR, "PCA_PRIM", "evolution")
os.makedirs(PCA_PRIM_EVOLUTION_DIR, exist_ok=True)


def update_plots(quad_sliders, num_dots, frac_inside, frac_outside, peel_frac,
                 prim_mass_min, cart_mass_min, plot_outputs, table_output):
    """
    Update the interactive plots using the current widget values.

    Returns:
        dict: The analysis results.
    """
    quad_coords = [slider.value for slider in quad_sliders]
    results = run_analysis(num_dots, quad_coords, frac_inside, frac_outside,
                           peel_frac, prim_mass_min, cart_mass_min)

    for out in plot_outputs:
        out.clear_output(wait=True)

    with plot_outputs[0]:
        plot_original_data(
            results["samples"], results["y_labels"], results["quadrilateral"],
            title="Original Data", note="", save_path=None, axis_limits=(0, 1, 0, 1)
        )

    with plot_outputs[1]:
        plot_spatial_evolution(
            results["samples"], results["y_labels"], results["quadrilateral"],
            results["boxes_history"], title="PRIM: Original Box Evolution",
            note="", save_path=None, axis_limits=(0, 1, 0, 1)
        )

    with plot_outputs[2]:
        plot_peeling_trajectory(
            results["prim_orig_cov"], results["prim_orig_dens"],
            title="PRIM: Peeling Trajectory", note="", save_path=None
        )

    with plot_outputs[3]:
        plot_rotated_data(
            results["X_rot"], results["y_labels"], results["quadrilateral_rot"],
            title="PCA-PRIM: Rotated Data", note="", save_path=None
        )

    with plot_outputs[4]:
        plot_rotated_with_boxes(
            results["X_rot"], results["y_labels"], results["quadrilateral_rot"],
            results["boxes_history_rot"], title="PCA-PRIM: Rotated Box Evolution",
            note="", save_path=None
        )

    with plot_outputs[5]:
        plot_original_with_boxes(
            results["samples"], results["y_labels"], results["boxes_history_rot"],
            results["V"], results["mu"], results["quadrilateral"],
            title="PCA-PRIM: Original Box Evolution", note="", save_path=None
        )

    with plot_outputs[6]:
        plot_peeling_trajectory(
            results["pcaprim_cov"], results["pcaprim_dens"],
            title="PCA-PRIM: Peeling Trajectory", note="", save_path=None
        )

    with plot_outputs[7]:
        plot_overlayed_peeling_trajectories(
            results["prim_orig_cov"], results["prim_orig_dens"],
            results["pcaprim_cov"], results["pcaprim_dens"],
            title="PRIM vs PCA-PRIM: Peeling Trajectories", save_path=None
        )

    with plot_outputs[8]:
        # Always update the CART plot block so that it is part of the grid.
        plot_cart_spatial_evolution(
            results["samples"], results["y_labels"], results["quadrilateral"],
            results["cart_boxes"], results["cart_class"],
            title="CART: Identified Boxes", note="",
            save_path=None, axis_limits=(0, 1, 0, 1)
        )
        print("CART Boxes Details:")
        for i, box in enumerate(results["cart_boxes"]):
            print(f"Box {i}:")
            print(f"   Limits: x: {box['x']}, y: {box['y']}")
            print(f"   Predicted Class: {results['cart_class'][i]}")
            print(f"   Coverage: {results['cart_cov'][i]:.2f}, Density: {results['cart_dens'][i]:.2f}")

    # Update summary table
    indices = [i for i, cls in enumerate(results["cart_class"]) if cls == 1]
    combined_coverage = sum(results["cart_cov"][i] for i in indices)
    combined_density = (sum(results["cart_cov"][i] * results["cart_dens"][i] for i in indices) /
                        combined_coverage) if combined_coverage > 0 else 0
    n_boxes = len(results["cart_boxes"])
    table_output.layout.height = f"{n_boxes * 37 + 30}px"
    table_output.clear_output(wait=True)
    with table_output:
        print(f"Combined Coverage (Class 1): {combined_coverage:.2f}")
        print(f"Combined Density (Class 1): {combined_density:.2f}\n")
        table_data = {
            "Box": list(range(len(results["cart_boxes"]))),
            "x_lim": [f"[{box['x'][0]:.2f}, {box['x'][1]:.2f}]" for box in results["cart_boxes"]],
            "y_lim": [f"[{box['y'][0]:.2f}, {box['y'][1]:.2f}]" for box in results["cart_boxes"]],
            "Coverage": [f"{cov:.2f}" for cov in results["cart_cov"]],
            "Density": [f"{dens:.2f}" for dens in results["cart_dens"]],
            "Predicted Class": results["cart_class"],
        }
        df_table = pd.DataFrame(table_data)
        print("CART Boxes Summary Table:")
        display(df_table)
    return results


def save_prim_plots(quad_sliders, num_dots, frac_inside, frac_outside, peel_frac,
                    prim_mass_min, cart_mass_min):
    """
    Save all PRIM and PCA-PRIM plots, including individual evolution boxes for every step.
    Evolution boxes from PRIM are saved in PRIM_EVOLUTION_DIR and PCA-PRIM boxes in PCA_PRIM_EVOLUTION_DIR.
    """
    quad_coords = [slider.value for slider in quad_sliders]
    results = run_analysis(num_dots, quad_coords, frac_inside, frac_outside,
                           peel_frac, prim_mass_min, cart_mass_min)

    # Save Sampled Distribution Plot
    plot_original_data(results["samples"], results["y_labels"], results["quadrilateral"],
                       title="Sampled Distribution", note="",
                       save_path=SAVE_PATHS["original_data"], axis_limits=(0, 1, 0, 1))

    # Save PRIM Analysis Plots
    plot_spatial_evolution(results["samples"], results["y_labels"], results["quadrilateral"],
                           results["boxes_history"], title="PRIM Box Progression", note="",
                           save_path=SAVE_PATHS["prim_spatial"], axis_limits=(0, 1, 0, 1))
    plot_peeling_trajectory(results["prim_orig_cov"], results["prim_orig_dens"],
                            title="PRIM Peeling Trajectory", note="",
                            save_path=SAVE_PATHS["prim_peeling"])

    # Save PCA-PRIM Analysis Plots
    plot_rotated_data(results["X_rot"], results["y_labels"], results["quadrilateral_rot"],
                      title="PCA Rotated Sampled Distribution", note="",
                      save_path=SAVE_PATHS["pcaprim_rotated"])
    plot_rotated_with_boxes(results["X_rot"], results["y_labels"], results["quadrilateral_rot"],
                            results["boxes_history_rot"], title="PCA-PRIM Rotated Box Progression", note="",
                            save_path=SAVE_PATHS["pcaprim_rotated_peeling"])
    plot_original_with_boxes(results["samples"], results["y_labels"], results["boxes_history_rot"],
                             results["V"], results["mu"], results["quadrilateral"],
                             title="PCA-PRIM Original Box Progression", note="",
                             save_path=SAVE_PATHS["pcaprim_spatial_orig"])
    plot_peeling_trajectory(results["pcaprim_cov"], results["pcaprim_dens"],
                            title="PCA-PRIM Peeling Trajectory", note="",
                            save_path=SAVE_PATHS["pcaprim_peeling"])

    # Save Overlayed Peeling Trajectories
    plot_overlayed_peeling_trajectories(results["prim_orig_cov"], results["prim_orig_dens"],
                                        results["pcaprim_cov"], results["pcaprim_dens"],
                                        title="Peeling Trajectories: PRIM vs. PCA-PRIM",
                                        save_path=SAVE_PATHS["overlay"])

    # Save individual PRIM evolution boxes (one file per evolution step)
    n_boxes = len(results["boxes_history"])
    for i, box in enumerate(results["boxes_history"]):
        prim_box_save_path = os.path.join(PRIM_EVOLUTION_DIR, f"box_{i + 1}.pdf")
        t = i / (n_boxes - 1) if n_boxes > 1 else 0
        box_color = interpolate_color(SECONDARY_LIGHT, SECONDARY_DARK, t)
        fill_rgba = to_rgba(box_color, alpha=0.2)
        edge_rgba = box_color

        def draw(ax, box_data=box, fill_color=fill_rgba, edge_color=edge_rgba):
            plot_base(ax, results["samples"], results["y_labels"], results["quadrilateral"],
                      quadrilateral_label="Sampled quadrilateral", xlim=(0, 1), ylim=(0, 1))
            rect = Rectangle((box_data["x"][0], box_data["y"][0]),
                             box_data["x"][1] - box_data["x"][0],
                             box_data["y"][1] - box_data["y"][0],
                             fill=True, facecolor=fill_color, edgecolor=edge_color,
                             linewidth=2, linestyle="--", zorder=4)
            ax.add_patch(rect)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        generic_plot(f"PRIM Box {i + 1}", "X-axis", "Y-axis", "",
                     prim_box_save_path, draw, save_figsize=(6, 5), grid=False)

    # Save individual PCA-PRIM evolution boxes (each transformed back to original coordinates)
    n_boxes_rot = len(results["boxes_history_rot"])
    for i, box_rot in enumerate(results["boxes_history_rot"]):
        pca_box_save_path = os.path.join(PCA_PRIM_EVOLUTION_DIR, f"box_{i + 1}.pdf")
        t = i / (n_boxes_rot - 1) if n_boxes_rot > 1 else 0
        box_color = interpolate_color(SECONDARY_LIGHT, SECONDARY_DARK, t)
        fill_rgba = to_rgba(box_color, alpha=0.2)
        edge_rgba = box_color
        box_orig = transform_box_to_original(box_rot, results["V"], results["mu"])

        def draw(ax, box_data=box_orig, fill_color=fill_rgba, edge_color=edge_rgba):
            from matplotlib.patches import Polygon
            plot_base(ax, results["samples"], results["y_labels"], results["quadrilateral"],
                      quadrilateral_label="Sampled quadrilateral", xlim=(0, 1), ylim=(0, 1))
            poly = Polygon(box_data, closed=True, fill=True, facecolor=fill_color, edgecolor=edge_color,
                           linewidth=2, linestyle="--", zorder=4)
            ax.add_patch(poly)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        generic_plot(f"PCA-PRIM Box {i + 1}", "X-axis", "Y-axis", "",
                     pca_box_save_path, draw, save_figsize=(6, 5), grid=False)

    # Save enhanced PRIM peeling trajectory (with constraints)
    orig_bounds = {
        "x": (results["samples"][:, 0].min(), results["samples"][:, 0].max()),
        "y": (results["samples"][:, 1].min(), results["samples"][:, 1].max())
    }
    peeling_trajectory_save_path = SAVE_PATHS["prim_peeling_constraints"]
    plot_peeling_trajectory_with_constraint_colors(
        results["prim_orig_cov"], results["prim_orig_dens"], results["boxes_history"],
        orig_bounds, title="PRIM Peeling Trajectory", save_path=peeling_trajectory_save_path
    )

    # Save enhanced PCA-PRIM peeling trajectory (with constraints)
    orig_bounds_rot = {
        "x": (results["X_rot"][:, 0].min(), results["X_rot"][:, 0].max()),
        "y": (results["X_rot"][:, 1].min(), results["X_rot"][:, 1].max())
    }
    pcaprim_peeling_trajectory_save_path = os.path.join(
        SCENARIO_METHODS_DEMO_OUTPUTS_DIR, "PCA_PRIM", "peeling_trajectory_with_constraints.pdf"
    )
    plot_peeling_trajectory_with_constraint_colors(
        results["pcaprim_cov"], results["pcaprim_dens"], results["boxes_history_rot"],
        orig_bounds_rot, title="PCA-PRIM Peeling Trajectory", save_path=pcaprim_peeling_trajectory_save_path
    )

    print("PRIM plots have been saved to the designated locations.")


def save_cart_plots(quad_sliders, num_dots, frac_inside, frac_outside, peel_frac, prim_mass_min, cart_mass_min):
    quad_coords = [slider.value for slider in quad_sliders]
    results = run_analysis(num_dots, quad_coords,
                           frac_inside, frac_outside,
                           peel_frac, prim_mass_min, cart_mass_min)
    cart_mass_value = cart_mass_min
    CART_DIR = os.path.join(SCENARIO_METHODS_DEMO_OUTPUTS_DIR, "CART")
    os.makedirs(CART_DIR, exist_ok=True)
    cart_save_path = os.path.join(CART_DIR, f"cart_plot_mass_min_{cart_mass_value:.2f}.pdf")
    plot_cart_spatial_evolution(results["samples"], results["y_labels"], results["quadrilateral"],
                                results["cart_boxes"], results["cart_class"],
                                title="CART Identified Boxes", note="",
                                save_path=cart_save_path, axis_limits=(0, 1, 0, 1))
    print("CART plots have been saved to the designated locations.")
