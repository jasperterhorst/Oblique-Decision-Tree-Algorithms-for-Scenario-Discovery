"""
Helper functions for the scenario methods demo visualisation notebook.

This module handles plotting and saving of all key outputs for PRIM, PCA–PRIM,
and CART algorithms—including box evolution steps, trajectories, and summary tables.

Figures and data are saved under: /_data/scenario_methods_demo_outputs
"""

import pandas as pd
from typing import Any

from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from IPython.display import display
from pathlib import Path

from A_scenario_methods_demo.analysis import run_analysis
from A_scenario_methods_demo.plotting.base_plots import plot_base, generic_plot
from A_scenario_methods_demo.plotting.prim_plots import (
    plot_original_data, plot_spatial_evolution, plot_peeling_trajectory,
    plot_rotated_data, plot_rotated_with_boxes, plot_original_with_boxes,
    plot_overlayed_peeling_trajectories, plot_peeling_trajectory_with_constraint_colors,
)
from A_scenario_methods_demo.plotting.cart_plots import plot_cart_spatial_evolution
from A_scenario_methods_demo.utils import interpolate_color
from src.config.paths import SCENARIO_METHODS_DEMO_OUTPUTS_DIR
from src.config.colors_and_plot_styles import EVOLUTION_COLORS

# === Save paths ===
SAVE_PATHS = {
    "original_data": SCENARIO_METHODS_DEMO_OUTPUTS_DIR / "data_plot.pdf",
    "prim_spatial": SCENARIO_METHODS_DEMO_OUTPUTS_DIR / "PRIM" / "prim_box_evolution.pdf",
    "prim_peeling": SCENARIO_METHODS_DEMO_OUTPUTS_DIR / "PRIM" / "prim_peeling_trajectory.pdf",
    "prim_constraints": SCENARIO_METHODS_DEMO_OUTPUTS_DIR / "PRIM" / "peeling_trajectory_with_constraints.pdf",
    "pcaprim_rotated": SCENARIO_METHODS_DEMO_OUTPUTS_DIR / "PCA_PRIM" / "pcaprim_rotated_data.pdf",
    "pcaprim_rotated_peeling": SCENARIO_METHODS_DEMO_OUTPUTS_DIR / "PCA_PRIM" / "pcaprim_rotated_box_evolution.pdf",
    "pcaprim_spatial_orig": SCENARIO_METHODS_DEMO_OUTPUTS_DIR / "PCA_PRIM" / "pcaprim_original_box_evolution.pdf",
    "pcaprim_peeling": SCENARIO_METHODS_DEMO_OUTPUTS_DIR / "PCA_PRIM" / "pcaprim_peeling_trajectory.pdf",
    "pcaprim_constraints": SCENARIO_METHODS_DEMO_OUTPUTS_DIR / "PCA_PRIM" / "peeling_trajectory_with_constraints.pdf",
    "overlay": SCENARIO_METHODS_DEMO_OUTPUTS_DIR / "peeling_trajectory_prim_vs_pca_prim.pdf",
    "cart": SCENARIO_METHODS_DEMO_OUTPUTS_DIR / "CART" / "cart_plot.pdf",
}


# === Evolution directories ===
PRIM_EVOLUTION_DIR = SCENARIO_METHODS_DEMO_OUTPUTS_DIR / "PRIM" / "evolution"
PCA_PRIM_EVOLUTION_DIR = SCENARIO_METHODS_DEMO_OUTPUTS_DIR / "PCA_PRIM" / "evolution"
PRIM_EVOLUTION_DIR.mkdir(parents=True, exist_ok=True)
PCA_PRIM_EVOLUTION_DIR.mkdir(parents=True, exist_ok=True)


# === Main interactive update ===
def update_plots(
    quad_sliders: list[Any], num_dots: int,
    frac_inside: float, frac_outside: float,
    peel_frac: float, prim_mass_min: float, cart_mass_min: float,
    plot_outputs: list[Any], table_output: Any
) -> dict:
    """
    Updates all interactive notebook plots for PRIM, PCA-PRIM, and CART.

    Parameters:
        quad_sliders (list): Sliders containing coordinates of the quadrilateral.
        num_dots (int): Number of generated samples.
        frac_inside (float): Fraction of noise inside.
        frac_outside (float): Fraction of noise outside.
        peel_frac (float): Peeling fraction used in PRIM.
        prim_mass_min (float): Minimum box mass for PRIM.
        cart_mass_min (float): Minimum box mass for CART.
        plot_outputs (list): List of output widgets.
        table_output: Widget to show summary statistics.

    Returns:
        dict: Dictionary with all analysis outputs.
    """
    quad_coords = [slider.value for slider in quad_sliders]
    results = run_analysis(num_dots, quad_coords, frac_inside, frac_outside,
                           peel_frac, prim_mass_min, cart_mass_min)

    plots = [
        (plot_original_data, [results["samples"], results["y_labels"], results["quadrilateral"]]),
        (plot_spatial_evolution, [results["samples"], results["y_labels"], results["quadrilateral"],
                                  results["boxes_history"]]),
        (plot_peeling_trajectory, [results["prim_orig_cov"], results["prim_orig_dens"]]),
        (plot_rotated_data, [results["X_rot"], results["y_labels"], results["quadrilateral_rot"]]),
        (plot_rotated_with_boxes, [results["X_rot"], results["y_labels"], results["quadrilateral_rot"],
                                   results["boxes_history_rot"]]),
        (plot_original_with_boxes, [results["samples"], results["y_labels"], results["boxes_history_rot"], results["V"],
                                    results["mu"], results["quadrilateral"]]),
        (plot_peeling_trajectory, [results["pcaprim_cov"], results["pcaprim_dens"]]),
        (plot_overlayed_peeling_trajectories, [results["prim_orig_cov"], results["prim_orig_dens"],
                                               results["pcaprim_cov"], results["pcaprim_dens"]]),
        (plot_cart_spatial_evolution, [results["samples"], results["y_labels"], results["quadrilateral"],
                                       results["cart_boxes"], results["cart_class"]]),
    ]

    for (func, args), out in zip(plots, plot_outputs):
        out.clear_output(wait=True)
        with out:
            func(*args, save_path=None)

    _display_cart_summary_table(results, table_output)
    return results


# === Save PRIM and PCA-PRIM figures ===
def save_prim_plots(
    quad_sliders: list[Any], num_dots: int,
    frac_inside: float, frac_outside: float,
    peel_frac: float, prim_mass_min: float, cart_mass_min: float
) -> None:
    """
    Save all plots for PRIM and PCA-PRIM, including spatial views, trajectories,
    and box evolution steps into the output folder.

    Parameters:
        quad_sliders (list): Sliders for defining the quadrilateral.
        num_dots (int): Number of samples.
        frac_inside (float): Fraction of label noise inside the quadrilateral.
        frac_outside (float): Fraction of label noise outside the quadrilateral.
        peel_frac (float): Peeling fraction used in PRIM.
        prim_mass_min (float): Minimum box mass for PRIM.
        cart_mass_min (float): Minimum box mass for CART (unused here).
    """
    quad_coords = [slider.value for slider in quad_sliders]
    results = run_analysis(num_dots, quad_coords, frac_inside, frac_outside,
                           peel_frac, prim_mass_min, cart_mass_min)

    # Save overview plots
    plot_original_data(results["samples"], results["y_labels"], results["quadrilateral"],
                       save_path=SAVE_PATHS["original_data"], axis_limits=(0, 1, 0, 1))
    plot_spatial_evolution(results["samples"], results["y_labels"], results["quadrilateral"],
                           results["boxes_history"], save_path=SAVE_PATHS["prim_spatial"], axis_limits=(0, 1, 0, 1))
    plot_peeling_trajectory(results["prim_orig_cov"], results["prim_orig_dens"], save_path=SAVE_PATHS["prim_peeling"],
                            title="PRIM Peeling Trajectory")
    plot_rotated_data(results["X_rot"], results["y_labels"], results["quadrilateral_rot"],
                      save_path=SAVE_PATHS["pcaprim_rotated"])
    plot_rotated_with_boxes(results["X_rot"], results["y_labels"], results["quadrilateral_rot"],
                            results["boxes_history_rot"], save_path=SAVE_PATHS["pcaprim_rotated_peeling"])
    plot_original_with_boxes(results["samples"], results["y_labels"], results["boxes_history_rot"], results["V"],
                             results["mu"], results["quadrilateral"], save_path=SAVE_PATHS["pcaprim_spatial_orig"])
    plot_peeling_trajectory(results["pcaprim_cov"], results["pcaprim_dens"], save_path=SAVE_PATHS["pcaprim_peeling"],
                            title="PCA–PRIM Peeling Trajectory")
    plot_overlayed_peeling_trajectories(results["prim_orig_cov"], results["prim_orig_dens"], results["pcaprim_cov"],
                                        results["pcaprim_dens"], save_path=SAVE_PATHS["overlay"])

    # Save individual box steps
    _save_individual_prim_boxes(results)
    _save_individual_pcaprim_boxes(results)

    # Save constrained peeling trajectories
    prim_bounds = {"x": (results["samples"][:, 0].min(), results["samples"][:, 0].max()),
                   "y": (results["samples"][:, 1].min(), results["samples"][:, 1].max())}
    plot_peeling_trajectory_with_constraint_colors(results["prim_orig_cov"], results["prim_orig_dens"],
                                                   results["boxes_history"], prim_bounds,
                                                   save_path=SAVE_PATHS["prim_constraints"],
                                                   title="PRIM Peeling Trajectory with Dimensional Constraints")

    pcaprim_bounds = {"x": (results["X_rot"][:, 0].min(), results["X_rot"][:, 0].max()),
                      "y": (results["X_rot"][:, 1].min(), results["X_rot"][:, 1].max())}
    plot_peeling_trajectory_with_constraint_colors(results["pcaprim_cov"], results["pcaprim_dens"],
                                                   results["boxes_history_rot"], pcaprim_bounds,
                                                   save_path=SAVE_PATHS["pcaprim_constraints"],
                                                   title="PCA–PRIM Peeling Trajectory with Dimensional Constraints")

    print("[✓] Saved PRIM and PCA-PRIM plots.")


def save_cart_plots(
    quad_sliders: list[Any], num_dots: int,
    frac_inside: float, frac_outside: float,
    peel_frac: float, prim_mass_min: float, cart_mass_min: float
) -> None:
    """
    Save the CART decision boundaries and associated shape into the output folder.

    Parameters:
        quad_sliders (list): UI sliders for quadrilateral points.
        num_dots (int): Number of generated samples.
        frac_inside (float): Inside noise fraction.
        frac_outside (float): Outside noise fraction.
        peel_frac (float): PRIM peeling fraction.
        prim_mass_min (float): PRIM mass threshold.
        cart_mass_min (float): CART minimum box mass to consider.
    """
    quad_coords = [slider.value for slider in quad_sliders]
    results = run_analysis(num_dots, quad_coords, frac_inside, frac_outside,
                           peel_frac, prim_mass_min, cart_mass_min)

    cart_save_path = get_cart_save_path(cart_mass_min)
    cart_save_path.parent.mkdir(parents=True, exist_ok=True)

    plot_cart_spatial_evolution(
        results["samples"], results["y_labels"], results["quadrilateral"],
        results["cart_boxes"], results["cart_class"],
        save_path=str(cart_save_path), axis_limits=(0, 1, 0, 1)
    )

    print("[✓] Saved CART plot.")


# === Internal Helpers ===
def _save_individual_prim_boxes(results: dict) -> None:
    """
    Save each PRIM box step individually as a separate PDF in the PRIM evolution folder.

    Parameters:
        results (dict): Output from run_analysis.
    """
    for i, box in enumerate(results["boxes_history"]):
        box_color = interpolate_color(EVOLUTION_COLORS["start"], EVOLUTION_COLORS["end"], i
                                      / max(len(results["boxes_history"]) - 1, 1))
        rect = Rectangle((box["x"][0], box["y"][0]), box["x"][1] - box["x"][0], box["y"][1] - box["y"][0],
                         facecolor=to_rgba(box_color, alpha=0.2), edgecolor=box_color, linewidth=2, linestyle="--",
                         zorder=4)

        def draw(ax):
            plot_base(ax, results["samples"], results["y_labels"], results["quadrilateral"],
                      quadrilateral_label="Quadrilateral", xlim=(0, 1), ylim=(0, 1))
            ax.add_patch(rect)

        out_path = PRIM_EVOLUTION_DIR / f"box_{i + 1}.pdf"
        generic_plot(f"PRIM Box {i + 1}", "X-axis", "Y-axis", "", str(out_path), draw,
                     save_figsize=(4.7, 3.7), grid=False)


def _save_individual_pcaprim_boxes(results: dict) -> None:
    """
    Save each PCA-PRIM box step individually in rotated space.

    Parameters:
        results (dict): Output from run_analysis.
    """
    for i, box in enumerate(results["boxes_history_rot"]):
        box_color = interpolate_color(EVOLUTION_COLORS["start"], EVOLUTION_COLORS["end"], i /
                                      max(len(results["boxes_history_rot"]) - 1, 1))
        rect = Rectangle((box["x"][0], box["y"][0]), box["x"][1] - box["x"][0], box["y"][1] - box["y"][0],
                         facecolor=to_rgba(box_color, alpha=0.2), edgecolor=box_color, linewidth=2, linestyle="--",
                         zorder=4)

        def draw(ax):
            plot_base(ax, results["X_rot"], results["y_labels"], results["quadrilateral_rot"],
                      quadrilateral_label="Quadrilateral")
            ax.add_patch(rect)
            ax.set_xlim(results["X_rot"][:, 0].min(), results["X_rot"][:, 0].max())
            ax.set_ylim(results["X_rot"][:, 1].min(), results["X_rot"][:, 1].max())

        out_path = PCA_PRIM_EVOLUTION_DIR / f"box_{i + 1}.pdf"
        generic_plot(f"PCA-PRIM Box {i + 1}", "PCA 1", "PCA 2", "", str(out_path), draw,
                     save_figsize=(4.7, 3.7), grid=False)


def get_cart_save_path(cart_mass_min: float) -> Path:
    """
    Generate a dynamic save path for the CART plot, including the mass threshold in the filename.

    Parameters:
        cart_mass_min (float): Minimum box mass threshold.

    Returns:
        str: Path to save the CART plot.
    """
    filename = f"cart_plot_mass_{cart_mass_min:.2f}.pdf"
    return SCENARIO_METHODS_DEMO_OUTPUTS_DIR / "CART" / filename


def _display_cart_summary_table(results: dict, table_output: Any) -> None:
    """
    Format and display a summary table of CART boxes in a provided output widget.

    Parameters:
        results (dict): Output from run_analysis.
        table_output: IPython display widget.
    """
    indices = [i for i, cls in enumerate(results["cart_class"]) if cls == 1]
    cov = sum(results["cart_cov"][i] for i in indices)
    dens = (sum(results["cart_cov"][i] * results["cart_dens"][i] for i in indices) / cov) if cov > 0 else 0

    table_output.layout.height = f"{len(results['cart_boxes']) * 37 + 30}px"
    table_output.clear_output(wait=True)

    with table_output:
        print(f"Combined Coverage (Class 1): {cov:.2f}")
        print(f"Combined Density (Class 1): {dens:.2f}\n")

        df = pd.DataFrame({
            "Box": range(len(results["cart_boxes"])),
            "x_lim": [f"[{b['x'][0]:.2f}, {b['x'][1]:.2f}]" for b in results["cart_boxes"]],
            "y_lim": [f"[{b['y'][0]:.2f}, {b['y'][1]:.2f}]" for b in results["cart_boxes"]],
            "Coverage": [f"{v:.2f}" for v in results["cart_cov"]],
            "Density": [f"{v:.2f}" for v in results["cart_dens"]],
            "Predicted Class": results["cart_class"]
        })

        print("CART Boxes Summary Table:")
        display(df)
