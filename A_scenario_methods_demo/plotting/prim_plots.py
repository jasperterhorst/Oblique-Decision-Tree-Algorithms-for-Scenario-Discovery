"""
PRIM and PCA-PRIM plotting routines for the scenario methods demo.
"""

import numpy as np
from matplotlib.patches import Rectangle, Polygon
from A_scenario_methods_demo.plotting.base_plots import generic_plot, plot_base
from A_scenario_methods_demo.utils import interpolate_color, lighten_color
from A_scenario_methods_demo.pca_rotation_module import transform_box_to_original
from src.config.colors import (
    PRIMARY_LIGHT, PRIMARY_MIDDLE, PRIMARY_DARK,
    SECONDARY_LIGHT, SECONDARY_MIDDLE, SECONDARY_DARK,
    QUADRILATERAL_COLOR
)


def plot_original_data(samples, y, quadrilateral, title="Sampled Data", note="",
                       save_path=None, axis_limits=(0, 1, 0, 1)):
    def draw(ax):
        xmin, xmax, ymin, ymax = axis_limits
        plot_base(ax, samples, y, quadrilateral, quadrilateral_label="Sampled Quadrilateral", xlim=(xmin, xmax),
                  ylim=(ymin, ymax))
    generic_plot(title, "X-axis", "Y-axis", note, save_path, draw, grid=False)


def plot_spatial_evolution(samples, y, quadrilateral, boxes_history,
                           start_color=SECONDARY_LIGHT, end_color=SECONDARY_DARK,
                           title="Spatial Evolution", note="", save_path=None, axis_limits=None):
    def draw(ax):
        if axis_limits:
            xmin, xmax, ymin, ymax = axis_limits
        else:
            xmin, xmax = np.min(samples[:, 0]), np.max(samples[:, 0])
            ymin, ymax = np.min(samples[:, 1]), np.max(samples[:, 1])
        plot_base(ax, samples, y, quadrilateral, xlim=(xmin, xmax), ylim=(ymin, ymax))
        for i, box in enumerate(boxes_history):
            t = i / (len(boxes_history) - 1) if len(boxes_history) > 1 else 0
            box_color = interpolate_color(start_color, end_color, t)
            rect = Rectangle((box["x"][0], box["y"][0]),
                             box["x"][1] - box["x"][0],
                             box["y"][1] - box["y"][0],
                             fill=False, edgecolor=box_color, linewidth=2,
                             linestyle="--", zorder=4, label="Cut Outline" if i == 0 else None)
            ax.add_patch(rect)
    generic_plot(title, "X-axis", "Y-axis", note, save_path, draw, grid=False)


def plot_rotated_data(x_rot, y, quadrilateral_rot, title="Rotated Data (PCA Space)", note="", save_path=None,
                      axis_limits=None):
    def draw(ax):
        if axis_limits:
            xmin, xmax, ymin, ymax = axis_limits
        else:
            xmin, xmax = np.min(x_rot[:, 0]), np.max(x_rot[:, 0])
            ymin, ymax = np.min(x_rot[:, 1]), np.max(x_rot[:, 1])
        plot_base(ax, x_rot, y, quadrilateral_rot, quadrilateral_label="Rotated Quadrilateral", xlim=(xmin, xmax),
                  ylim=(ymin, ymax))
    generic_plot(title, "PCA 1", "PCA 2", note, save_path, draw, grid=False)


def plot_rotated_with_boxes(x_rot, y, quadrilateral_rot, boxes_history_rot,
                            title="Box Evolution in PCA Space", note="", save_path=None):
    def draw(ax):
        xmin, xmax = np.min(x_rot[:, 0]), np.max(x_rot[:, 0])
        ymin, ymax = np.min(x_rot[:, 1]), np.max(x_rot[:, 1])
        plot_base(ax, x_rot, y, quadrilateral_rot, quadrilateral_label="Rotated Quadrilateral", xlim=(xmin, xmax),
                  ylim=(ymin, ymax))
        for i, box in enumerate(boxes_history_rot):
            t = i / (len(boxes_history_rot) - 1) if len(boxes_history_rot) > 1 else 0
            box_color = interpolate_color(SECONDARY_LIGHT, SECONDARY_DARK, t)
            poly = Polygon(np.array([[box['x'][0], box['y'][0]],
                                     [box['x'][1], box['y'][0]],
                                     [box['x'][1], box['y'][1]],
                                     [box['x'][0], box['y'][1]]]),
                           closed=True, fill=False, edgecolor=box_color,
                           linewidth=2, linestyle="--", zorder=4,
                           label="Cut Outline" if i == 0 else None)
            ax.add_patch(poly)
    generic_plot(title, "PCA 1", "PCA 2", note, save_path, draw, grid=False)


def plot_original_with_boxes(original_samples, y, boxes_history_rot, v, mu, quadrilateral,
                             title="Box Evolution in Original Coordinates", note="", save_path=None):
    def draw(ax):
        plot_base(ax, original_samples, y, quadrilateral, quadrilateral_label="Sampled Quadrilateral",
                  xlim=(0, 1), ylim=(0, 1))
        for i, box_rot in enumerate(boxes_history_rot):
            t = i / (len(boxes_history_rot) - 1) if len(boxes_history_rot) > 1 else 0
            box_color = interpolate_color(SECONDARY_LIGHT, SECONDARY_DARK, t)
            corners_orig = transform_box_to_original(box_rot, v, mu)
            poly = Polygon(corners_orig, closed=True, fill=False, edgecolor=box_color, linewidth=2,
                           linestyle="--", zorder=4, label="Cut Outline" if i == 0 else None)
            ax.add_patch(poly)
    generic_plot(title, "X-axis", "Y-axis", note, save_path, draw, grid=False)


def plot_overlayed_peeling_trajectories(prim_cov, prim_dens, pcaprim_cov, pcaprim_dens,
                                        title="Overlayed Peeling Trajectories", save_path=None):
    def draw(ax):
        ax.plot(prim_cov, prim_dens, linestyle="-", color=PRIMARY_MIDDLE, marker="o", markersize=6,
                label="PRIM Trajectory", zorder=3)
        ax.plot(pcaprim_cov, pcaprim_dens, linestyle="-", color=SECONDARY_MIDDLE, marker="s", markersize=6,
                label="PCA-PRIM Trajectory", zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    generic_plot(title, "Coverage", "Density", "", save_path, draw)


def plot_peeling_trajectory(coverage_vals, density_vals, start_color=SECONDARY_LIGHT, end_color=SECONDARY_DARK,
                            title="Peeling Trajectory", note="", save_path=None):
    def draw(ax):
        n = len(coverage_vals)
        ax.plot(coverage_vals, density_vals, linestyle="-", color="gray", linewidth=1.5, alpha=0.8, zorder=1)
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0
            point_color = interpolate_color(start_color, end_color, t)
            ax.scatter(coverage_vals[i], density_vals[i], color=point_color, s=60, zorder=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    generic_plot(title, "Coverage", "Density", note, save_path, draw)


def plot_peeling_trajectory_with_constraint_colors(coverage_vals, density_vals, boxes_history, orig_bounds,
                                                   title="Peeling Trajectory with Constraint Colors", save_path=None):
    def draw(ax):
        n = len(coverage_vals)
        for i in range(n):
            box = boxes_history[i]
            x_constrained = (not np.isclose(box['x'][0], orig_bounds['x'][0], atol=1e-6) or
                             not np.isclose(box['x'][1], orig_bounds['x'][1], atol=1e-6))
            y_constrained = (not np.isclose(box['y'][0], orig_bounds['y'][0], atol=1e-6) or
                             not np.isclose(box['y'][1], orig_bounds['y'][1], atol=1e-6))
            if x_constrained and y_constrained:
                color = PRIMARY_DARK
            elif x_constrained or y_constrained:
                color = lighten_color(PRIMARY_DARK, amount=0.5)
            else:
                color = QUADRILATERAL_COLOR  # Use this as a neutral gray
            ax.scatter(coverage_vals[i], density_vals[i], color=color, s=60, zorder=3)

        ax.plot(coverage_vals, density_vals, linestyle="-", color="gray", linewidth=1.5, alpha=0.8, zorder=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Density")

        dummy_both = ax.scatter([], [], color=PRIMARY_DARK, s=60, label="Both Dimensions Constrained")
        dummy_one = ax.scatter([], [], color=lighten_color(PRIMARY_DARK, amount=0.5), s=60,
                               label="One Dimension Constrained")
        dummy_none = ax.scatter([], [], color=QUADRILATERAL_COLOR, s=60, label="No Constraint")
        ax.legend(handles=[dummy_both, dummy_one, dummy_none], loc="lower right", fontsize=10)
    generic_plot(title, "Coverage", "Density", "", save_path, draw, save_figsize=(6, 5))
