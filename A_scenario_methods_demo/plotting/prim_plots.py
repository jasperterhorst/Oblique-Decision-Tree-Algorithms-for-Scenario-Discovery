"""
PRIM and PCA-PRIM plotting routines for the scenario methods demo.
"""

from typing import Optional, Tuple, List
import numpy as np
from matplotlib.patches import Rectangle, Polygon
from matplotlib.axes import Axes
from A_scenario_methods_demo.plotting.base_plots import generic_plot, plot_base
from A_scenario_methods_demo.utils import interpolate_color, lighten_color
from A_scenario_methods_demo.pca_rotation_module import transform_box_to_original
from src.config import (
    SCATTER_COLORS,
    EVOLUTION_COLORS,
    QUADRILATERAL_COLOR
)


def plot_original_data(
    samples: np.ndarray,
    y: np.ndarray,
    quadrilateral: np.ndarray,
    title: str = "Sampled Data",
    note: str = "",
    save_path: Optional[str] = None,
    axis_limits: Tuple[float, float, float, float] = (0, 1, 0, 1)
) -> None:
    """
    Plot original 2D data samples with binary classification and overlay the sampling quadrilateral.
    """
    def draw(ax: Axes) -> None:
        xmin, xmax, ymin, ymax = axis_limits
        plot_base(ax, samples, y, quadrilateral, quadrilateral_label="Quadrilateral", xlim=(xmin, xmax),
                  ylim=(ymin, ymax))
    generic_plot(title, "X-axis", "Y-axis", note, save_path, draw, grid=False)


def plot_spatial_evolution(
    samples: np.ndarray,
    y: np.ndarray,
    quadrilateral: np.ndarray,
    boxes_history: List[dict],
    start_color: str = EVOLUTION_COLORS["start"],
    end_color: str = EVOLUTION_COLORS["end"],
    title: str = "Spatial Evolution",
    note: str = "",
    save_path: Optional[str] = None,
    axis_limits: Optional[Tuple[float, float, float, float]] = None
) -> None:
    """
    Plot spatial box evolution on the original data using interpolated color to show progression.
    """
    def draw(ax: Axes) -> None:
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


def plot_rotated_data(
    x_rot: np.ndarray,
    y: np.ndarray,
    quadrilateral_rot: np.ndarray,
    title: str = "Data in PCA-Rotated Space",
    note: str = "",
    save_path: Optional[str] = None,
    axis_limits: Optional[Tuple[float, float, float, float]] = None
) -> None:
    """
    Plot rotated 2D data samples (e.g. after PCA) and overlay the rotated sampling quadrilateral.
    """
    def draw(ax: Axes) -> None:
        if axis_limits:
            xmin, xmax, ymin, ymax = axis_limits
        else:
            xmin, xmax = np.min(x_rot[:, 0]), np.max(x_rot[:, 0])
            ymin, ymax = np.min(x_rot[:, 1]), np.max(x_rot[:, 1])
        plot_base(ax, x_rot, y, quadrilateral_rot, quadrilateral_label="Quadrilateral", xlim=(xmin, xmax),
                  ylim=(ymin, ymax))
    generic_plot(title, "PCA 1", "PCA 2", note, save_path, draw, grid=False)


def plot_rotated_with_boxes(
    x_rot: np.ndarray,
    y: np.ndarray,
    quadrilateral_rot: np.ndarray,
    boxes_history_rot: List[dict],
    title: str = "PRIM Evolution in PCA-Rotated Space",
    note: str = "",
    save_path: Optional[str] = None
) -> None:
    """
    Plot rotated 2D data samples and overlay a sequence of rectangular boxes to show peeling evolution.
    """
    def draw(ax: Axes) -> None:
        xmin, xmax = np.min(x_rot[:, 0]), np.max(x_rot[:, 0])
        ymin, ymax = np.min(x_rot[:, 1]), np.max(x_rot[:, 1])
        plot_base(ax, x_rot, y, quadrilateral_rot, quadrilateral_label="Quadrilateral", xlim=(xmin, xmax),
                  ylim=(ymin, ymax))
        for i, box in enumerate(boxes_history_rot):
            t = i / (len(boxes_history_rot) - 1) if len(boxes_history_rot) > 1 else 0
            box_color = interpolate_color(EVOLUTION_COLORS["start"], EVOLUTION_COLORS["end"], t)
            poly = Polygon(np.array([[box['x'][0], box['y'][0]],
                                     [box['x'][1], box['y'][0]],
                                     [box['x'][1], box['y'][1]],
                                     [box['x'][0], box['y'][1]]]),
                           closed=True, fill=False, edgecolor=box_color,
                           linewidth=2, linestyle="--", zorder=4,
                           label="Cut Outline" if i == 0 else None)
            ax.add_patch(poly)
    generic_plot(title, "PCA 1", "PCA 2", note, save_path, draw, grid=False)


def plot_original_with_boxes(
    original_samples: np.ndarray,
    y: np.ndarray,
    boxes_history_rot: List[dict],
    v: np.ndarray,
    mu: np.ndarray,
    quadrilateral: np.ndarray,
    title: str = "PCA-PRIM Evolution in Original Space",
    note: str = "",
    save_path: Optional[str] = None
) -> None:
    """
    Plot original data with evolution boxes projected back from PCA space to original coordinates.
    """
    def draw(ax: Axes) -> None:
        plot_base(ax, original_samples, y, quadrilateral, quadrilateral_label="Quadrilateral",
                  xlim=(0, 1), ylim=(0, 1))
        for i, box_rot in enumerate(boxes_history_rot):
            t = i / (len(boxes_history_rot) - 1) if len(boxes_history_rot) > 1 else 0
            box_color = interpolate_color(EVOLUTION_COLORS["start"], EVOLUTION_COLORS["end"], t)
            corners_orig = transform_box_to_original(box_rot, v, mu)
            poly = Polygon(corners_orig, closed=True, fill=False, edgecolor=box_color, linewidth=2,
                           linestyle="--", zorder=4, label="Cut Outline" if i == 0 else None)
            ax.add_patch(poly)
    generic_plot(title, "X-axis", "Y-axis", note, save_path, draw, grid=False)


def plot_overlayed_peeling_trajectories(
    prim_cov: List[float],
    prim_dens: List[float],
    pcaprim_cov: List[float],
    pcaprim_dens: List[float],
    title: str = "PRIM vs PCA-PRIM Peeling Trajectories",
    save_path: Optional[str] = None
) -> None:
    """
    Overlay two peeling trajectories (PRIM and PCA-PRIM) in coverage-density space for comparison.
    """
    def draw(ax: Axes) -> None:
        ax.plot(prim_cov, prim_dens, linestyle="-", color=EVOLUTION_COLORS["start"], marker="o", markersize=6,
                label="PRIM Trajectory", zorder=3)
        ax.plot(pcaprim_cov, pcaprim_dens, linestyle="-", color=EVOLUTION_COLORS["end"], marker="s", markersize=6,
                label="PCA-PRIM Trajectory", zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    generic_plot(title, "Coverage", "Density", "", save_path, draw)


def plot_peeling_trajectory(
    coverage_vals: List[float],
    density_vals: List[float],
    start_color: str = EVOLUTION_COLORS["start"],
    end_color: str = EVOLUTION_COLORS["end"],
    title: str = "Peeling Trajectory",
    note: str = "",
    save_path: Optional[str] = None
) -> None:
    """
    Plot a peeling trajectory in coverage-density space with a gradient color scheme over steps.
    """
    def draw(ax: Axes) -> None:
        n = len(coverage_vals)
        ax.plot(coverage_vals, density_vals, linestyle="-", color="gray", linewidth=1.5, alpha=0.8, zorder=1)
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0
            point_color = interpolate_color(start_color, end_color, t)
            ax.scatter(coverage_vals[i], density_vals[i], color=point_color, s=60, zorder=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    generic_plot(title, "Coverage", "Density", note, save_path, draw)


def plot_peeling_trajectory_with_constraint_colors(
    coverage_vals: List[float],
    density_vals: List[float],
    boxes_history: List[dict],
    orig_bounds: dict,
    title: str = "Peeling Trajectory",
    save_path: Optional[str] = None
) -> None:
    """
    Plot a peeling trajectory using color to indicate which dimensions were constrained
    in each step.

    - Dark green: both x and y dimensions were constrained
    - Light green: only one of the two dimensions was constrained
    - Grey: no constraint was applied (box still spans the full input space)

    Parameters:
        coverage_vals (list of float): Coverage values per iteration.
        density_vals (list of float): Density values per iteration.
        boxes_history (list of dict): List of box bounds at each iteration.
        orig_bounds (dict): Original bounds of the space, to check constraints.
        title (str): Title for the plot.
        save_path (Optional[str]): Where to save the resulting figure.
    """
    def draw(ax: Axes) -> None:
        n = len(coverage_vals)
        for i in range(n):
            box = boxes_history[i]
            x_constrained = (not np.isclose(box['x'][0], orig_bounds['x'][0], atol=1e-6) or
                             not np.isclose(box['x'][1], orig_bounds['x'][1], atol=1e-6))
            y_constrained = (not np.isclose(box['y'][0], orig_bounds['y'][0], atol=1e-6) or
                             not np.isclose(box['y'][1], orig_bounds['y'][1], atol=1e-6))
            if x_constrained and y_constrained:
                color = SCATTER_COLORS["interest"]
            elif x_constrained or y_constrained:
                color = lighten_color(SCATTER_COLORS["interest"], amount=0.5)
            else:
                color = QUADRILATERAL_COLOR
            ax.scatter(coverage_vals[i], density_vals[i], color=color, s=60, zorder=3)

        ax.plot(coverage_vals, density_vals, linestyle="-", color="gray", linewidth=1.5, alpha=0.8, zorder=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Density")

        dummy_both = ax.scatter([], [], color=SCATTER_COLORS["interest"], s=60, label="Both Dimensions Constrained")
        dummy_one = ax.scatter([], [], color=lighten_color(SCATTER_COLORS["interest"], amount=0.5), s=60,
                               label="One Dimension Constrained")
        dummy_none = ax.scatter([], [], color=QUADRILATERAL_COLOR, s=60, label="No Constraint")
        ax.legend(handles=[dummy_both, dummy_one, dummy_none], loc="lower right")

    generic_plot(title, "Coverage", "Density", "", save_path, draw, save_figsize=(4.7, 3.7))
