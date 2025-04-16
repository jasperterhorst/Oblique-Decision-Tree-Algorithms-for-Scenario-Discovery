"""
Module for plotting decision boundaries, printing tree structures, and coverage and density trade-offs.

Provides functions to visualize the structure of decision trees.
"""


import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.colors import ListedColormap
from shapely.geometry import LineString, box
from shapely.ops import split

from C_oblique_decision_trees.core.tree import DecisionNode
from C_oblique_decision_trees.evaluation.io_utils import get_output_dir
from src.config.colors import PRIMARY_LIGHT, SECONDARY_LIGHT
from src.config.plot_settings import beautify_plot, beautify_subplot


def create_initial_polygon(x_min, x_max, y_min, y_max):
    return box(x_min, y_min, x_max, y_max)


def cut_polygon_with_line(polygon, w, b, side):
    # Create the line representing the hyperplane: w^T x + b = 0
    x_min, y_min, x_max, y_max = polygon.bounds
    if not np.isclose(w[1], 0.0):
        x_vals = np.array([x_min - 1, x_max + 1])
        y_vals = -(w[0] * x_vals + b) / w[1]
        line = LineString([(x_vals[0], y_vals[0]), (x_vals[1], y_vals[1])])
    else:
        x_split = -b / w[0]
        y_vals = np.array([y_min - 1, y_max + 1])
        line = LineString([(x_split, y_vals[0]), (x_split, y_vals[1])])

    try:
        pieces = split(polygon, line)
        if len(pieces.geoms) != 2:
            return None
        if side == '<':
            return min(pieces.geoms, key=lambda p: np.dot(p.centroid.coords[0], w) + b)
        else:
            return max(pieces.geoms, key=lambda p: np.dot(p.centroid.coords[0], w) + b)
    except Exception as e:
        print(f"[!] Split error: {e}")
        return None


def construct_region_from_constraints(constraints, initial_region):
    region = initial_region
    for w, b, side in constraints:
        region = cut_polygon_with_line(region, w, b, side)
        if region is None or region.is_empty:
            return None
    return region


def plot_oblique_splits_clipped(X, y, trees_dict, max_depth, save_name=None, output_subfolder=None):
    import matplotlib as mpl
    import os
    from C_oblique_decision_trees.evaluation.io_utils import get_output_dir

    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']

    model_key = next(iter(trees_dict))
    available_depths = sorted(trees_dict[model_key].keys())
    n_plots = len(available_depths)

    n_cols = 3
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten()

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    bounds = (x_min, x_max, y_min, y_max)

    scatter_colors = np.where(y == 0, SECONDARY_LIGHT, PRIMARY_LIGHT)

    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], c=scatter_colors, s=5)
    beautify_subplot(ax, title="Sampled Data Points", xlabel="Feature 1", ylabel="Feature 2")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    def plot_split_recursive(node, constraints, ax, depth):
        if not isinstance(node, DecisionNode):
            return

        w, b = node.weights, node.bias
        region = construct_region_from_constraints(constraints, create_initial_polygon(*bounds))
        if region is None or not region.is_valid or region.area < 1e-8:
            print(f"[!] Invalid region at depth {depth}")
            return

        # Plot the clipped line
        if not np.isclose(w[1], 0.0):
            x_vals = np.linspace(x_min - 2, x_max + 2, 2)
            y_vals = -(w[0] * x_vals + b) / w[1]
            line_seg = LineString([(x_vals[0], y_vals[0]), (x_vals[1], y_vals[1])])
        else:
            x_split = -b / w[0]
            y_vals = np.linspace(y_min - 2, y_max + 2, 2)
            line_seg = LineString([(x_split, y_vals[0]), (x_split, y_vals[1])])

        clipped = region.intersection(line_seg)
        if not clipped.is_empty and hasattr(clipped, "xy"):
            x, y = clipped.xy
            ax.plot(x, y, 'k--', linewidth=1)
        else:
            print(f"[!] No clipped segment at depth {depth}")

        # Recurse with constraints
        if len(node.children) > 0:
            left_constraints = constraints + [(w, b, '<')]
            right_constraints = constraints + [(w, b, '>=')]
            plot_split_recursive(node.children[0], left_constraints, ax, depth + 1)
            if len(node.children) > 1:
                plot_split_recursive(node.children[1], right_constraints, ax, depth + 1)

    for depth in range(1, max_depth + 1):
        model_trees = trees_dict.get(model_key, {})
        pruned_tree = model_trees.get(depth)
        if pruned_tree is None:
            print(f"Depth {depth} missing in trees_dict.")
            continue

        ax = axes[depth]
        ax.scatter(X[:, 0], X[:, 1], c=scatter_colors, s=5, alpha=0.6)

        initial_constraints = []
        plot_split_recursive(pruned_tree.root, initial_constraints, ax, 0)

        beautify_subplot(ax, title=f"Split Regions (Tree Depth = {depth})", xlabel="Feature 1", ylabel="Feature 2")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    for j in range(n_plots, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save_name:
        output_dir = get_output_dir("single", output_subfolder)
        save_path = os.path.join(output_dir, save_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[\u2713] Saved clipped split figure: {save_path}")

    plt.show()
    plt.close(fig)


def plot_coverage_density_for_shape(df, algorithm=None, dataset=None, seed=None, print_points=False,
                                    save_name=None, output_subfolder=None):
    """
    Plot coverage vs. density trade-off for one shape and one algorithm.
    Each point is colored by tree depth and connected by a line.
    """
    algorithms = df["algorithm"].unique()
    datasets = df["dataset"].unique()
    seeds = df["seed"].unique()

    if algorithm is None:
        if len(algorithms) > 1:
            raise ValueError(f"Multiple algorithms found: {algorithms}. Please specify `algorithm`.")
        algorithm = algorithms[0]

    if dataset is None:
        if len(datasets) > 1:
            raise ValueError(f"Multiple datasets found: {datasets}. Please specify `dataset`.")
        dataset = datasets[0]

    if seed is None:
        if len(seeds) > 1:
            raise ValueError(f"Multiple seeds found: {seeds}. Please specify `seed`.")
        seed = seeds[0]

    filtered = df[
        (df["algorithm"] == algorithm) &
        (df["dataset"] == dataset) &
        (df["seed"] == seed)
    ].copy()

    if filtered.empty:
        print(f"No data found for shape: {dataset} with algorithm: {algorithm} and seed: {seed}")
        return

    filtered.sort_values("depth", inplace=True)
    cov = filtered["gini_coverage_all_leaves"].values
    dens = filtered["gini_density_all_leaves"].values
    depths = filtered["depth"].values

    if print_points:
        print(f"\n=== {dataset.upper()} ===")
        for d, c, den in zip(depths, cov, dens):
            print(f"Depth {d}: Coverage = {c:.3f}, Density = {den:.3f}")

    max_depth = depths.max()
    cmap = cm.get_cmap("viridis", max_depth + 1)
    norm = mcolors.Normalize(vmin=0, vmax=max_depth)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dens, cov, color="gray", linestyle="-", linewidth=1, zorder=1)

    for d, x, y in zip(depths, dens, cov):
        color = cmap(norm(d))
        ax.scatter(x, y, color=color, s=40, zorder=2)
        # ax.text(x + 0.01, y + 0.01, f"{d}", fontsize=7, alpha=0.6)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Tree Depth", fontsize=13)

    save_path = None
    if save_name:
        output_dir = get_output_dir("single", output_subfolder)
        save_path = os.path.join(output_dir, save_name)

    beautify_plot(
        ax,
        title=f"Coverage vs Density Trade-Off\n{algorithm.upper()}, {dataset.replace('_', ' ').title()}",
        xlabel="Density",
        ylabel="Coverage",
        save_path=save_path
    )


def plot_decision_regions_from_dict(X, y, trees_dict, max_depth, save_name=None, output_subfolder=None):
    """
    Plots a multi-panel figure showing the decision boundaries at each tree depth
    using the pruned trees from the trees_dict.
    """
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']

    custom_cmap = ListedColormap([SECONDARY_LIGHT, PRIMARY_LIGHT])

    model_key = next(iter(trees_dict))
    available_depths = sorted(trees_dict[model_key].keys())
    n_plots = len(available_depths)

    n_cols = 3
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten()

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    ax = axes[0]
    scatter_colors = np.where(y == 0, SECONDARY_LIGHT, PRIMARY_LIGHT)
    ax.scatter(X[:, 0], X[:, 1], c=scatter_colors, s=5)
    beautify_subplot(ax, title="Sampled Data Points", xlabel="Feature 1", ylabel="Feature 2")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    for depth in range(1, max_depth + 1):
        model_key = next(iter(trees_dict))
        model_trees = trees_dict.get(model_key, {})
        pruned_tree = model_trees.get(depth)

        if pruned_tree is None:
            print(f"Depth {depth} missing in trees_dict.")
            continue

        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = np.array([pruned_tree.predict(point) for point in grid_points]).reshape(xx.shape)

        ax = axes[depth]
        ax.pcolormesh(xx, yy, Z, cmap=custom_cmap, alpha=0.5, shading='auto')
        beautify_subplot(ax, title=f"Decision Boundary (Tree Depth = {depth})", xlabel="Feature 1", ylabel="Feature 2")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    for j in range(n_plots, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save_name:
        output_dir = get_output_dir("single", output_subfolder)
        save_path = os.path.join(output_dir, save_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[\u2713] Saved decision region figure: {save_path}")

    plt.show()
    plt.close(fig)


def print_tree_structure(tree):
    """Traverse and print the tree structure."""
    tree.traverse(lambda node: print(repr(node)))


def plot_decision_boundaries(tree, X, y, ax=None, show_data=True, xlim=(-1, 1), ylim=(-1, 1)):
    """
    Plot decision boundaries of a decision tree on 2D data.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    if show_data:
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=20, alpha=0.6)

    x_vals = np.linspace(xlim[0], xlim[1], 500)

    def plot_split(node):
        if isinstance(node, DecisionNode):
            w = node.weights
            b = node.bias
            if len(w) != 2:
                return
            if np.isclose(w[1], 0.0):
                x_line = -b / w[0]
                ax.axvline(x_line, color='k', linestyle='--', linewidth=1)
            else:
                y_vals = -(w[0] * x_vals + b) / w[1]
                ax.plot(x_vals, y_vals, 'k--', linewidth=1)
            for child in node.children:
                plot_split(child)

    plot_split(tree.root)
    ax.set_title("Oblique Decision Boundaries")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    plt.tight_layout()
    return ax


def plot_oblique_splits_from_dict(X, y, trees_dict, max_depth, save_name=None, output_subfolder=None):
    """
    Plots a multi-panel figure showing oblique decision boundaries (splits)
    from pruned trees in the trees_dict at each tree depth.
    """
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']

    model_key = next(iter(trees_dict))
    available_depths = sorted(trees_dict[model_key].keys())
    n_plots = len(available_depths)

    n_cols = 3
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten()

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    scatter_colors = np.where(y == 0, SECONDARY_LIGHT, PRIMARY_LIGHT)

    # First panel: show data
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], c=scatter_colors, s=5)
    beautify_subplot(ax, title="Sampled Data Points", xlabel="Feature 1", ylabel="Feature 2")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Subsequent panels: show tree splits
    for depth in range(1, max_depth + 1):
        model_key = next(iter(trees_dict))
        model_trees = trees_dict.get(model_key, {})
        pruned_tree = model_trees.get(depth)

        if pruned_tree is None:
            print(f"Depth {depth} missing in trees_dict.")
            continue

        ax = axes[depth]
        ax.scatter(X[:, 0], X[:, 1], c=scatter_colors, s=5, alpha=0.6)

        x_vals = np.linspace(x_min, x_max, 500)

        def plot_split(node):
            if isinstance(node, DecisionNode):
                w = node.weights
                b = node.bias
                if len(w) != 2:
                    return
                if np.isclose(w[1], 0.0):
                    x_line = -b / w[0]
                    ax.axvline(x_line, color='k', linestyle='--', linewidth=1)
                else:
                    y_vals = -(w[0] * x_vals + b) / w[1]
                    ax.plot(x_vals, y_vals, 'k--', linewidth=1)
                for child in node.children:
                    plot_split(child)

        plot_split(pruned_tree.root)
        beautify_subplot(ax, title=f"Oblique Splits (Tree Depth = {depth})", xlabel="Feature 1", ylabel="Feature 2")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # Turn off unused axes
    for j in range(n_plots, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save_name:
        output_dir = get_output_dir("single", output_subfolder)
        save_path = os.path.join(output_dir, save_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[\u2713] Saved oblique split figure: {save_path}")

    plt.show()
    plt.close(fig)
