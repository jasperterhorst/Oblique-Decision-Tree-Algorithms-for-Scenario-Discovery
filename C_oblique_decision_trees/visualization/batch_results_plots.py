"""
Module for plotting benchmark results.

Provides functions to visualize metrics such as accuracy vs. depth or aggregated metrics.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import LogFormatterSciNotation
import seaborn as sns
import pandas as pd
from scipy.stats import linregress
import numpy as np
from src.config.plot_settings import (beautify_plot, beautify_subplot,
                                      ALGORITHM_COLORS, SHAPE_TYPE_LINESTYLES, NOISE_MARKERS)
from src.config.paths import DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR
from src.config.settings import DEFAULT_VARIABLE_SEEDS


def plot_benchmark_metrics(df, metric="accuracy", xlabel="Depth", ylabel=None,
                           group_by=None, show_std=False, save_name=None, filter_noise=False,
                           max_depth=None, exclude=None, title=None, title_postfix=None):
    """
    Generate a line plot showing benchmark performance (e.g., accuracy, runtime)
    across tree depths for multiple algorithms and datasets.

    Supports visual grouping, noise/dataset splitting, shape type inference,
    standard deviation bands, and advanced legend customization.

    Parameters:
    -----------
    df : pd.DataFrame
        Benchmark results. Must contain columns like 'depth', 'algorithm', 'dataset', and the target `metric`.

    metric : str, default="accuracy"
        Name of the metric to plot on the y-axis (e.g., "accuracy", "runtime", "f_score").
        Determines y-axis scaling and labeling.

    xlabel : str, default="Depth"
        Label for the x-axis. Commonly "Depth".

    ylabel : str, optional
        Custom label for the y-axis. If not set, inferred from `metric` via a label map.

    group_by : list of str, default=["algorithm", "dataset"]
        Columns to group lines by (e.g., ["algorithm", "shape_type"]). The first column controls line color,
        the second controls line style.

    show_std : bool, default=False
        If True, adds shaded bands representing standard deviation across seeds/runs for each group.

    save_name : str, optional
        Name of the output PDF file (just filename, not path). If None, auto-generated from metric and group_by.

    filter_noise : bool, default=False
        If True, splits the 'dataset' column into 'dataset_name' and 'noise_type' by removing suffixes
        (e.g., "barbell_2d_label_noise_000" → dataset_name = "Barbell 2D", noise_type = "Label Noise 000").

    max_depth : int, optional
        If provided, restricts the x-axis and filters out rows with depth > max_depth.

    exclude : dict, optional
        Dictionary of filters to exclude from the plot. For example:
        {"algorithm": ["CART"], "shape_type": ["3D"]} excludes CART and all 3D datasets.

    title : str, optional
        Custom plot title. If not provided, a title is auto-generated from the metric and group_by settings.

    title_postfix : str, optional
        Postfix to append to the title. Useful for adding extra context or details.

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axis object of the generated plot (for further tweaking if needed).
    """

    if group_by is None:
        group_by = ["algorithm", "shape", "label_noise", "data_dim", "n_samples"]

    if metric not in df.columns:
        print(f"Metric '{metric}' not found in DataFrame columns.")
        return

    LABEL_MAP = {
        "runtime": "Runtime",
        "accuracy": "Accuracy",
        "coverage": "Coverage",
        "density": "Density",
        "f_score": "F-score",
        "gini_coverage_all_leaves": "Gini Coverage",
        "gini_density_all_leaves": "Gini Density",
    }
    y_label_final = ylabel or LABEL_MAP.get(metric, metric.capitalize())

    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    plot_df = df.copy()

    # Only filter by depth if specified
    if max_depth is not None:
        plot_df = df[df["depth"] <= max_depth].copy()
    else:
        plot_df = df.copy()

    # Optional filtering based on exclude dict
    if exclude:
        for key, values in exclude.items():
            if key in plot_df.columns:
                plot_df = plot_df[~plot_df[key].isin(values)]

    # Create a shape_type column based on the shape name suffix
    def infer_shape_type(name):
        name = name.lower()
        if name.endswith("2d"):
            return "2D"
        elif name.endswith("3d"):
            return "3D"
        else:
            return "Other"

    plot_df["shape_type"] = plot_df["shape"].apply(infer_shape_type)

    # Format algorithm names nicely
    if "algorithm" in plot_df.columns:
        plot_df["algorithm"] = plot_df["algorithm"].str.replace("_", " ").str.upper()

    # Use group_by directly (do not transform dataset)
    effective_group_by = group_by.copy()

    # Aggregate
    agg_df = plot_df.groupby(effective_group_by + ["depth"], as_index=False)[metric].mean()

    # Plot
    fig, ax = plt.subplots(figsize=(8.5, 5))

    # Styling based on second group_by dimension, if present
    style_col = effective_group_by[1] if len(effective_group_by) > 1 else None
    style_order = sorted(agg_df[style_col].unique()) if style_col else None

    if style_col == "shape_type":
        dashes = [SHAPE_TYPE_LINESTYLES.get(val, "solid") for val in style_order]
        markers = False
    elif style_col == "label_noise":
        markers = [NOISE_MARKERS.get(val, "o") for val in style_order]
        dashes = False
    else:
        markers = True
        dashes = True

    sns.lineplot(
        data=agg_df,
        x="depth",
        y=metric,
        hue=effective_group_by[0],
        style=style_col,
        style_order=style_order,
        dashes=dashes,
        markers=markers,
        palette=ALGORITHM_COLORS,
        ax=ax,
        alpha=0.6
    )

    if show_std:
        std_df = plot_df.groupby(effective_group_by + ["depth"], as_index=False)[metric].std()
        std_col = f"{metric}_std"
        std_df.rename(columns={metric: std_col}, inplace=True)

        for key, subdf in std_df.groupby(effective_group_by):
            key = (key,) if isinstance(key, str) else key
            sub_mean = agg_df
            for col, val in zip(effective_group_by, key):
                sub_mean = sub_mean[sub_mean[col] == val]
            merged = pd.merge(sub_mean, subdf, on=["depth"])
            ax.fill_between(
                merged["depth"],
                merged[metric] - merged[std_col],
                merged[metric] + merged[std_col],
                alpha=0.2
            )

    if max_depth is not None:
        ax.set_xlim(0, max_depth)

    if metric in [
        "accuracy", "coverage", "density", "f_score",
        "gini_coverage_all_leaves", "gini_density_all_leaves"
    ]:
        ax.set_ylim(0, 1)
    elif metric in ["time", "runtime"]:
        y_max = agg_df[metric].max() * 1.1
        ax.set_ylim(0, y_max)

    title = title or (f"{y_label_final} vs Depth by " + " and ".join([
        col.replace("_", " ").capitalize() for col in group_by
    ]))

    group_tag = "_".join(group_by)
    postfix_str = title_postfix if title_postfix else ""
    filename = f"{metric}_vs_depth_by_{group_tag}{postfix_str}.pdf"

    save_dir = os.path.join(DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR, "plots", "plot_over_depth", metric)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, filename)

    # === Format Unified Legend === #
    legend = ax.get_legend()
    if legend:
        new_labels = []
        for text in legend.get_texts():
            label = text.get_text()
            if label.upper() == label:  # algorithms
                label_fmt = label.replace("_", " ").upper()
            elif label.lower() in ["2d", "3d", "other"]:
                label_fmt = label.upper()
            else:
                label_fmt = label.replace("_", " ").title()
            new_labels.append(label_fmt)

        for text, new in zip(legend.get_texts(), new_labels):
            text.set_text(new)

        # Title formatting
        title_text = legend.get_title().get_text()
        if title_text:
            legend.get_title().set_text(title_text.replace("_", " ").capitalize())

        legend.set_bbox_to_anchor((1.05, 0.5))
        legend.set_loc("center left")
        legend.set_frame_on(True)
        legend.get_title().set_fontsize(18)

        for text in legend.get_texts():
            text.set_fontsize(12)

    beautify_plot(
        ax=ax,
        title=title,
        xlabel=xlabel,
        ylabel=y_label_final,
        save_path=save_path
    )

    return ax


def plot_coverage_density_all_shapes_for_algorithm(df, algorithm="hhcart", coverage_col="coverage",
                                                   density_col="density", max_depth=15, seed=1,
                                                   save_name=None, print_points=False):
    """
    Plot coverage vs. density trade-off across depths for each shape, using one algorithm and one seed.
    Each point is colored by depth and connected by a line. Each subplot shows (depth, cov, dens) values.

    Parameters:
        df (pd.DataFrame): Benchmark results.
        algorithm (str): Algorithm to visualize (e.g., 'hhcart').
        coverage_col (str): Column name for coverage values.
        density_col (str): Column name for density values.
        max_depth (int): Maximum depth to color.
        seed (int): Choose which seeds from the list of DEFAULT_VARIABLE_SEEDS.
        save_name (str): Path to save the resulting figure.
        print_points (bool): Whether to print the (depth, coverage, density) points. Default is False.
    """
    seed = DEFAULT_VARIABLE_SEEDS[seed]
    all_shapes = sorted(df["dataset"].unique())
    ncols = 4
    nrows = (len(all_shapes) + ncols - 1) // ncols

    cmap = cm.get_cmap("viridis", max_depth + 1)
    norm = mcolors.Normalize(vmin=0, vmax=max_depth)

    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols + 1, 4 * nrows))
    axes = axes.flatten()

    for i, shape in enumerate(all_shapes):
        ax = axes[i]
        filtered = df[
            (df["algorithm"] == algorithm) &
            (df["dataset"] == shape) &
            (df["seed"] == seed) &
            df[coverage_col].apply(np.isfinite) &
            df[density_col].apply(np.isfinite)
        ].copy()

        if filtered.empty:
            ax.set_title(f"{shape}\n(No Data)")
            ax.axis("off")
            continue

        filtered.sort_values("depth", inplace=True)
        cov = filtered[coverage_col].values
        dens = filtered[density_col].values
        depths = filtered["depth"].values

        if print_points:
            print(f"\n=== {shape.upper()} ===")
            for d, c, den in zip(depths, cov, dens):
                print(f"Depth {d}: Coverage = {c:.3f}, Density = {den:.3f}")

        # === LINE and SCATTER === #
        ax.plot(dens, cov, color="gray", linestyle="-", linewidth=1, zorder=1)

        for d, x, y in zip(depths, dens, cov):
            color = cmap(norm(d))
            ax.scatter(x, y, color=color, edgecolor="black", s=40, zorder=2)
            ax.text(x + 0.01, y + 0.01, f"{d}", fontsize=7, alpha=0.6)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        beautify_subplot(
            ax,
            title=shape.replace("_", " ").title(),
            xlabel="Density",
            ylabel="Coverage"
        )

    # Remove unused subplots
    for j in range(len(all_shapes), len(axes)):
        axes[j].axis("off")

    # Add colorbar on right
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.subplots_adjust(right=0.92, hspace=0.3)
    cbar_ax = fig.add_axes((0.94, 0.15, 0.015, 0.7))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Tree Depth", fontsize=13)

    fig.suptitle(f"Coverage–Density Tradeoff\n{algorithm.upper()}, Seed {seed}", fontsize=16)

    save_path = save_name or os.path.join(
        DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR, "plots", f"coverage_density_colormap_{algorithm}.pdf"
    )
    fig.savefig(save_path, bbox_inches="tight")
    print(f"\nSaved to: {save_path}")


def plot_runtime_over_depth(df, algorithm, vary_by="data_dim", title=None,
                            save_name="runtime_over_depth_plot.pdf", y_bounds=None):
    """
    Plot average runtime over depth for a single algorithm, varying by dimension or sample size.

    Parameters:
    -----------
    df : pd.DataFrame
        Must contain 'depth', 'runtime', 'algorithm', and `vary_by` columns.

    algorithm : str
        The algorithm to plot (e.g., "OC1", "HHCART D").

    vary_by : str, default="data_dim"
        Column to vary (either 'data_dim' or 'n_samples').

    title : str, optional
        Custom title for the plot.

    save_name : str, default="runtime_over_depth_plot.pdf"
        Output filename for the saved plot.

    y_bounds : tuple(float, float), optional
        If provided, sets manual y-axis limits.

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The Matplotlib axis object.
    """
    # === Preprocessing ===
    df = df.copy()
    df["algorithm"] = df["algorithm"].str.replace("_", " ").str.upper()
    algorithm = algorithm.replace("_", " ").upper()
    df = df[df["algorithm"] == algorithm]

    if vary_by not in df.columns:
        raise ValueError(f"Column '{vary_by}' not found in dataframe.")

    if df.empty:
        print(f"No data found for algorithm: {algorithm}")
        return

    values_sorted = sorted(df[vary_by].unique())
    alpha_map = {
        v: 0.3 + 0.7 * ((len(values_sorted) - 1 - i) / (len(values_sorted) - 1 or 1))
        for i, v in enumerate(values_sorted)
    }

    # === Plotting ===
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    fig, ax = plt.subplots(figsize=(8.5, 5))

    for v in values_sorted:
        sub = df[df[vary_by] == v]
        if sub.empty:
            continue

        mean_runtime = sub.groupby("depth")["runtime"].mean()
        std_runtime = sub.groupby("depth")["runtime"].std()

        label = f"{v}D" if vary_by == "data_dim" else f"{v:,} samples"

        ax.plot(
            mean_runtime.index,
            mean_runtime.values,
            label=label,
            color=ALGORITHM_COLORS.get(algorithm, "gray"),
            alpha=alpha_map[v],
            linewidth=2,
        )

        ax.fill_between(
            mean_runtime.index,
            mean_runtime - std_runtime,
            mean_runtime + std_runtime,
            color=ALGORITHM_COLORS.get(algorithm, "gray"),
            alpha=alpha_map[v] * 0.4,
            linewidth=0,
        )

    ax.set_xlim(0, df["depth"].max())
    if y_bounds:
        ax.set_ylim(*y_bounds)
    else:
        ax.set_ylim(0, df["runtime"].max() * 1.1)

    legend_title = "Feature Count" if vary_by == "data_dim" else "Sample Size"
    legend = ax.legend(title=legend_title, loc="center left", bbox_to_anchor=(1.05, 0.5))
    legend.get_title().set_fontsize(18)
    for text in legend.get_texts():
        text.set_fontsize(12)

    # === Save logic ===
    save_path = os.path.join(
        DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR,
        "plots",
        f"runtime_by_{vary_by}",
        save_name
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plot_title = title or f"Runtime over Depth for {algorithm}"
    plot_title = plot_title.replace("_", " ")

    beautify_plot(
        ax=ax,
        title=plot_title,
        xlabel="Depth",
        ylabel="Runtime (s)",
        save_path=save_path
    )

    return ax


def plot_scaling_loglog(df, vary_by="data_dim",
                        algorithms=("OC1", "HHCART D"),
                        title=None,
                        save_name="combined_scaling_loglog.pdf",
                        fixed_depth=None):
    """
    Create a combined log-log plot showing runtime scaling for given algorithms.

    Parameters:
    -----------
    df : pd.DataFrame
        Input data with columns: 'algorithm', 'depth', 'runtime', and vary_by.

    vary_by : str, default="data_dim"
        Column to vary on the x-axis (e.g., "data_dim" or "n_samples").

    algorithms : tuple of str, default=("OC1", "HHCART D")
        Algorithms to compare.

    title : str, optional
        Plot title.

    save_name : str
        Filename to save the plot.

    fixed_depth : int, optional
        If specified, uses this exact depth for runtime comparison.
        If None, uses the maximum depth per vary_by group.

    Returns:
    --------
    ax : matplotlib.axes.Axes
    """
    df = df.copy()
    df["algorithm"] = df["algorithm"].str.replace("_", " ").str.upper()
    vary_label = "Feature Count" if vary_by == "data_dim" else "Sample Count"

    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    fig, ax = plt.subplots(figsize=(8.5, 5))

    for algo in algorithms:
        algo_df = df[df["algorithm"] == algo.upper()]
        if algo_df.empty:
            continue

        if fixed_depth is not None:
            filtered_df = algo_df[algo_df["depth"] == fixed_depth]
        else:
            max_depths = algo_df.groupby(vary_by)["depth"].max().reset_index()
            filtered_df = pd.merge(algo_df, max_depths, on=[vary_by, "depth"])

        grouped = filtered_df.groupby(vary_by)["runtime"].mean().reset_index()
        x_vals = grouped[vary_by].values
        y_vals = grouped["runtime"].values

        # Linear regression in log-log space
        slope, intercept, *_ = linregress(np.log10(x_vals), np.log10(y_vals))
        y_fit = 10 ** (intercept + slope * np.log10(x_vals))

        label = f"{algo} (slope = {slope:.2f})"
        ax.plot(x_vals, y_vals, label=label,
                color=ALGORITHM_COLORS.get(algo.upper(), "gray"),
                marker="o", linewidth=2)
        ax.plot(x_vals, y_fit, linestyle="dashed",
                color=ALGORITHM_COLORS.get(algo.upper(), "gray"), alpha=0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(LogFormatterSciNotation())
    ax.yaxis.set_major_formatter(LogFormatterSciNotation())
    ax.set_xlabel(vary_label)
    ax.set_ylabel("Runtime (s)")

    plot_title = title or f"Runtime Scaling with {vary_label}"
    ax.set_title(plot_title.replace("_", " "))

    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=12, title="Algorithm")
    ax.get_legend().get_title().set_fontsize(18)

    save_path = os.path.join(
        DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR, "plots", "runtime_by_data_dim", save_name
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    beautify_plot(ax=ax, title=plot_title, xlabel=vary_label, ylabel="Runtime (s)", save_path=save_path)

    return ax


def plot_metrics_vs_depth_grouped_by_dims_or_samples(
    df, algorithm, input_axis="data_dim", metrics=("accuracy", "coverage", "density"),
    save_name=None
):
    """
    Plot average metric trajectories over depth for one algorithm, grouped by feature count or sample size.

    Parameters:
    -----------
    df : pd.DataFrame
        Input benchmark data.

    algorithm : str
        Algorithm to filter and plot.

    input_axis : str, default="data_dim"
        Either 'data_dim' or 'n_samples'.

    metrics : tuple of str
        Metrics to plot. Each is rendered as a subplot with shared x-axis (depth).

    save_name : str, optional
        Output PDF path. If None, autogenerated.

    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    # === Plot settings ===
    plt.rcParams["text.usetex"] = False
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    assert input_axis in ["data_dim", "n_samples"], "input_axis must be 'data_dim' or 'n_samples'"
    for m in metrics:
        assert m in df.columns, f"Metric '{m}' not found in dataframe."

    df = df.copy()
    df["algorithm"] = df["algorithm"].str.replace("_", " ").str.upper()
    algorithm = algorithm.replace("_", " ").upper()
    df = df[df["algorithm"] == algorithm]

    if df.empty:
        print(f"No data for algorithm {algorithm}")
        return

    values_sorted = sorted(df[input_axis].unique())

    # Get base color from ALGORITHM_COLORS, fallback to gray
    base_color = ALGORITHM_COLORS.get(algorithm, "#888888")
    cmap = cm.get_cmap("Greens" if "HHCART" in algorithm else "Reds", len(values_sorted))
    color_map = {v: cmap(i) for i, v in enumerate(values_sorted)}

    fig, axes = plt.subplots(nrows=1, ncols=len(metrics), figsize=(6 * len(metrics), 5), sharex=True)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for v in values_sorted:
            sub = df[df[input_axis] == v]
            grouped = sub.groupby("depth")[metric]
            mean = grouped.mean()

            label = f"{v}D" if input_axis == "data_dim" else f"{v:,} samples"
            color = color_map[v]

            ax.plot(
                mean.index, mean.values,
                label=label,
                color=color,
                linewidth=2,
                marker='o'
            )

        beautify_subplot(
            ax,
            xlabel="Depth",
            ylabel=metric.capitalize(),
            xlim=(0, 12),
            ylim=(0, 1)
        )

    # Add shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center left", bbox_to_anchor=(0.97, 0.5),
        title="Feature Count" if input_axis == "data_dim" else "Sample Size",
        title_fontsize=16,
        fontsize=14
    )

    fig.suptitle(
        f"{algorithm} – Performance by " + ("Feature Count" if input_axis == "data_dim" else "Sample Size"),
        fontsize=22
    )
    fig.tight_layout(rect=(0, 0, 0.965, 1))

    save_path = save_name or os.path.join(
        DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR,
        "plots",
        "depth_trajectories",
        f"{algorithm.replace(' ', '_').lower()}_metric_trajectories_by_{input_axis}.pdf"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    print(f"[SAVED] {save_path}")

    plt.close(fig)
    return fig


def plot_metrics_vs_depth_grouped_by_label_noise(
    df, algorithm, metrics=("accuracy", "coverage", "density"), save_name=None
):
    def generate_interpolated_colormap(base_color, n_levels):
        base_rgb = mcolors.to_rgb(base_color)
        light_rgb = tuple(1 - 0.4 * (1 - c) for c in base_rgb)
        dark_rgb = tuple(0.5 * c for c in base_rgb)
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_interp", [light_rgb, base_rgb, dark_rgb], N=n_levels)
        return [cmap(i / (n_levels - 1)) for i in range(n_levels)]

    # === Setup ===
    plt.rcParams["text.usetex"] = False
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    df = df.copy()
    for m in metrics:
        assert m in df.columns, f"Metric '{m}' not found in dataframe."

    df["algorithm"] = df["algorithm"].str.replace("_", " ").str.upper()
    algorithm = algorithm.replace("_", " ").upper()
    df = df[df["algorithm"] == algorithm]
    if df.empty:
        print(f"No data for algorithm {algorithm}")
        return

    df["label_noise"] = df["label_noise"].astype(str)
    values_sorted = sorted(df["label_noise"].unique())
    base_color = ALGORITHM_COLORS.get(algorithm, "#888888")
    colors = generate_interpolated_colormap(base_color, len(values_sorted))
    color_map = dict(zip(values_sorted, colors))

    fig, axes = plt.subplots(nrows=1, ncols=len(metrics), figsize=(6 * len(metrics), 5), sharex=True)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for v in values_sorted:
            sub = df[df["label_noise"] == v]
            grouped = sub.groupby("depth")[metric]
            mean = grouped.mean()
            label = f"Noise {v}"
            ax.plot(
                mean.index, mean.values,
                label=label,
                color=color_map[v],
                linewidth=2,
                marker='o'
            )

        beautify_subplot(
            ax,
            xlabel="Depth",
            ylabel=metric.capitalize(),
            xlim=(0, 12),
            ylim=(0, 1)
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center left", bbox_to_anchor=(0.97, 0.5),
        title="Label Noise",
        title_fontsize=16,
        fontsize=14
    )

    fig.suptitle(f"{algorithm} – Performance by Label Noise Level", fontsize=22)
    fig.tight_layout(rect=(0, 0, 0.965, 1))

    # === Save ===
    folder = os.path.join(DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR, "plots", "performance_by_label_noise")
    os.makedirs(folder, exist_ok=True)
    if save_name is None or not os.path.dirname(save_name):
        filename = save_name or f"{algorithm.replace(' ', '_').lower()}_metric_trajectories_by_label_noise.pdf"
        save_path = os.path.join(folder, filename)
    else:
        save_path = save_name

    fig.savefig(save_path, bbox_inches="tight")
    print(f"[SAVED] {save_path}")
    plt.close(fig)
    return fig
