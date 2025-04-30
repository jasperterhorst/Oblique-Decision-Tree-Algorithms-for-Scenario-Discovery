"""
Module for plotting benchmark results.

Provides functions to visualize metrics such as accuracy vs. depth or aggregated metrics.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import numpy as np
from src.config.plot_settings import beautify_plot, beautify_subplot
from src.config.paths import DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR
from src.config.settings import DEFAULT_VARIABLE_SEEDS


def plot_benchmark_metrics(df, metric="accuracy", xlabel="Depth", ylabel=None,
                           group_by=None,
                           show_std=False, save_name=None, filter_noise=False,
                           max_depth=None, group_by_shape_type=False,
                           exclude=None, title=None):
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
        (e.g., "barbell_2d_fuzziness_000" → dataset_name = "Barbell 2D", noise_type = "Fuzziness 000").

    max_depth : int, optional
        If provided, restricts the x-axis and filters out rows with depth > max_depth.

    group_by_shape_type : bool, default=False
        If True, infers 'shape_type' ("2D", "3D", or "Other") from the dataset name suffix, and allows
        grouping lines by shape type.

    exclude : dict, optional
        Dictionary of filters to exclude from the plot. For example:
        {"algorithm": ["CART"], "shape_type": ["3D"]} excludes CART and all 3D datasets.

    title : str, optional
        Custom plot title. If not provided, a title is auto-generated from the metric and group_by settings.

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axis object of the generated plot (for further tweaking if needed).

    Notes:
    ------
    - Automatically handles label formatting: algorithms are uppercased, shape types are title-cased.
    - Produces a unified legend with bold section titles ("Algorithm", "Shape Type") and spacing between blocks.
    - Legend is placed to the right of the plot (outside the main axis).
    - Saves the figure as a PDF to DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR.
    """
    if group_by is None:
        group_by=["algorithm", "shape", "label_noise"]

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

    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    plot_df = df.copy()

    # === Dataset & Noise Parsing === #
    if filter_noise:
        split_cols = plot_df["dataset"].str.rsplit("_", n=2, expand=True)
        plot_df["dataset_name"] = split_cols[0].str.replace("_", " ", regex=False).str.strip("_").str.title()
        plot_df["noise_type"] = (
            split_cols[1].str.replace("_", " ", regex=False).str.strip("_") + " " +
            split_cols[2].str.replace("_", " ", regex=False).str.strip("_")
        ).str.title()
    else:
        plot_df["dataset_name"] = plot_df["dataset"].str.replace("_", " ", regex=False).str.title()

    # === Automatic Shape Type Assignment === #
    if group_by_shape_type:
        def infer_shape_type(name):
            suffix = name[-2:].lower()
            if suffix == "2d":
                return "2D"
            elif suffix == "3d":
                return "3D"
            else:
                return "Other"
        plot_df["shape_type"] = plot_df["dataset_name"].apply(infer_shape_type)
        if "dataset" in group_by:
            group_by = [col if col != "dataset" else "shape_type" for col in group_by]

    # === Capitalize Algorithm Labels === #
    if "algorithm" in plot_df.columns:
        plot_df["algorithm"] = plot_df["algorithm"].str.replace("_", " ").str.upper()

    # === Apply Exclusion Filters === #
    if exclude:
        for key, values in exclude.items():
            if key in plot_df.columns:
                plot_df = plot_df[~plot_df[key].isin(values)]

    # === Filter Max Depth === #
    if max_depth is not None:
        plot_df = plot_df[plot_df["depth"] <= max_depth]

    # === Grouping Adjustments === #
    effective_group_by = group_by.copy()
    if filter_noise:
        effective_group_by = [col.replace("dataset", "dataset_name") for col in group_by]

    # === Aggregate Means === #
    agg_df = plot_df.groupby(effective_group_by + ["depth"], as_index=False)[metric].mean()

    # === Plotting === #
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.lineplot(
        data=agg_df,
        x="depth",
        y=metric,
        hue=effective_group_by[0],
        style=effective_group_by[1] if len(effective_group_by) > 1 else None,
        marker="o",         # turn on markers
        alpha=0.7,         # set transparency
        ax=ax
    )

    # === Std Deviation Bands (Optional) === #
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

    # === Axis Limits === #
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

    # === Title & Save Path === #
    title = title or (f"{y_label_final} vs Depth by " + " and ".join([
        col.replace("_", " ").capitalize() for col in group_by
    ]))
    if save_name is None:
        save_name = f"{metric}_vs_depth_by_{'_'.join(group_by)}.pdf"
    save_path = os.path.join(DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR, save_name)

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
        legend.get_title().set_fontsize(13)

        for text in legend.get_texts():
            text.set_fontsize(12)

    # === Finalize === #
    beautify_plot(
        ax=ax,
        title=title,
        xlabel=xlabel,
        ylabel=y_label_final,
        save_path=save_path
    )

    return ax


def plot_runtime_per_shape_grouped_by_noise_and_algorithm(df, max_depth=8, show_std=True, exclude_algos=None):
    """
    Generates runtime vs. depth plots for each shape separately.
    Lines are grouped by algorithm (color) and noise level (line style).

    Parameters:
    -----------
    df : pd.DataFrame
        Benchmark results DataFrame.
    max_depth : int
        Maximum tree depth to display.
    show_std : bool
        Whether to include standard deviation bands.
    exclude_algos : list
        List of algorithms to exclude from plots.
    """
    exclude_algos = exclude_algos or []

    # Apply noise parsing once
    df_processed = df.copy()
    split_cols = df_processed["dataset"].str.rsplit("_", n=2, expand=True)
    df_processed["dataset_name"] = split_cols[0].str.replace("_", " ", regex=False).str.strip("_").str.title()
    df_processed["noise_type"] = (
        split_cols[1].str.replace("_", " ", regex=False).str.strip("_") + " " +
        split_cols[2].str.replace("_", " ", regex=False).str.strip("_")
    ).str.title()

    shapes = df_processed["dataset_name"].unique()

    for shape in shapes:
        df_shape = df_processed[df_processed["dataset_name"] == shape]
        plot_benchmark_metrics(
            df_shape,
            metric="runtime",
            group_by=["algorithm", "noise_type"],
            show_std=show_std,
            filter_noise=False,   # Already processed
            max_depth=max_depth,
            group_by_shape_type=False,
            exclude={"algorithm": exclude_algos},
            title=f"Runtime vs Depth\n{shape}",
            save_name=f"runtime_vs_depth_{shape.replace(' ', '_')}.pdf"
        )


def plot_metric_vs_depth_per_dataset_and_algorithm(df, metric="accuracy", title=None, xlabel="Depth", ylabel=None,
                                                   x_lim=None, y_lim=None, save_name=None):
    """
    Plot a given metric as a function of tree depth for each algorithm and dataset,
    averaging over seeds.

    Parameters:
        df (pd.DataFrame): DataFrame with benchmark results.
        metric (str): The metric to plot.
        title (str): Plot title.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis (defaults to metric name if not provided).
        x_lim (tuple): Limits for x-axis.
        y_lim (tuple): Limits for y-axis.
        save_name (str): Filename to save the plot.
    """
    if metric not in df.columns:
        print(f"Metric '{metric}' not in columns: {list(df.columns)}")
        return

    # Average over seeds
    df_agg = df.groupby(["algorithm", "dataset", "depth"], as_index=False)[metric].mean()

    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    if save_name is None:
        save_name = f"{metric}_vs_depth_per_dataset.pdf"
    save_path = os.path.join(DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR, save_name)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(data=df_agg, x="depth", y=metric, hue="algorithm", style="dataset", ax=ax)

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if title is None:
        title = f"{metric.capitalize()} Across Tree Depths by Algorithm\nand Dataset"

    legend = ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=15)

    for text in legend.get_texts():
        label = text.get_text().lower()
        if label == "algorithm":
            text.set_text("Algorithm")
            text.set_fontweight("bold")
            text.set_fontsize(15)
        elif label == "dataset":
            text.set_text("Dataset")
            text.set_fontweight("bold")
            text.set_fontsize(15)

    beautify_plot(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel or metric.capitalize(), save_path=save_path)
    return ax


def plot_metric_vs_depth_per_algorithm(df, metric="accuracy", title=None, xlabel="Depth", ylabel=None,
                                       x_lim=None, y_lim=None, save_name=None, show_bands=False):
    """
    Plot an aggregated metric (averaged over datasets) as a function of tree depth,
    with shaded bands showing variability (std deviation).

    Parameters:
        df (pd.DataFrame): DataFrame with benchmark results.
        metric (str): The metric to plot.
        title (str): Plot title.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        x_lim (tuple): x-axis limits.
        y_lim (tuple): y-axis limits.
        save_name (str): Filename to save the plot.
        show_bands (bool): Enable or disable plotting of standard deviation.
    """
    if metric not in df.columns:
        print(f"Metric '{metric}' not found in columns: {list(df.columns)}")
        return

    # Compute mean and std per algorithm-depth
    agg_mean = df.groupby(["algorithm", "depth"], as_index=False)[metric].mean()
    agg_std = df.groupby(["algorithm", "depth"], as_index=False)[metric].std().rename(columns={metric: f"{metric}_std"})
    agg_df = pd.merge(agg_mean, agg_std, on=["algorithm", "depth"])

    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    save_path = save_name or os.path.join(DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR, f"{metric}_vs_depth_mean.pdf")

    fig, ax = plt.subplots(figsize=(7, 5))

    for algo in agg_df["algorithm"].unique():
        sub = agg_df[agg_df["algorithm"] == algo]
        ax.plot(sub["depth"], sub[metric], label=algo)
        if show_bands:
            ax.fill_between(
                sub["depth"],
                sub[metric] - sub[f"{metric}_std"],
                sub[metric] + sub[f"{metric}_std"],
                alpha=0.2
            )

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)

    if title is None:
        title = f"{metric.capitalize()} Across Tree Depths by Algorithm"

    legend = ax.legend(title="Algorithm", title_fontsize=15, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=15)
    legend.get_title().set_fontweight("bold")

    beautify_plot(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel or metric.capitalize(), save_path=save_path)
    return ax


def plot_metric_vs_depth_per_shape(df, metric="accuracy", title=None, xlabel="Depth", ylabel=None,
                                   x_lim=None, y_lim=None, save_name=None, show_bands=True):
    """
    Plot a given metric as a function of tree depth, with one line per dataset (shape).

    Parameters:
        df (pd.DataFrame): DataFrame with benchmark results.
        metric (str): The metric to plot.
        title (str): Plot title.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        x_lim (tuple): Limits for x-axis.
        y_lim (tuple): Limits for y-axis.
        save_name (str): Filename to save the plot.
        show_bands (bool): Enable or disable plotting of standard deviation.
    """
    if metric not in df.columns:
        print(f"Metric '{metric}' not in columns: {list(df.columns)}")
        return

    if "dataset" not in df.columns:
        print("Column 'dataset' not found in the DataFrame.")
        return

    # Compute mean and std per dataset-depth
    agg_mean = df.groupby(["dataset", "depth"], as_index=False)[metric].mean()
    agg_std = df.groupby(["dataset", "depth"], as_index=False)[metric].std().rename(columns={metric: f"{metric}_std"})
    agg_df = pd.merge(agg_mean, agg_std, on=["dataset", "depth"])

    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    save_path = save_name or os.path.join(DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR, f"{metric}_vs_depth_per_shape.pdf")

    fig, ax = plt.subplots(figsize=(7, 5))

    for shape in agg_df["dataset"].unique():
        sub = agg_df[agg_df["dataset"] == shape]
        ax.plot(sub["depth"], sub[metric], label=shape)
        if show_bands:
            ax.fill_between(
                sub["depth"],
                sub[metric] - sub[f"{metric}_std"],
                sub[metric] + sub[f"{metric}_std"],
                alpha=0.2
            )

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)

    if title is None:
        title = f"{metric.capitalize()} Across Tree Depths by Shape"

    legend = ax.legend(title="Shape", title_fontsize=15, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=15)
    legend.get_title().set_fontweight("bold")
    legend.get_title().set_horizontalalignment("center")

    beautify_plot(ax=ax, title=title, xlabel=xlabel or "Tree Depth",
                  ylabel=ylabel or metric.capitalize(), save_path=save_path)
    return ax


def plot_seed_std_vs_depth_per_algorithm(df, metric="accuracy", title=None, xlabel="Tree Depth", ylabel=None,
                                         x_lim=None, y_lim=None, save_name=None):
    """
    Plot average standard deviation across seeds per depth, grouped by algorithm.
    Aligned with project-wide plotting structure and styling.

    Parameters:
        df (pd.DataFrame): DataFrame with benchmark results.
        metric (str): Metric to compute std deviation of across seeds.
        title (str): Plot title.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis (defaults to std of metric).
        x_lim (tuple): Limits for x-axis.
        y_lim (tuple): Limits for y-axis.
        save_name (str): Filename to save the plot.
    """
    if metric not in df.columns:
        print(f"Metric '{metric}' not found in DataFrame columns: {df.columns}")
        return

    # Compute std deviation across seeds per shape-depth-algorithm
    grouped = df.groupby(["algorithm", "dataset", "depth"])[metric].std().reset_index(name="seed_std")

    # Then average across datasets per algorithm-depth pair
    agg_std = grouped.groupby(["algorithm", "depth"], as_index=False)["seed_std"].mean()

    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    save_path = save_name or os.path.join(DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR,
                                          f"{metric}_seed_std_vs_depth_per_algorithm.pdf")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(data=agg_std, x="depth", y="seed_std", hue="algorithm", marker="o", ax=ax)

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)

    if title is None:
        title = f"Average Seed Std of {metric.capitalize()} Across Tree Depths by Algorithm"

    legend = ax.legend(title="Algorithm", title_fontsize=15, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=15)
    legend.get_title().set_fontweight("bold")

    beautify_plot(
        ax=ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel or f"Std of {metric.capitalize()}",
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
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Tree Depth", fontsize=13)

    fig.suptitle(f"Coverage–Density Tradeoff\n{algorithm.upper()}, Seed {seed}", fontsize=16)

    save_path = save_name or os.path.join(
        DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR, "plots", f"coverage_density_colormap_{algorithm}.pdf"
    )
    fig.savefig(save_path, bbox_inches="tight")
    print(f"\nSaved to: {save_path}")
