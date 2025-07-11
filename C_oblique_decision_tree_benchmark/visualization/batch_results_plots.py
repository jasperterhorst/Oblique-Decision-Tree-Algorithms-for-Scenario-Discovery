"""
Module for plotting benchmark results.

Provides functions to visualize metrics such as accuracy vs. depth or aggregated metrics.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import LogFormatterSciNotation
import numpy as np
import seaborn as sns
from scipy.stats import linregress
from typing import Optional, Union, List, Tuple

from src.config import (
    apply_global_plot_settings,
    beautify_plot,
    beautify_subplot,
    save_figure,
    generate_color_gradient,
    ALGORITHM_COLORS,
    SHAPE_TYPE_LINESTYLES,
    NOISE_MARKERS,
    DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR,
    DEFAULT_VARIABLE_SEEDS
)

# Figure size constants
HEIGHT = 3.7
WIDTH = 4.7
FIGSIZE_STANDARD = (WIDTH, HEIGHT)


def figsize_wide(n):
    return min(WIDTH * n, 18.8), HEIGHT


def figsize_grid(rows, cols):
    return min(WIDTH * cols, 18.8), HEIGHT * rows


def plot_metric_over_depth_by_algorithm_and_group(
    df: pd.DataFrame,
    metric: Union[str, List[str]] = "accuracy",
    xlabel: str = "Depth",
    ylabel: Optional[str] = None,
    group_by: Optional[List[str]] = None,
    show_std: bool = False,
    save_name: Optional[str] = None,
    max_depth: Optional[int] = None,
    exclude: Optional[dict] = None,
    title: Optional[str] = None,
    title_postfix: Optional[str] = None
) -> plt.Figure:
    apply_global_plot_settings()

    if group_by is None:
        group_by = ["algorithm", "shape", "label_noise", "data_dim", "n_samples"]

    metrics = [metric] if isinstance(metric, str) else metric
    LABEL_MAP = {
        "runtime": "Runtime (s)", "accuracy": "Accuracy",
        "coverage": "Coverage", "density": "Density",
        "f_score": "F-score", "gini_coverage_all_leaves": "Gini Coverage",
        "gini_density_all_leaves": "Gini Density"
    }

    plot_df = df.copy()
    if max_depth is not None:
        plot_df = plot_df[plot_df["depth"] <= max_depth].copy()

    if exclude:
        for key, values in exclude.items():
            if key in plot_df.columns:
                plot_df = plot_df[~plot_df[key].isin(values)]

    def infer_shape_type(name):
        name = name.lower()
        if name.endswith("2d"):
            return "2D"
        elif name.endswith("3d"):
            return "3D"
        else:
            return "Other"

    plot_df["shape_type"] = plot_df["shape"].apply(infer_shape_type)
    if "algorithm" in plot_df.columns:
        plot_df["algorithm"] = plot_df["algorithm"].str.replace("_", " ").str.upper()

    effective_group_by = group_by.copy()
    num_metrics = len(metrics)

    if num_metrics == 4:
        nrows, ncols = 2, 2
        figsize = figsize_grid(nrows, ncols)
    elif num_metrics > 1:
        nrows, ncols = 1, num_metrics
        figsize = figsize_wide(ncols)
    else:
        nrows, ncols = 1, 1
        figsize = FIGSIZE_STANDARD

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        y_label = ylabel or LABEL_MAP.get(metric, metric.capitalize())
        agg_df = plot_df.groupby(effective_group_by + ["depth"], as_index=False)[metric].mean()

        style_col = effective_group_by[1] if len(effective_group_by) > 1 else None
        style_order = sorted(agg_df[style_col].unique()) if style_col else None

        dashes, markers = True, True
        if style_col == "shape_type":
            dashes = [SHAPE_TYPE_LINESTYLES.get(val, "solid") for val in style_order]
            markers = False
        elif style_col == "label_noise":
            markers = [NOISE_MARKERS.get(val, "o") for val in style_order]
            dashes = False

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
            alpha=0.85
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

        if ax.get_legend():
            ax.legend_.remove()

        y_vals = agg_df[metric].dropna()
        ax.set_ylim(0, 1 if y_vals.max() <= 1 else y_vals.max() * 1.1)
        ax.set_xlim(0, plot_df["depth"].max())
        beautify_subplot(ax, xlabel=xlabel, ylabel=y_label)

    # Unified legend
    # Unified grouped legend per group_by dimension (e.g., algorithm, shape_type)
    from matplotlib.lines import Line2D
    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    legend = None

    grouped_handles = {}
    for h, l in zip(legend_handles, legend_labels):
        if l is None or l.strip() == "":
            continue
        for col in effective_group_by:
            col_name = col.replace("_", " ").title()
            if col_name not in grouped_handles:
                grouped_handles[col_name] = []
            # naive inclusion: we keep all labels grouped by the col names that are styled
            # we filter smarter below
            grouped_handles[col_name].append((h, l))

    # Smart filtering: keep only legend items that match unique values in the plot
    final_handles, final_labels = [], []
    for i, col in enumerate(effective_group_by):
        col_name = col.replace("_", " ").title()
        unique_values = sorted(plot_df[col].dropna().unique())
        if not unique_values:
            continue

        # Add a fake handle as section title
        final_handles.append(Line2D([], [], linestyle='none', color='none', label=col_name))
        final_labels.append(col_name)

        seen = set()
        for h, l in grouped_handles.get(col_name, []):
            if l not in seen and l in unique_values or l.upper() in unique_values or l.title() in unique_values:
                final_handles.append(h)
                final_labels.append(l)
                seen.add(l)

    if final_handles:
        legend = fig.legend(
            final_handles, final_labels,
            loc="center left", bbox_to_anchor=(0.97, 0.5),
            fontsize=14, frameon=True
        )
        # Bold the section headers
        for i, text in enumerate(legend.get_texts()):
            if final_labels[i] in [g.replace("_", " ").title() for g in effective_group_by]:
                text.set_fontweight("bold")

    sup_title = title or f"{' – '.join(metrics).title()} vs Depth by {', '.join(group_by)}"
    fig.suptitle(sup_title, fontsize=22)
    fig.tight_layout(rect=(0, 0, 0.965, 1))

    # Save
    save_dir = os.path.join(DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR, "plots", "plot_over_depth")
    os.makedirs(save_dir, exist_ok=True)
    filename = save_name or f"{'_'.join(metrics)}_vs_depth_by_{'_'.join(group_by)}{title_postfix or ''}.pdf"
    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, bbox_inches="tight")
    print(f"[SAVED] {save_path}")
    plt.close(fig)
    return fig


def plot_runtime_over_depth_grouped_by_data_dim_or_samples(
    df: pd.DataFrame,
    algorithm: str,
    vary_by: str = "data_dim",
    title: Optional[str] = None,
    save_name: str = "runtime_over_depth_plot.pdf",
    y_bounds: Optional[Tuple[float, float]] = None,
    max_depth: Optional[int] = None,
) -> plt.Axes:
    """
    Plot average runtime over tree depth for a single algorithm, grouped by either feature count or sample size.

    Parameters:
        df (pd.DataFrame): Input data containing 'depth', 'runtime', 'algorithm', and vary_by columns.
        algorithm (str): The algorithm to plot (e.g., "MOC1", "HHCART_SD D").
        vary_by (str): Column to group lines by: either 'data_dim' or 'n_samples'.
        title (str, optional): Custom plot title.
        save_name (str, optional): Output filename (PDF).
        y_bounds (tuple of float, optional): Manual y-axis limits.
        max_depth (int, optional): Maximum depth to consider for the plot.

    Returns:
        ax (plt.Axes): The axis object with the rendered plot.
    """
    apply_global_plot_settings()

    df = df.copy()
    df["algorithm"] = df["algorithm"].str.replace("_", " ").str.upper()
    algorithm_fmt = algorithm.replace("_", " ").upper()
    df = df[df["algorithm"] == algorithm_fmt]

    if vary_by not in df.columns:
        raise ValueError(f"Column '{vary_by}' not found in dataframe.")
    if df.empty:
        raise ValueError(f"No data found for algorithm: {algorithm_fmt}")

    values_sorted = sorted(df[vary_by].unique())
    base_color = ALGORITHM_COLORS.get(algorithm_fmt, "#888888")
    color_map = dict(zip(values_sorted, generate_color_gradient(base_color, len(values_sorted))))

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    for v in values_sorted:
        sub = df[df[vary_by] == v]

        if max_depth is not None:
            sub = sub[sub["depth"] <= max_depth]

        if sub.empty:
            continue

        mean_runtime = sub.groupby("depth")["runtime"].mean()
        std_runtime = sub.groupby("depth")["runtime"].std()

        label = f"{v}D" if vary_by == "data_dim" else f"{v:,}"
        color = color_map[v]

        ax.plot(mean_runtime.index, mean_runtime.values, label=label, color=color, linewidth=2)
        ax.fill_between(
            mean_runtime.index,
            mean_runtime - std_runtime,
            mean_runtime + std_runtime,
            color=color,
            alpha=0.3,
            linewidth=0,
        )

    depth_limit = max_depth if max_depth is not None else df["depth"].max()
    ax.set_xlim(0, depth_limit)
    if y_bounds:
        ax.set_ylim(*y_bounds)
    else:
        ax.set_ylim(0, df["runtime"].max() * 1.1)

    legend_title = "Feature Count" if vary_by == "data_dim" else "Sample Size"
    legend = ax.legend(
        title=legend_title,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        fontsize=12,
        title_fontsize=16,
        frameon=True
    )

    plot_title = title or f"Runtime over Depth for {algorithm_fmt}"
    beautify_plot(ax, title=plot_title, xlabel="Depth", ylabel="Runtime (s)")

    # ax.set_title(title, fontsize=19, pad=20, wrap=True)

    save_figure(
        fig,
        subfolder="depth_sweep_batch_results/plots",
        subsubfolder=f"runtime_by_{vary_by}",
        filename=save_name.replace(".pdf", "")
    )

    return ax


def plot_active_features_over_depth_grouped_by_data_dim_or_samples(
    df: pd.DataFrame,
    algorithm: str,
    vary_by: str = "data_dim",
    title: Optional[str] = None,
    save_name: str = "active_features_over_depth_plot.pdf",
    y_bounds: Optional[Tuple[float, float]] = None,
    max_depth: Optional[int] = None,
) -> plt.Axes:
    """
    Plot average number of active features used per depth, grouped by feature count or sample size.

    Parameters:
        df (pd.DataFrame): Input data containing 'depth', 'avg_active_feature_count', 'algorithm', and vary_by columns.
        algorithm (str): The algorithm to plot (e.g., "MOC1", "HHCART_SD D").
        vary_by (str): Column to group lines by: either 'data_dim' or 'n_samples'.
        title (str, optional): Custom plot title.
        save_name (str, optional): Output filename (PDF).
        y_bounds (tuple of float, optional): Manual y-axis limits.
        max_depth (int, optional): Maximum depth to consider for the plot.

    Returns:
        ax (plt.Axes): The axis object with the rendered plot.
    """
    apply_global_plot_settings()

    df = df.copy()
    df["algorithm"] = df["algorithm"].str.replace("_", " ").str.upper()
    algorithm_fmt = algorithm.replace("_", " ").upper()
    df = df[df["algorithm"] == algorithm_fmt]

    if vary_by not in df.columns:
        raise ValueError(f"Column '{vary_by}' not found in dataframe.")
    if df.empty:
        raise ValueError(f"No data found for algorithm: {algorithm_fmt}")

    values_sorted = sorted(df[vary_by].unique())
    base_color = ALGORITHM_COLORS.get(algorithm_fmt, "#888888")
    color_map = dict(zip(values_sorted, generate_color_gradient(base_color, len(values_sorted))))

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    for v in values_sorted:
        sub = df[df[vary_by] == v]

        if max_depth is not None:
            sub = sub[sub["depth"] <= max_depth]

        if sub.empty:
            continue

        mean_features = sub.groupby("depth")["avg_active_feature_count"].mean()
        std_features = sub.groupby("depth")["avg_active_feature_count"].std()

        label = f"{v}D" if vary_by == "data_dim" else f"{v:,}"
        color = color_map[v]

        ax.plot(mean_features.index, mean_features.values, label=label, color=color, linewidth=2)
        ax.fill_between(
            mean_features.index,
            mean_features - std_features,
            mean_features + std_features,
            color=color,
            alpha=0.3,
            linewidth=0,
        )

    depth_limit = max_depth if max_depth is not None else df["depth"].max()
    ax.set_xlim(0, depth_limit)
    if y_bounds:
        ax.set_ylim(*y_bounds)
    else:
        ax.set_ylim(0, df["avg_active_feature_count"].max() * 1.1)

    legend_title = "Feature Count" if vary_by == "data_dim" else "Sample Size"
    ax.legend(
        title=legend_title,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        fontsize=12,
        title_fontsize=16,
        frameon=True
    )

    plot_title = title or f"Active Features over Depth for {algorithm_fmt}"
    beautify_plot(ax, title=plot_title, xlabel="Depth", ylabel="Avg. Active Features")

    save_figure(
        fig,
        subfolder="depth_sweep_batch_results/plots",
        subsubfolder=f"active_features_by_{vary_by}",
        filename=save_name.replace(".pdf", "")
    )

    return ax


def plot_loglog_runtime_scaling_by_dimension_or_sample_count(
    df: pd.DataFrame,
    vary_by: str = "data_dim",
    algorithms: Tuple[str, ...] = ("MOC1", "HHCART_SD D"),
    title: Optional[str] = None,
    save_name: str = "combined_scaling_loglog.pdf",
    max_depth: Optional[int] = None,
) -> plt.Axes:
    """
    Create a log-log plot showing runtime scaling across data size or feature dimensionality.

    Parameters:
    -----------
    df : pd.DataFrame
        Input benchmark data. Must contain 'algorithm', 'depth', 'runtime', and `vary_by` columns.

    vary_by : str, default="data_dim"
        Column on the x-axis. Should be either "data_dim" or "n_samples".

    algorithms : tuple[str, ...], default=("MOC1", "HHCART_SD D")
        Algorithm names to compare. Names are case-insensitive and matched after replacing underscores with spaces.

    title : str, optional
        Custom plot title. If None, generated automatically from `vary_by`.

    save_name : str, default="combined_scaling_loglog.pdf"
        Name of the PDF file to save under 'runtime_by_data_dim'.

    max_depth : int, optional
        Maximum depth to consider for the plot. If None, uses the maximum depth in the data.

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The plot axis.
    """
    apply_global_plot_settings()

    df = df.copy()
    df["algorithm"] = df["algorithm"].str.replace("_", " ").str.upper()
    vary_label = "Feature Count" if vary_by == "data_dim" else "Sample Size"

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    for algo in algorithms:
        algo_fmt = algo.replace("_", " ").upper()
        algo_df = df[df["algorithm"] == algo_fmt]
        if algo_df.empty:
            continue

        filtered_df = algo_df.copy()
        if max_depth is not None:
            filtered_df = filtered_df[filtered_df["depth"] <= max_depth]

        grouped = filtered_df.groupby(vary_by)["runtime"].mean().reset_index()
        x_vals = grouped[vary_by].values
        y_vals = grouped["runtime"].values

        # Fit regression line in log-log space
        slope, intercept, *_ = linregress(np.log10(x_vals), np.log10(y_vals))
        y_fit = 10 ** (intercept + slope * np.log10(x_vals))

        label = f"{algo_fmt} (slope = {slope:.2f})"
        color = ALGORITHM_COLORS.get(algo_fmt, "gray")
        ax.plot(x_vals, y_vals, label=label, color=color, marker="o", linewidth=2)
        ax.plot(x_vals, y_fit, linestyle="--", color=color, alpha=0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(LogFormatterSciNotation())
    ax.yaxis.set_major_formatter(LogFormatterSciNotation())

    plot_title = title or f"Runtime Scaling by {vary_label}"
    ax.set_title(plot_title.replace("_", " "))

    ax.legend(loc="lower right", fontsize=10, title="Algorithm", frameon=True)
    ax.get_legend().get_title().set_fontsize(15)

    save_path = (
        DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR
        / "plots"
        / "runtime_by_data_dim"
        / save_name
    )

    beautify_plot(ax=ax, title=plot_title, xlabel=vary_label, ylabel="Runtime (s)")
    ax.legend(fontsize=14)
    save_figure(fig, subfolder="depth_sweep_batch_results/plots/runtime_by_data_dim",
                filename=save_name.replace(".pdf", ""))
    return ax


def plot_multiple_metrics_over_depth_by_dim_or_sample_size(
    df: pd.DataFrame,
    algorithm: str,
    input_axis: str = "data_dim",
    metrics: Tuple[str, ...] = ("accuracy", "coverage", "density"),
    save_name: str = None,
    max_depth: Optional[int] = None
) -> plt.Figure:
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

    max_depth : int, optional
        Maximum depth to consider for the plot. If None, uses the maximum depth in the data.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    apply_global_plot_settings()

    assert input_axis in ["data_dim", "n_samples"], "input_axis must be 'data_dim' or 'n_samples'"
    for m in metrics:
        assert m in df.columns, f"Metric '{m}' not found in dataframe."

    df = df.copy()
    df["algorithm"] = df["algorithm"].str.replace("_", " ").str.upper()
    algorithm = algorithm.replace("_", " ").upper()
    df = df[df["algorithm"] == algorithm]

    if max_depth is not None:
        df = df[df["depth"] <= max_depth]

    if df.empty:
        print(f"No data for algorithm {algorithm}")
        return plt.figure()

    values_sorted = sorted(df[input_axis].unique())
    base_color = ALGORITHM_COLORS.get(algorithm, "#888888")
    color_map = dict(zip(values_sorted, generate_color_gradient(base_color, len(values_sorted))))

    # fig, axes = plt.subplots(nrows=1, ncols=len(metrics), figsize=(6 * len(metrics), 5), sharex=True)
    fig, axes = plt.subplots(nrows=1, ncols=len(metrics), figsize=figsize_wide(len(metrics)), sharex=True)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for v in values_sorted:
            sub = df[df[input_axis] == v]
            grouped = sub.groupby("depth")[metric]
            mean = grouped.mean()

            label = f"{v}" if input_axis == "data_dim" else f"{v:,}"
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
            xlim=(0, max_depth),
            ylim=(0, 1)
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center left", bbox_to_anchor=(0.97, 0.5),
        title="Feature Count" if input_axis == "data_dim" else "Sample Size",
        title_fontsize=16,
        fontsize=14
    )

    fig.suptitle(
        f"{algorithm} - Performance by " + ("Feature Count" if input_axis == "data_dim" else "Sample Size"),
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


def plot_multiple_metrics_over_depth_by_label_noise(
    df: pd.DataFrame,
    algorithm: str,
    metrics: Tuple[str, ...] = ("accuracy", "coverage", "density"),
    save_name: str = None,
    max_depth: Optional[int] = None
) -> plt.Figure:
    """
    Plot average metric trajectories over depth for one algorithm, grouped by label noise.

    Parameters:
    -----------
    df : pd.DataFrame
        Input benchmark data.

    algorithm : str
        Algorithm to filter and plot.

    metrics : tuple of str
        Metrics to plot. Each is rendered as a subplot with shared x-axis (depth).

    save_name : str, optional
        Output PDF path. If None, autogenerated.

    max_depth : int, optional
        Maximum depth to consider for the plot. If None, uses the maximum depth in the data.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    apply_global_plot_settings()
    from src.config.colors_and_plot_styles import generate_color_gradient

    df = df.copy()
    for m in metrics:
        assert m in df.columns, f"Metric '{m}' not found in dataframe."

    df["algorithm"] = df["algorithm"].str.replace("_", " ").str.upper()
    algorithm = algorithm.replace("_", " ").upper()
    df = df[df["algorithm"] == algorithm]

    if max_depth is not None:
        df = df[df["depth"] <= max_depth]

    if df.empty:
        print(f"No data for algorithm {algorithm}")
        return plt.figure()

    df["label_noise"] = df["label_noise"].astype(str)
    values_sorted = sorted(df["label_noise"].unique())
    base_color = ALGORITHM_COLORS.get(algorithm, "#888888")
    color_map = dict(zip(values_sorted, generate_color_gradient(base_color, len(values_sorted))))

    fig, axes = plt.subplots(nrows=1, ncols=len(metrics), figsize=figsize_wide(len(metrics)), sharex=True)

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
            xlim=(0, max_depth),
            ylim=(0, 1)
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center left", bbox_to_anchor=(0.97, 0.5),
        title="Boundary Noise",
        title_fontsize=16,
        fontsize=14
    )

    fig.suptitle(f"{algorithm} - Performance by Boundary Noise Level", fontsize=22)
    fig.tight_layout(rect=(0, 0, 0.965, 1))

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
    apply_global_plot_settings()

    seed = DEFAULT_VARIABLE_SEEDS[seed]
    all_shapes = sorted(df["dataset"].unique())
    ncols = 4
    nrows = (len(all_shapes) + ncols - 1) // ncols

    cmap = cm.get_cmap("viridis", max_depth + 1)
    norm = mcolors.Normalize(vmin=0, vmax=max_depth)

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

    fig.suptitle(f"Coverage - Density Tradeoff\n{algorithm.upper()}, Seed {seed}", fontsize=16)

    save_path = save_name or os.path.join(
        DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR, "plots", f"coverage_density_colormap_{algorithm}.pdf"
    )
    fig.savefig(save_path, bbox_inches="tight")
    print(f"\nSaved to: {save_path}")
