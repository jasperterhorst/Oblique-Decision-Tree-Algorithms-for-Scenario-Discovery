import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.config.plot_settings import beautify_plot
from src.config.paths import DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR


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
        group_by = ["algorithm", "dataset"]

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
