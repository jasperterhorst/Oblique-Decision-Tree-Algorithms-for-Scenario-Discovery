"""
Module for plotting benchmark results.

Provides functions to visualize metrics such as accuracy vs. depth or aggregated metrics.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.config.plot_settings import beautify_plot
import pandas as pd


def plot_separate_metric_against_depth(df, metric="accuracy", title=None, xlabel="Depth", ylabel=None,
                                       x_lim=None, y_lim=None, save_name=None):
    """
    Plot a given metric as a function of tree depth for each algorithm and dataset.

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

    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    save_dir = os.path.join("..", "_data", "depth_sweep_batch_results")
    os.makedirs(save_dir, exist_ok=True)
    if save_name is None:
        save_name = f"{metric}_vs_depth_per_dataset.pdf"
    save_path = os.path.join(save_dir, save_name)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(data=df, x="depth", y=metric, hue="algorithm", style="dataset", ax=ax)

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


def plot_aggregated_metric_against_depth(df, metric="accuracy", title=None, xlabel="Depth", ylabel=None,
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

    save_dir = os.path.join("..", "_data", "depth_sweep_batch_results")
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_name or os.path.join(save_dir, f"{metric}_vs_depth_mean.pdf")

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


def plot_metric_by_depth_per_shape(df, metric="accuracy", title=None, xlabel="Depth", ylabel=None,
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

    save_dir = os.path.join("..", "_data", "depth_sweep_batch_results")
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_name or os.path.join(save_dir, f"{metric}_vs_depth_per_shape.pdf")

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
