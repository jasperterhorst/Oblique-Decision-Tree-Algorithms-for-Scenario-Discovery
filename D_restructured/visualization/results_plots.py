"""
Module for plotting benchmark results.

Provides functions to visualize metrics such as accuracy vs. depth or aggregated metrics.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.config.colors import AXIS_LINE_COLOR
from src.config.plot_settings import beautify_plot


def plot_results(df, metric="accuracy", title=None, xlabel="Depth", ylabel=None,
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

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df, x="depth", y=metric, hue="algorithm", style="dataset", ax=ax)

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if title is None:
        title = f"{metric.capitalize()} vs. Depth (per dataset)"

    beautify_plot(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel or metric.capitalize(), save_path=save_path)
    return ax


def plot_aggregated_metric(df, metric="accuracy", title=None, xlabel="Depth", ylabel=None,
                           x_lim=None, y_lim=None, save_name=None):
    """
    Plot an aggregated metric (averaged over datasets) as a function of tree depth.

    Parameters:
        df (pd.DataFrame): DataFrame with benchmark results.
        metric (str): The metric to plot.
        title (str): Plot title.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        x_lim (tuple): x-axis limits.
        y_lim (tuple): y-axis limits.
        save_name (str): Filename to save the plot.
    """
    if metric not in df.columns:
        print(f"Metric '{metric}' not found in columns: {list(df.columns)}")
        return

    agg_df = df.groupby(["algorithm", "depth"], as_index=False)[metric].mean()

    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    save_dir = os.path.join("..", "_data", "depth_sweep_batch_results")
    os.makedirs(save_dir, exist_ok=True)
    if save_name is None:
        save_name = f"{metric}_vs_depth_mean.pdf"
    save_path = os.path.join(save_dir, save_name)

    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in agg_df["algorithm"].unique():
        sub = agg_df[agg_df["algorithm"] == algo]
        ax.plot(sub["depth"], sub[metric], label=algo)

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if title is None:
        title = f"Mean {metric.capitalize()} vs. Depth (averaged over datasets)"

    beautify_plot(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel or metric.capitalize(), save_path=save_path)
    return ax
