"""
Node Sample Size Distribution (node_size_distribution.py)
------------------------------
Visualizes the distribution of sample sizes across
all nodes in the tree, grouped by depth.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

from .base.plot_settings import apply_global_plot_settings, beautify_plot
from .base.colors import PRIMARY_MIDDLE
from .base.save_figure import save_figure


def plot_node_size_distribution(hh, save=False, filename=None, title=None):
    """
    Plot the distribution of node sizes (sample counts) grouped by depth.

    Args:
        hh (HHCartD): Trained HHCART_SD-D wrapper with .trees_by_depth populated.
        save (bool): Whether to save the figure.
        filename (str): PDF output filename.
        title (str, optional): Optional plot title.

    Returns:
        (fig, ax): Tuple of matplotlib figure and axis.
    """
    if filename is None:
        filename = "node_size_distribution.pdf"

    apply_global_plot_settings()
    size_data = defaultdict(list)

    # Gather sizes from all nodes via traversal
    for depth, tree in hh.trees_by_depth.items():
        for node in tree.root.traverse_yield():
            if hasattr(node, "n_samples") and node.n_samples is not None:
                size_data[depth].append(node.n_samples)

    # Flatten into DataFrame for Seaborn
    rows = [(depth, size) for depth, sizes in size_data.items() for size in sizes]
    df = pd.DataFrame(rows, columns=["depth", "n_samples"])

    fig, ax = plt.subplots(figsize=(5.5, 4))
    sns.boxplot(data=df, x="depth", y="n_samples", ax=ax,
                boxprops=dict(facecolor=PRIMARY_MIDDLE, edgecolor="black"),
                medianprops=dict(color="black"))

    final_title = title or "Node Size Distribution by Depth"
    beautify_plot(ax,
                  title=final_title,
                  xlabel="Tree Depth",
                  ylabel="Samples per Node")

    ax.set_ylim(bottom=0)

    plt.tight_layout()
    save_figure(hh, filename, save)

    return fig, ax
