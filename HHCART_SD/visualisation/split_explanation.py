"""
Split Explanation Plotting (split_evaluation.py)
---------------------------------------------------------
Visualises the geometric construction of a split in HHCART(D) decision trees.

This module plots three panels for a selected oblique node and depth:
(1) the original feature space with dominant direction,
(2) the reflected feature space with the axis-aligned split, and
(3) the oblique split in the original space.

For axis aligned splits, it shows only the original space with the split line.

Intended to support interpretability and explanation of oblique splits.
Requires nodes to contain `split_metadata` (produced by HHCART).

Dependencies:
- matplotlib
- numpy
- HHCartD object structure with `get_deepest_tree()` and metadata

"""


import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from .base.colors import PRIMARY_LIGHT, PRIMARY_DARK
from .base.plot_settings import apply_global_plot_settings, beautify_subplot
from .base.save_figure import save_figure
from matplotlib.lines import Line2D


def plot_split_explanation_2d(
    hh,
    depth: int = None,
    node_id: int = None,
    save: bool = True,
    filename: str = None,
    title: str = None
):
    """
    Visualise the construction of an oblique or axis-aligned split at a specified node in 2D.

    If the split is oblique:
        - Shows 3 panels: original space, reflected space, and final oblique split.
    If the split is axis-aligned:
        - Shows only 1 panel with the original data and split line.

    Args:
        hh (HHCartD): Trained HHCART-D model with .get_deepest_tree() and .X attributes.
        depth (int, optional): Tree depth to inspect. If None, prompts user.
        node_id (int, optional): Node ID to inspect. If None, prompts user.
        save (bool, optional): Whether to save the figure as PDF.
        filename (str, optional): Output filename for saved figure. Defaults to:
                                  "split_explanation_d{depth}_n{node_id}.pdf".
        title (str, optional): Title shown above the plot. Defaults to auto-generated title.

    Raises:
        ValueError: If the specified depth has no metadata nodes, or the node lacks metadata.
    """
    apply_global_plot_settings()
    tree = hh.get_deepest_tree()

    # --- Case 1: Node ID given directly ---
    if node_id is not None:
        node = next((n for n in tree.root.traverse_yield() if n.node_id == node_id), None)
        if node is None:
            raise ValueError(f"Node ID {node_id} not found in the tree.")
        if not hasattr(node, "split_metadata") or not node.split_metadata:
            raise ValueError(f"No split metadata found on node {node_id}.")

        if depth is not None and node.depth != depth:
            raise ValueError(
                f"\n[ERROR] Node ID {node.node_id} is at depth {node.depth}, not depth {depth}.\n"
                f"Tip: Call hh.plot_split_explanation(depth={node.depth}, node_id={node.node_id}) to display,\n"
                f"or hh.plot_split_explanation() to select interactively."
            )

        depth = node.depth  # always infer correct depth from actual node

    else:
        # --- Case 2: Interactive selection by depth and node ---
        if depth is None:
            depths = hh.available_depths()
            print("\nAvailable depths:")
            for i, d in enumerate(depths):
                print(f"  [{i}] Depth {d}")
            sys.stdout.flush()
            time.sleep(0.1)
            while True:
                try:
                    d_idx = int(input("\nEnter number to select depth: ").strip())
                    if 0 <= d_idx < len(depths):
                        depth = depths[d_idx]
                        break
                    print(f"[WARNING] Choose a valid index (0 to {len(depths)-1})")
                except ValueError:
                    print("[WARNING] Please enter a valid integer.")

        candidates = [n for n in tree.root.traverse_yield()
                      if hasattr(n, "split_metadata") and n.depth == depth]
        if not candidates:
            raise ValueError(f"No nodes with metadata found at depth {depth}")

        print(f"\nNodes with metadata at depth {depth}:")
        for i, node in enumerate(candidates):
            split_type = "axis-aligned" if node.is_axis_aligned else "oblique"
            print(f"  [{i}] Node ID {node.node_id} ({split_type})")
        sys.stdout.flush()
        time.sleep(0.1)
        while True:
            try:
                n_idx = int(input("\nEnter number to select node: ").strip())
                if 0 <= n_idx < len(candidates):
                    node = candidates[n_idx]
                    node_id = node.node_id
                    break
                print(f"[WARNING] Choose a valid index (0 to {len(candidates)-1})")
            except ValueError:
                print("[WARNING] Please enter a valid integer.")
        save_input = input("Save figure as PDF? [y/N]: ").strip().lower()
        save = save_input == "y"

    # --- Extract metadata ---
    meta = node.split_metadata
    X_sub = meta["X_orig"]
    y_sub = meta["y_orig"]
    i, thr = meta["axis_rule"]

    X_all = hh.X.values
    x0_min, x0_max = X_all[:, 0].min(), X_all[:, 0].max()
    x1_min, x1_max = X_all[:, 1].min(), X_all[:, 1].max()

    # === Axis-Aligned Split: Single Panel ===
    if node.is_axis_aligned:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(X_sub[:, 0], X_sub[:, 1],
                   c=np.where(y_sub == 1, PRIMARY_DARK, PRIMARY_LIGHT),
                   s=12, alpha=0.7)

        if i == 0:
            ax.axvline(x=thr, linestyle="--", color="blue")
        elif i == 1:
            ax.axhline(y=thr, linestyle="--", color="blue")

        beautify_subplot(ax, xlabel="X₀", ylabel="X₁",
                         xlim=(x0_min, x0_max), ylim=(x1_min, x1_max))

        fig.text(0.5, 0.02,
                 f"Axis-aligned split in original space (X{i} = {thr:.2f})",
                 ha='center', fontsize=11)

        fig.suptitle(title or f"Axis-Aligned Split (Depth {depth}, Node {node_id})", fontsize=20)
        fig.subplots_adjust(bottom=0.15)

        if save:
            filename = filename or f"axis_split_d{depth}_n{node_id}.pdf"
            save_figure(hh, filename=filename, save=True)

        plt.show()
        plt.close(fig)
        return

    # === Oblique Split: Full 3-Panel Plot ===
    H = meta["H_used"]
    X_rot = X_sub @ H

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel 1: Original space with dominant direction
    axes[0].scatter(X_sub[:, 0], X_sub[:, 1],
                    c=np.where(y_sub == 1, PRIMARY_DARK, PRIMARY_LIGHT),
                    s=12, alpha=0.7)

    dominant_dir = meta.get("dominant_vector", None)
    if dominant_dir is not None:
        dominant_dir = dominant_dir / np.linalg.norm(dominant_dir)
        dominant_class = meta.get("dominant_class", "?")
        center = X_sub.mean(axis=0)
        arrow_length = 0.4 * max(x0_max - x0_min, x1_max - x1_min)
        end = center + arrow_length * dominant_dir
        axes[0].plot([center[0], end[0]], [center[1], end[1]],
                     linestyle="--", color="black", linewidth=2)
    beautify_subplot(axes[0], title="Original Space", xlabel="X₀", ylabel="X₁",
                     xlim=(x0_min, x0_max), ylim=(x1_min, x1_max))

    # Panel 2: Reflected space with axis-aligned split
    axes[1].scatter(X_rot[y_sub == 0][:, 0], X_rot[y_sub == 0][:, 1],
                    color=PRIMARY_LIGHT, s=12, alpha=0.7)
    axes[1].scatter(X_rot[y_sub == 1][:, 0], X_rot[y_sub == 1][:, 1],
                    color=PRIMARY_DARK, s=12, alpha=0.7)
    if i == 0:
        axes[1].axvline(x=thr, linestyle="--", color="blue")
    elif i == 1:
        axes[1].axhline(y=thr, linestyle="--", color="blue")
    beautify_subplot(axes[1], title="Reflected Space", xlabel="Reflected X₀", ylabel="Reflected X₁")

    # Panel 3: Oblique split in original space
    w, b = node.weights, node.bias
    x_vals = np.linspace(x0_min, x0_max, 200)
    if not np.isclose(w[1], 0.0):
        y_vals = -(w[0] * x_vals + b) / w[1]
    else:
        x_vals = np.full_like(x_vals, -b / w[0])
        y_vals = np.linspace(x1_min, x1_max, len(x_vals))
    axes[2].scatter(X_sub[:, 0], X_sub[:, 1],
                    c=np.where(y_sub == 1, PRIMARY_DARK, PRIMARY_LIGHT),
                    s=12, alpha=0.7)
    axes[2].plot(x_vals, y_vals, linestyle="--", color="red", linewidth=1.5)
    beautify_subplot(axes[2], title="Split in Original Space", xlabel="X₀", ylabel="X₁",
                     xlim=(x0_min, x0_max), ylim=(x1_min, x1_max))

    # Combined legend
    line_handles = []
    if dominant_dir is not None:
        line_handles.append(Line2D([0], [0], linestyle="--", color="black",
                                   label=f"Dominant direction (class {dominant_class})"))
    line_handles.append(Line2D([0], [0], linestyle="--", color="blue",
                               label=f"Axis-aligned split in reflected space (X{i} = {thr:.2f})"))
    line_handles.append(Line2D([0], [0], linestyle="--", color="red",
                               label="Oblique split in original space"))

    fig.legend(handles=line_handles,
               loc="lower center", bbox_to_anchor=(0.5, -0.01),
               ncol=3, frameon=False, fontsize=10)

    fig.suptitle(title or f"Split Construction (Depth {depth}, Node {node_id})", fontsize=22)
    fig.subplots_adjust(wspace=0.4, top=0.82, bottom=0.22)

    if save:
        filename = filename or f"split_explanation_d{depth}_n{node_id}.pdf"
        save_figure(hh, filename=filename, save=True)

    plt.show()
    plt.close(fig)
