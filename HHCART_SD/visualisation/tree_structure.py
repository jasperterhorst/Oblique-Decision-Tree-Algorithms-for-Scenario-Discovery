"""
Oblique Tree Visualizer (tree_structure.py)
----------------------------------------------
Visualize the structure of oblique decision trees from HHCART_SD using Graphviz.
Mimics sklearn-style layout with linear splits, sample sizes, and leaf predictions.

Supports:
- Uses feature names from HHCartD.X
- Optional class labels
- Depth selection via HHCartD
- Integration into HHCartD objects
"""

import graphviz
import numpy as np
from matplotlib import colors as mcolors
from ..tree import LeafNode, DecisionNode
from .base.colors import generate_color_gradient, PRIMARY_LIGHT, PRIMARY_DARK, PRIMARY_MIDDLE


def plot_tree_structure(hh, depth=None, coloring="distribution", color_axis_aligned=False,
                        stack_split_terms=True, decimals=2, save=False, filename=None, title=None):
    """
    Plot the oblique tree structure at a given depth using Graphviz.

    Args:
        hh (HHCartD): A fitted HHCART_SD model.
        depth (int): Desired depth to visualize (must exist in hh.trees_by_depth).
        coloring (str): Node coloring strategy: 'class', 'samples', 'distribution', or None.
        color_axis_aligned (bool): If True, highlight axis-aligned splits in red.
        stack_split_terms (bool): If True, display each term of a split equation on a new line.
                                  If False, display on a single line. Defaults to True.
        decimals (int): The number of decimal places to display for weights and bias values. Defaults to 2.
        save (bool): Whether to save the figure.
        filename (str): Custom filename to save as (PDF).
        title (str): Optional graph title.

    Returns:
        graphviz.Digraph: The rendered tree graph.
    """
    feature_names = hh.X.columns.tolist()
    if depth is None:
        if hh.selected_depth is None:
            raise ValueError("No depth provided and no depth selected on HHCartD object.")
        depth = hh.selected_depth
    if depth not in hh.trees_by_depth:
        raise ValueError(f"Tree at depth {depth} not found. Available: {list(hh.trees_by_depth.keys())}")

    tree = hh.get_tree_by_depth(depth)
    final_title = title or f"Oblique Tree Structure – Depth {depth}"

    dot = graphviz.Digraph(comment=final_title)
    dot.attr(label=f"{final_title}\n\n", labelloc="t", fontsize="20")
    dot.attr('node', fontsize='14')

    all_nodes = list(tree.root.traverse_yield())
    max_samples = max(
        node.n_samples if isinstance(node, LeafNode) else (len(node.y) if node.y is not None else 0)
        for node in all_nodes
    )

    sample_gradient = generate_color_gradient(PRIMARY_MIDDLE, 100)

    def compute_sample_color(sample_count):
        idx = int((sample_count / max_samples) * (len(sample_gradient) - 1))
        rgba = sample_gradient[idx]
        return mcolors.to_hex(rgba)

    node_counter = [0]
    node_depth_map = {}

    def add_node_to_graph(node, is_root=False):
        node_id = str(id(node))
        node_label_id = f"Node {node_counter[0]}"
        node_counter[0] += 1
        node_depth_map.setdefault(node.depth, []).append(node_id)

        # Fallback default values
        label = f"{node_label_id}\n[Unknown node type]"
        color = "white"

        if isinstance(node, LeafNode):
            if coloring == "distribution" and node.n_samples and node.purity is not None:
                class1_pct = node.purity if node.prediction == 1 else 1 - node.purity
                class0_pct = 1 - class1_pct
                label = (f"<<TABLE BORDER='0' CELLBORDER='1' CELLSPACING='0'>\n<tr><td BGCOLOR='{PRIMARY_LIGHT}' "
                         f"WIDTH='{int(100 * class0_pct)}'></td><td BGCOLOR='{PRIMARY_DARK}' "
                         f"WIDTH='{int(100 * class1_pct)}'></td></tr>\n<tr><td COLSPAN='2'><FONT POINT-SIZE='2'>"
                         f"<br/></FONT>{node_label_id}<br/>Predict: {node.prediction}<br/>Samples: {node.n_samples}"
                         f"</td></tr></TABLE>>")
                color = None
            else:
                label = f"{node_label_id}\nPrediction: {node.prediction}\nSamples: {node.n_samples}"
                color = (
                    PRIMARY_DARK if node.prediction == 1 else PRIMARY_LIGHT
                    if coloring == "class" else
                    compute_sample_color(node.n_samples) if coloring == "samples" else
                    "white"
                )

        elif isinstance(node, DecisionNode):
            sample_count = len(node.y) if node.y is not None else 0

            if node.is_axis_aligned:
                idxs = np.flatnonzero(node.weights)
                if len(idxs) == 1:
                    i = idxs[0]
                    coef = node.weights[i]
                    lefths = f"{coef:.{decimals}f}·{feature_names[i]}"
                else:
                    lefths = "[WARNING] axis-aligned malformed"
            else:
                terms = []
                for i, w in enumerate(node.weights):
                    if w != 0:
                        sign = "+" if w > 0 and terms else ""
                        weight_val = abs(w)
                        op_sign = "-" if w < 0 else sign
                        terms.append(f"{op_sign} {weight_val:.{decimals}f}·{feature_names[i]}")

                separator = "<br/>" if stack_split_terms else " "
                lefths = separator.join(terms).strip()

            # --- MODIFICATION IS HERE ---
            # Reverted to a <FONT> tag structure, which is more robust for the DOT parser.
            # The global fontsize='14' will control the text size.
            label = (f"<<FONT>{node_label_id}<BR/>{lefths} &le; {node.bias:.{decimals}f}"
                     f"<BR/>Samples: {sample_count}</FONT>>")

            color = (
                "#ffcccc" if color_axis_aligned and node.is_axis_aligned else
                PRIMARY_DARK if coloring == "class" and node.get_majority_class() == 1 else
                PRIMARY_LIGHT if coloring == "class" else
                compute_sample_color(sample_count) if coloring == "samples" else
                "white"
            )

        if isinstance(node, LeafNode) and coloring == "distribution" and node.purity is not None:
            dot.node(node_id, label=label, shape='box', style='filled', fillcolor="white", margin="0")
        else:
            dot.node(node_id, label=label, style='filled', fillcolor=color, shape='box')

        for i, child in enumerate(node.children):
            child_id = str(id(child))
            add_node_to_graph(child)
            dot.edge(node_id, child_id, label="  True" if is_root and i == 1 else "  False" if is_root else "")

    add_node_to_graph(tree.root, is_root=True)

    prev_dummy = None
    for d in sorted(node_depth_map.keys()):
        dummy_id = f"depth_{d}"
        dot.node(dummy_id, label=f"Depth {d}", shape='plaintext', fontsize='14', fontcolor='gray')
        if prev_dummy:
            dot.edge(prev_dummy, dummy_id, style='invis')
        prev_dummy = dummy_id

    if save:
        suffix = "_axis_aligned_colored" if color_axis_aligned else ""
        filename = filename or f"tree_structure_d{depth}{suffix}"
        if hh.save_dir is None:
            raise ValueError("Cannot save Graphviz output: `hh.save_dir` is not set.")
        filepath = f"{hh.save_dir}/{filename}"
        dot.render(filepath, format="pdf", cleanup=True)
        print(f"[SAVE] Graphviz tree saved to: {filepath}.pdf")

    return dot
