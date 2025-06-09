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
from matplotlib import colors as mcolors
from ..tree import LeafNode, DecisionNode
from .base.colors import generate_color_gradient, PRIMARY_LIGHT, PRIMARY_DARK, PRIMARY_MIDDLE


def plot_tree_structure(hh, depth=None, coloring="distribution", save=False, filename=None, title=None):
    """
    Plot the oblique tree structure at a given depth using Graphviz.

    Args:
        hh (HHCartD): A fitted HHCART_SD model.
        depth (int): Desired depth to visualize (must exist in hh.trees_by_depth).
        coloring (str): Node coloring strat egy: 'class', 'samples', 'distribution', or None.
        save (bool): Whether to save the figure.
        filename (str): Custom filename to save as (PDF).
        title (str): Optional graph title.

    Returns:
        graphviz.Digraph: The rendered tree graph.
    """
    if depth is None:
        if hh.selected_depth is None:
            raise ValueError("No depth provided and no depth selected on HHCartD object.")
        depth = hh.selected_depth

    if depth not in hh.trees_by_depth:
        raise ValueError(f"Tree at depth {depth} not found. Available: {list(hh.trees_by_depth.keys())}")

    feature_names = hh.X.columns.tolist()
    tree = hh.get_tree_by_depth(depth)
    dot = graphviz.Digraph(comment=title or f"Tree Depth {depth}")
    dot.attr(label=f"{title or f'Oblique Tree Structure – Depth {depth}'}\n\n", labelloc="t", fontsize="20")

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

    node_counter = [0]  # mutable counter
    node_depth_map = {}

    def add_node_to_graph(node, is_root=False):
        node_id = str(id(node))
        node_label_id = f"Node {node_counter[0]}"
        node_counter[0] += 1

        node_depth = node.depth
        node_depth_map.setdefault(node_depth, []).append(node_id)

        if isinstance(node, LeafNode):
            if coloring == "distribution" and node.n_samples and node.purity is not None:
                if node.prediction == 1:
                    class1_pct = node.purity
                    class0_pct = 1 - class1_pct
                else:
                    class0_pct = node.purity
                    class1_pct = 1 - class0_pct
                label = (f"<<TABLE BORDER='0' CELLBORDER='1' CELLSPACING='0'>\n<tr><td BGCOLOR='{PRIMARY_LIGHT}' "
                         f"WIDTH='{int(100 * class0_pct)}'></td><td BGCOLOR='{PRIMARY_DARK}' "
                         f"WIDTH='{int(100 * class1_pct)}'></td></tr>\n<tr><td COLSPAN='2'><FONT POINT-SIZE='2'>"
                         f"<br/></FONT>{node_label_id}<br/>Predict: {node.prediction}<br/>Samples: {node.n_samples}"
                         f"</td></tr></TABLE>>")
                color = None
            else:
                label = f"{node_label_id}\nPrediction: {node.prediction}\nSamples: {node.n_samples}"
                if coloring == "class":
                    color = PRIMARY_DARK if node.prediction == 1 else PRIMARY_LIGHT
                elif coloring == "samples":
                    color = compute_sample_color(node.n_samples)
                else:
                    color = "white"
        elif isinstance(node, DecisionNode):
            parts = [
                f"{w:+.2f}·{feature_names[i]}"
                for i, w in enumerate(node.weights)
            ]
            lhs = " ".join(parts)
            sample_count = len(node.y) if node.y is not None else 0
            label = f"{node_label_id}\n{lhs} ≤ {node.bias:.2f}\nSamples: {sample_count}"
            if coloring == "class":
                majority = node.get_majority_class()
                color = PRIMARY_DARK if majority == 1 else PRIMARY_LIGHT
            elif coloring == "samples":
                color = compute_sample_color(sample_count)
            else:
                color = "white"
        else:
            label = "Unknown node"
            color = "white"

        if coloring == "distribution" and isinstance(node, LeafNode) and node.purity is not None:
            dot.node(node_id, label=label, shape='box', style='filled', fillcolor="white", margin="0")
        else:
            dot.node(node_id, label=label, style='filled', fillcolor=color, shape='box')

        for i, child in enumerate(node.children):
            child_id = str(id(child))
            add_node_to_graph(child)
            if is_root:
                edge_label = "  True" if i == 1 else "  False"
                dot.edge(node_id, child_id, label=edge_label)
            else:
                dot.edge(node_id, child_id)

    add_node_to_graph(tree.root, is_root=True)

    # Add vertical depth labels
    prev_dummy = None
    for d in sorted(node_depth_map.keys()):
        dummy_id = f"depth_{d}"
        dot.node(dummy_id, label=f"Depth {d}", shape='plaintext', fontsize='12', fontcolor='gray')
        if prev_dummy:
            dot.edge(prev_dummy, dummy_id, style='invis')
        prev_dummy = dummy_id

    if save:
        filename = filename or f"tree_structure_d{depth}"
        if hh.save_dir is None:
            raise ValueError("Cannot save Graphviz output: `hh.save_dir` is not set.")
        filepath = f"{hh.save_dir}/{filename}"
        dot.render(filepath, format="pdf", cleanup=True)
        print(f"[SAVE] Graphviz tree saved to: {filepath}.pdf")

    return dot
