import os
import pydot
import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from datetime import datetime

from ema_workbench.analysis.cart import CART
from ema_workbench.analysis import scenario_discovery_util as sdutil

from src.config.colors_and_plot_styles import PRIMARY_MIDDLE, PRIMARY_LIGHT


class ScenarioNode:
    """A custom class to represent a single node in our processed tree."""

    def __init__(self, node_id: int, cart_alg: CART, parent_box: pd.DataFrame, parent_rule: tuple = None):
        self.node_id = node_id
        tree_ = cart_alg.clf.tree_

        self.n_samples = tree_.n_node_samples[node_id]
        self.value = tree_.value[node_id][0] * self.n_samples
        self.is_leaf = tree_.children_left[node_id] == -1
        self.children = []

        total_samples_in_dataset = tree_.n_node_samples[0]
        total_coi_in_dataset = np.sum(cart_alg.y)

        self.sample_perc = (self.n_samples / total_samples_in_dataset) * 100
        self.density = self.value[1] / self.n_samples if self.n_samples > 0 else 0
        self.coverage = self.value[1] / total_coi_in_dataset if total_coi_in_dataset > 0 else 0
        self.majority_class = np.argmax(self.value)

        self.rule_feature, self.rule_threshold, self.rule_text = None, None, ""
        self.box = self._calculate_box(cart_alg, parent_box, parent_rule)
        self.bounds = self._get_formatted_bounds(cart_alg, self.box)

    def _calculate_box(self, cart_alg: CART, parent_box: pd.DataFrame, parent_rule: tuple) -> pd.DataFrame:
        box = parent_box.copy()
        if not parent_rule: return box
        is_left_child, feature_name, threshold = parent_rule
        if cart_alg.sep in feature_name:
            original_feature, category_value_str = feature_name.split(cart_alg.sep)
            category_value = cart_alg.dummiesmap[original_feature][category_value_str]
            if is_left_child:
                box.loc[0, original_feature].discard(category_value)
                box.loc[1, original_feature].discard(category_value)
        else:
            if is_left_child:
                box.loc[1, feature_name] = min(box.loc[1, feature_name], threshold)
            else:
                box.loc[0, feature_name] = max(box.loc[0, feature_name], threshold)
        return box

    def _get_formatted_bounds(self, cart_alg: CART, box: pd.DataFrame) -> Dict[str, str]:
        box_init = sdutil._make_box(cart_alg.x)
        restricted_dims = sdutil._determine_restricted_dims(box, box_init)
        bounds_dict = {}
        for dim in restricted_dims:
            limit = box[dim]
            if pd.api.types.is_numeric_dtype(limit.dtype):
                bounds_dict[dim] = f"[{limit.iloc[0]:.3f}, {limit.iloc[1]:.3f}]"
            else:
                bounds_dict[dim] = f"categories = {sorted(list(limit.iloc[0]))}"
        return bounds_dict


def build_custom_tree(cart_alg: CART) -> ScenarioNode:
    """Translates a trained scikit-learn tree into our own robust ScenarioNode structure."""
    tree_ = cart_alg.clf.tree_
    initial_box = sdutil._make_box(cart_alg.x)

    def build_recursive(node_id: int, parent_box: pd.DataFrame, parent_rule: tuple = None) -> ScenarioNode:
        node = ScenarioNode(node_id, cart_alg, parent_box, parent_rule)
        if not node.is_leaf:
            feature_name = cart_alg.feature_names[tree_.feature[node_id]]
            threshold = tree_.threshold[node_id]
            node.rule_feature, node.rule_threshold = feature_name, threshold
            node.rule_text = f"{feature_name} &lt;= {threshold:.3f}"
            left_rule, right_rule = (True, feature_name, threshold), (False, feature_name, threshold)
            node.children.append(build_recursive(tree_.children_left[node_id], node.box, left_rule))
            node.children.append(build_recursive(tree_.children_right[node_id], node.box, right_rule))
        return node

    return build_recursive(0, initial_box)


def prune_custom_tree(node: ScenarioNode, target_class: int = 1) -> bool:
    """Prunes the custom tree in-place by merging leaves that predict the same class."""
    if node.is_leaf: return False
    pruned_children = any([prune_custom_tree(child, target_class) for child in node.children])
    if all(child.is_leaf and child.majority_class == target_class for child in node.children):
        node.is_leaf = True
        node.children = []
        node.rule_feature, node.rule_threshold, node.rule_text = None, None, ""
        return True
    return pruned_children


def get_all_leaves(root_node: ScenarioNode) -> List[ScenarioNode]:
    """Traverses the custom tree and returns a list of all leaf nodes."""
    if root_node.is_leaf:
        return [root_node]
    else:
        return [leaf for child in root_node.children for leaf in get_all_leaves(child)]


def generate_cart_summary(root_node: ScenarioNode, density_threshold: float = 0.5, print_summary: bool = True,
                          save: bool = False, save_path: Optional[str] = None) -> str:
    """Generates a text summary from our custom tree structure with consistent scenario numbering."""
    all_leaves = get_all_leaves(root_node)

    # --- FIX: Assign a persistent scenario number to every leaf before filtering ---
    for i, leaf in enumerate(all_leaves):
        leaf.scenario_num = i + 1

    target_scenarios = [leaf for leaf in all_leaves if leaf.density > density_threshold and leaf.majority_class == 1]

    total_coi_in_boxes = sum(leaf.value[1] for leaf in target_scenarios)
    total_samples_in_boxes = sum(leaf.n_samples for leaf in target_scenarios)
    total_coi_in_dataset = root_node.value[1] if root_node.n_samples > 0 else 0

    combined_coverage = total_coi_in_boxes / total_coi_in_dataset if total_coi_in_dataset > 0 else 0
    combined_density = total_coi_in_boxes / total_samples_in_boxes if total_samples_in_boxes > 0 else 0

    summary_lines = [f"# CART Scenarios Summary (class 1, Density > {density_threshold})"]
    summary_lines.append(f"\n=== Combined Statistics (for shown scenarios) ===")
    summary_lines.append(f"Combined coverage: {combined_coverage:.3f}")
    summary_lines.append(f"Combined density:  {combined_density:.3f}")
    summary_lines.append(f"\n=== Individual Scenarios (Density > {density_threshold}) ===")

    for scenario in target_scenarios:
        # --- FIX: Use the persistent scenario number ---
        summary_lines.append(f"\nScenario {scenario.scenario_num} (Original Box ID {scenario.node_id}):")
        summary_lines.append(f"  Coverage: {scenario.coverage:.3f}")
        summary_lines.append(f"  Density:  {scenario.density:.3f}")
        if scenario.bounds:
            summary_lines.append("  Bounds:")
            for dim, bound in sorted(scenario.bounds.items()):
                summary_lines.append(f"    {dim}: {bound}")

    full_summary = "\n".join(summary_lines)
    if print_summary: print(full_summary)
    if save:
        if save_path is None:
            save_folder, ts = os.path.join("data", "cart"), datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, f"cart_summary_{ts}.txt")
        else:
            save_folder = os.path.dirname(save_path)
            if save_folder: os.makedirs(save_folder, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(full_summary)
        print(f"\nCART summary saved to: {save_path}")
    return full_summary


def visualize_cart_tree(root_node: ScenarioNode, title: Optional[str] = None, save: bool = False,
                        save_path: Optional[str] = None) -> pydot.Dot:
    """Generates and displays a visualization from our custom tree structure with consistent scenario numbering."""
    all_leaves = get_all_leaves(root_node)

    # --- FIX: Create a map from node_id to the persistent scenario number ---
    leaf_scenarios = {leaf.node_id: i + 1 for i, leaf in enumerate(all_leaves)}

    target_scenarios = [leaf for leaf in all_leaves if leaf.majority_class == 1]
    total_coi_in_boxes = sum(leaf.value[1] for leaf in target_scenarios)
    total_samples_in_boxes = sum(leaf.n_samples for leaf in target_scenarios)
    total_coi_in_dataset = root_node.value[1] if root_node.n_samples > 0 else 0
    combined_coverage = total_coi_in_boxes / total_coi_in_dataset if total_coi_in_dataset > 0 else 0
    combined_density = total_coi_in_boxes / total_samples_in_boxes if total_samples_in_boxes > 0 else 0

    dot = pydot.Dot(graph_type='digraph', fontname="helvetica")

    title_main = title if title is not None else 'CART Analysis'
    title_combined_stats = f'Combined Coverage: {combined_coverage * 100:.0f}% | Combined Density: {combined_density * 100:.0f}%'
    full_title_html = f'<{title_main}<BR/><FONT POINT-SIZE="16">{title_combined_stats}</FONT><BR/>&nbsp;>'
    dot.set_label(full_title_html)
    dot.set_labelloc('t')
    dot.set_fontsize(24)

    nodes_to_process = [root_node]
    while nodes_to_process:
        node = nodes_to_process.pop(0)

        samples_text = f'samples = {int(node.n_samples)}' if node.node_id == 0 else f'samples = {node.sample_perc:.0f}%'
        value_text = f'value = [{int(node.value[0])}, {int(node.value[1])}]'

        if node.is_leaf:
            # --- FIX: Use the persistent scenario number from the map ---
            scenario_num = leaf_scenarios.get(node.node_id, '?')
            scenario_title = f'<B>Scenario {scenario_num}</B>'
            coverage_text = f'Coverage: {node.coverage * 100:.0f}%'
            density_text = f'Density: {node.density * 100:.0f}%'
            label_html = f'<{scenario_title}<BR/>{samples_text}<BR/>{value_text}<BR/>{coverage_text}<BR/>{density_text}>'
            color = PRIMARY_MIDDLE if node.majority_class == 1 else PRIMARY_LIGHT
        else:
            rule_text = f'<B>{node.rule_text}</B>'
            label_html = f'<{rule_text}<BR/>{samples_text}<BR/>{value_text}>'
            color = "#e0e0e0"
            nodes_to_process.extend(node.children)

        pydot_node = pydot.Node(name=str(node.node_id), label=label_html, shape="box", style="filled, rounded",
                                fillcolor=color)
        dot.add_node(pydot_node)

        for child in node.children:
            edge = pydot.Edge(str(node.node_id), str(child.node_id))
            dot.add_edge(edge)

    if save:
        if save_path is None:
            save_folder, ts = os.path.join("data", "cart"), datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, f"cart_tree_{ts}.pdf")
        else:
            save_folder = os.path.dirname(save_path)
            if save_folder: os.makedirs(save_folder, exist_ok=True)
        dot.write_pdf(save_path)
        print(f"Formatted CART tree saved to: {save_path}")

    return dot
