import os
import pickle
import pandas as pd

from HHCART.HHCartDLayerwise import HHCartDLayerwiseClassifier
from HHCART.split_criteria import gini


def run_layerwise_benchmark(
    X,
    y,
    *,
    dataset_name: str = "unnamed",
    save_dir: str = None,
    max_depth: int = 8,
    min_samples_split: int = 10,
    min_purity: float = 0.95,
    tau: float = 1e-4,
    random_state: int = None,
    output_name: str = None,
    save: bool = True
) -> tuple[pd.DataFrame, dict]:
    """
    Train and benchmark a layer-wise HHCART(D) model.

    Fits HHCartDLayerwiseClassifier, evaluates metrics at each depth,
    and optionally saves the results and tree to disk.

    Parameters:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Binary labels.
        dataset_name (str): Identifier used for naming outputs.
        save_dir (str): Directory to save output files. Required if save=True.
        max_depth (int): Maximum tree depth.
        min_samples_split (int): Minimum samples required to split.
        min_purity (float): Node purity threshold to stop splitting.
        tau (float): Householder reflection stability threshold.
        random_state (int): Random seed for reproducibility.
        output_name (str): Base name for output files. Defaults to "{dataset_name}__hhcart_d_layerwise".
        save (bool): Whether to write metrics/tree to disk.

    Returns:
        tuple:
            - pd.DataFrame: Evaluation metrics by depth.
            - DecisionTree: Final fitted decision tree object.
    """
    model = HHCartDLayerwiseClassifier(
        impurity=gini,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_purity=min_purity,
        tau=tau,
        random_state=random_state,
    )
    model.fit(X, y)

    tree = model.get_tree()
    tree.variable_names = model.variable_names

    metrics_by_depth = _format_metrics(
        model.metrics_by_depth,
        n_samples=X.shape[0],
        data_dim=X.shape[1],
    )
    metrics_df = pd.DataFrame(metrics_by_depth)

    if save:
        output_name = output_name or f"{dataset_name}__hhcart_d_layerwise"
        _persist_outputs(metrics_df, model.trees_by_depth, save_dir, output_name)

    return metrics_df, model.trees_by_depth


def _format_metrics(metrics_dict, *, n_samples, data_dim):
    """Format metrics and attach basic dataset metadata."""
    return [
        {
            **metrics.copy(),
            "n_samples": n_samples,
            "data_dim": data_dim,
            "depth": depth,
        }
        for depth, metrics in sorted(metrics_dict.items())
    ]


def _persist_outputs(df: pd.DataFrame, trees_by_depth: dict, save_dir: str, output_name: str):
    """
    Save metrics DataFrame and all per-depth decision trees to disk.

    Parameters:
        df (pd.DataFrame): Evaluation metrics.
        trees_by_depth (dict): Mapping from depth to DecisionTree.
        save_dir (str): Directory to write outputs.
        output_name (str): Prefix used in saved filenames.
    """
    if not save_dir:
        raise ValueError("save_dir must be specified when save=True.")
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, f"{output_name}__metrics.csv")
    pkl_path = os.path.join(save_dir, f"{output_name}__trees_by_depth.pkl")

    df.to_csv(csv_path, index=False)

    with open(pkl_path, "wb") as f:
        pickle.dump(trees_by_depth, f)
