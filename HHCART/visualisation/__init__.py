from .tradeoff import plot_coverage_density_tradeoff
from .split_pairwise import plot_pairwise_splits as _plot_pairwise_splits_impl
# from .split_regions import plot_decision_regions as _plot_decision_regions_impl


def bind_plotting_methods(obj):
    """
    Dynamically bind visualisation methods to an HHCartD object instance.

    Adds the following methods to the instance:
    - plot_tradeoff(k_range=None, save_path=None)
    - plot_pairwise_splits(k, depth, save_path=None)
    - plot_decision_regions(k, max_depth=None, save_path=None)

    Args:
        obj: An instance of HHCartD.

    Raises:
        AttributeError: If required attributes are missing on the object.
    """
    required_attrs = ["trees_by_depth", "feature_selector_top_k", "X", "y"]
    for attr in required_attrs:
        if not hasattr(obj, attr):
            raise AttributeError(f"The object must have attribute '{attr}'.")

    def plot_tradeoff(k_range=None, save_path=None):
        """
        Plot the coverage–density trade-off for all or a subset of feature counts (k).

        Args:
            k_range (list[int], optional): Which k values to include. If None, all available are used.
            save_path (str, optional): If provided, saves the figure to this path. Otherwise shows interactively.
        """
        return plot_coverage_density_tradeoff(obj, k_range=k_range, save_path=save_path)

    def plot_pairwise_splits(k=None, depth=None, save_path=None):
        """
        Plot clipped oblique splits across all 2D feature pairs for a given (k, depth).

        Args:
            k (int or None): Number of selected features. If None, use all features.
            depth (int): Depth of the tree to visualise.
            save_path (str, optional): Path to save the figure. Otherwise shows interactively.

        Raises:
            ValueError: If no tree is found for the (k, depth) combination.
        """
        key = (k, depth)
        tree = obj.trees_by_depth.get(key)
        if tree is None:
            raise ValueError(f"No tree found for k={k}, depth={depth}.")

        if k is None:
            features = list(obj.X.columns)
            X_subset = obj.X
        else:
            features = obj.feature_selector_top_k(k)
            X_subset = obj.X[features]

        return _plot_pairwise_splits_impl(X_subset, obj.y, tree, features, save_path=save_path)

    # def plot_decision_regions(k, max_depth=None, save_path=None):
    #     """
    #     Plot decision regions for all depths of the tree trained with k features.
    #
    #     Args:
    #         k (int): Feature subset size used for training.
    #         max_depth (int, optional): Max depth to plot. If None, plots up to trained depth.
    #         save_path (str, optional): Save path for figure. If None, shows interactively.
    #
    #     Raises:
    #         ValueError: If no trees exist for the provided k.
    #     """
    #     trees = {
    #         depth: tree for (kk, depth), tree in obj.trees_by_depth.items() if kk == k
    #     }
    #     if not trees:
    #         raise ValueError(f"No trees found for k={k}.")
    #     max_depth = max(trees) if max_depth is None else max_depth
    #     tree_dict = {f"k={k}": trees}
    #     return _plot_decision_regions_impl(obj.X, obj.y, tree_dict, max_depth=max_depth, save_path=save_path)

    # Attach all methods to the HHCartD instance
    obj.plot_tradeoff = plot_tradeoff
    obj.plot_pairwise_splits = plot_pairwise_splits
    # obj.plot_decision_regions = plot_decision_regions
