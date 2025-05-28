def bind_help_method(hh):
    """
    Attach a .help() method to an HHCART object that explains available methods,
    including inputs, usage tips, and descriptions.
    """
    def help():
        print("\n🧠 HHCART D — Oblique Decision Tree Interface\n")
        print("─────────────────────────────────────────────────────")
        print("Core Methods\n")

        print("• hh.build_tree(k_range: list[int], save_path: Optional[str] = None)")
        print("  → Train oblique trees using Householder reflections.")
        print("    - k_range: List of feature subset sizes (e.g., [2, 3, 4])")
        print("    - save_path: Optional path to save intermediate results.")
        print("    Builds trees for each (k, depth) up to the specified max depth.\n")

        print("• hh.select_tree(k: int, depth: int) -> DecisionTree")
        print("  → Select and return a specific trained tree.")
        print("    - k: Number of selected features used for training")
        print("    - depth: Depth level of the tree")
        print("    Returns a DecisionTree object.\n")

        print("• hh.inspect(k: int, depth: int)")
        print("  → Print structure of a specific tree (splits, thresholds, classes).\n")

        print("• hh.evaluate(k: int, depth: int, metrics: Optional[list[str]] = None) -> dict")
        print("  → Compute evaluation metrics for a selected tree.")
        print("    - metrics: List of metric names (e.g., ['accuracy', 'coverage'])")
        print("    Returns a dictionary with metric values.\n")

        print("─────────────────────────────────────────────────────")
        print("Visualisation Methods\n")

        print("• hh.plot_tradeoff(k_range: Optional[list[int]] = None, save_path: Optional[str] = None)")
        print("  → Plot coverage–density trade-off curves.")
        print("    - k_range: Optional list of feature counts to include.")
        print("    - save_path: Path to save the plot (PDF, PNG). If None, display live.\n")

        print("• hh.plot_pairwise_splits(k: int, depth: int, save_path: Optional[str] = None)")
        print("  → Show 2D scatter plots of decision boundaries for each feature pair.")
        print("    - k: Number of top features used")
        print("    - depth: Depth of the tree")
        print("    - save_path: File path for output (optional).\n")

        print("• hh.plot_decision_regions(k: int, max_depth: Optional[int] = None, save_path: Optional[str] = None)")
        print("  → Plot decision regions for all splits at depth ≤ max_depth.")
        print("    - k: Number of top features selected")
        print("    - max_depth: Maximum depth to include (optional)")
        print("    - save_path: File path for output (optional)\n")

        print("─────────────────────────────────────────────────────")
        print("Utility\n")

        print("• hh.help()")
        print("  → Show this interactive help message.\n")

    hh.help = help
