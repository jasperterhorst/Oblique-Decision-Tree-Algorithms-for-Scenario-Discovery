"""
Unified project-wide config import hub.

This makes all color settings, paths, save functions, and dataset loaders available
in a single namespace for convenience across notebooks and modules.

Example usage:
    from src import SCATTER_COLORS, save_figure, load_all_shape_datasets
"""

from src.config import (
    # Colors and plot style access
    ALGORITHM_COLORS,
    SHAPE_TYPE_LINESTYLES,
    NOISE_MARKERS,
    SCATTER_COLORS,
    EVOLUTION_COLORS,
    PRIMARY_LIGHT, PRIMARY_DARK, PRIMARY_MIDDLE,
    SECONDARY_LIGHT, SECONDARY_DARK, SECONDARY_MIDDLE,
    CART_OUTLINE_COLOR, QUADRILATERAL_COLOR,
    AXIS_LINE_COLOR, GRID_COLOR,
    midpoint_hex,
    get_algorithm_color,
    get_shape_linestyle,
    get_noise_marker,
    generate_color_gradient,

    # Plot styling
    apply_global_plot_settings,
    beautify_plot,
    beautify_subplot,

    # Paths
    DATA_DIR,
    SHAPES_DIR,
    SCENARIO_METHODS_DEMO_OUTPUTS_DIR,
    DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR,
    DEPTH_SWEEP_SINGLE_RUN_RESULTS_OUTPUTS_DIR,

    # Save logic
    save_figure,
    save_dataframe,

    # Settings
    DEFAULT_SEED,
    DEFAULT_VARIABLE_SEEDS
)

from src.load_shapes import load_shape_dataset
