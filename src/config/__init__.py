"""
Unified config module for centralised access to colors, paths, and settings.

Import specific items from here as needed:
    from src.config import ALGORITHM_COLORS, save_figure, apply_global_plot_settings
"""

from .colors_and_plot_styles import (
    ALGORITHM_COLORS,
    SHAPE_TYPE_LINESTYLES,
    NOISE_MARKERS,
    SCATTER_COLORS,
    EVOLUTION_COLORS,

    PRIMARY_LIGHT, PRIMARY_DARK, PRIMARY_MIDDLE, SECONDARY_LIGHT, SECONDARY_DARK, SECONDARY_MIDDLE,

    CART_OUTLINE_COLOR,
    QUADRILATERAL_COLOR,
    AXIS_LINE_COLOR,
    GRID_COLOR,

    midpoint_hex,
    get_algorithm_color,
    get_shape_linestyle,
    get_noise_marker,
    generate_color_gradient
)

from .plot_settings import (
    apply_global_plot_settings,
    beautify_plot,
    beautify_subplot
)

from .paths import (
    DATA_DIR,
    SHAPES_DIR,
    SCENARIO_METHODS_DEMO_OUTPUTS_DIR,
    DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR,
    DEPTH_SWEEP_SINGLE_RUN_RESULTS_OUTPUTS_DIR
)

from .settings import (
    DEFAULT_SEED,
    DEFAULT_VARIABLE_SEEDS
)

from .save_settings import (
    save_figure,
    save_dataframe
)
