"""
B_test_shape_generation Package

This package provides functions for generating 2D and 3D shapes as well as interactive interfaces
(using ipywidgets) for exploring these shapes. It includes:
  — interfaces_ui: Functions to create interactive UI widgets.
  — shape_generators: Functions to generate sample points for various shapes.
  — utils: Utility functions for plotting shapes and saving generated data.
"""

from .interfaces_ui import *
from .shape_generators import *
from .utils import *

__all__ = [
    # From interfaces_ui.py
    "setup_interface",
    "update_interface_2d",
    "update_interface_3d",
    "save_interface_data",
    "create_2d_rectangle_interface",
    "create_2d_radial_segment_interface",
    "create_2d_barbell_interface",
    "create_2d_sine_wave_interface",
    "create_2d_star_interface",
    "create_3d_radial_segment_interface",
    "create_3d_barbell_interface",
    "create_3d_saddle_interface",
    # From shape_generators.py
    "generate_2d_rectangle",
    "generate_2d_radial_segment",
    "generate_2d_barbell",
    "generate_2d_sine_wave",
    "generate_2d_star",
    "generate_3d_radial_segment",
    "generate_3d_barbell",
    "generate_3d_saddle",
    # From utils.py
    "plot_2d_shape",
    "plot_3d_shape",
    "save_data",
    "format_note_text",
]
