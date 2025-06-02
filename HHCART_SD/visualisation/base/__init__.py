"""
Base Utilities for Styling and Export
=====================================

This module provides low-level utilities for styling and exporting plots.
These are used internally by higher-level visualisation functions.

Modules:
--------
- colors.py         : Color mappings and class palette
- plot_settings.py  : Global Matplotlib configuration
- save_figure.py    : PDF export utility

Usage Example:
--------------
from hhcart_sd.visualisation.base.plot_settings import apply_global_plot_settings
apply_global_plot_settings()

from hhcart_sd.visualisation.base.save_figure import save_figure
save_figure(hh, "tree_depth3.pdf", save=True)
"""

from .colors import *
from .plot_settings import *
from .save_figure import *
