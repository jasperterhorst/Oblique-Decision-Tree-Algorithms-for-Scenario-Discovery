"""
HHCART_SD: Oblique Decision Tree Framework for Scenario Discovery
==================================================================

This is the top-level interface for the `hhcart_sd` package — an interpretable,
depth-aware implementation of oblique decision trees designed for scenario discovery.

It extends the original HHCART-D algorithm (Wickramarachchi et al., 2016), adapted from
Majumder's reference codebase, with key additions like:
- Minimum purity thresholds
- Depth-based saving and evaluation
- Scenario discovery metrics (coverage and density)
- Visualization methods integrated into the main model class

Modules:
--------
- core.py          : Main model interface (HHCartD)
- io/save_load.py  : Model persistence utilities

Usage Example:
--------------
from hhcart_sd import HHCartD, save_model, load_model

hh = HHCartD(X, y, max_depth=5, min_purity=0.9)
hh.build_tree()
hh.select(depth=3)
hh.plot_tree_structure()

save_model(hh, "run_3")
loaded = load_model("run_3")
"""

from .core import HHCartD
from .io import save_load

save_model = save_load.save_full_model
load_model = save_load.load_full_model

__all__ = ["HHCartD", "save_model", "load_model"]
