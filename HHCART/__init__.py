"""
HHCART Package Interface
-------------------------
Provides access to the high-level HHCART scenario discovery class.

Usage:
    from HHCART import HHCartD

    hh = HHCartD(X_df, y_array, max_depth=6, feature_selector="mutual_info")
    hh.build_tree(k_range=[3, 5, 7])
    hh.select(depth=3, k=5)
    hh.plot_tree()
"""

from .core import HHCartD
