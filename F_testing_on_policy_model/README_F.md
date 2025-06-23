# F - Testing on the EU Energy Policy Model

This module applies scenario discovery algorithms, **PRIM**, **PCA-PRIM**, **CART**, and **HHCART(D)**, to the high-dimensional EU energy transition model introduced by Hamarat et al. The goal is to evaluate how each method performs in a realistic policy setting by analysing subspace coverage, density, and interpretability.

Main notebooks:
- [`hamarat_prim.ipynb`](./hamarat_prim.ipynb)
- [`hamarat_pca_prim.ipynb`](./hamarat_pca_prim.ipynb)
- [`hamarat_cart.ipynb`](./hamarat_cart.ipynb)
- [`hamarat_hhcart.ipynb`](./hamarat_hhcart.ipynb)
- [`comparison_plot.ipynb`](./comparison_plot.ipynb)

---

## Folder Structure

All outputs are saved in the local `data/` directory under subfolders named by algorithm and configuration. Each folder contains saved models and visualisations of trade-offs or splits, along with scenario selection results.

```text
F_testing_on_policy_model/
├── hamarats_*.ipynb                        # Method notebooks
├── comparison_plot.ipynb                   # Final comparison plot
├── notebook_helpers/                       # Utilities for analysis and plotting
│   ├── cart_vis_and_print.py
│   ├── clean_hamarat.py
│   ├── plot_outcomes_lineplot_gaussian.py
│   └── run_hhcart_over_top_features.py
└── data/
    ├── cart/                                   # CART tree visualisations
    ├── prim/                                   # PRIM visualisations
    ├── pca_prim/                               # PCA-PRIM visualisations
    ├── feature_scoring/                        # Feature scoring plot
    ├── comparison/                             # Comparison plots of all methods' coverage-density trade-off for best models
    ├── hhcart_top*/                            # HHCART(D) objects for top features
    └── ...
```