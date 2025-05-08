# A - Scenario Methods Demo

This module of my thesis visually explores how three scenario discovery algorithms — PRIM, PCA-PRIM, and CART — isolate regions of interest in a 2D input space. It is useful to show the inherent limitations of these methods when the relation between input variable and the output of interest are not actually axis-aligned.
Furthermore, looking at the plots can help at understanding how different methods trade off coverage with density across iterative refinement steps.

**Author**: Jasper ter Horst  
**Output folder**: All figures and data are saved under `/_data/scenario_methods_demo_outputs`.

## Overview

This component of the project visualises how scenario discovery algorithms iteratively refine decision boundaries to isolate "regions of interest" in the input space.

The notebook generates and saves the following visual outputs:

- **Sample Distribution Plots**: Visualises the original 2D data space and classification labels.
- **Box Evolution Visualisations**: Step-by-step visualisation of PRIM and PCA–PRIM box refinements, both aggregated and per iteration.
- **Peeling Trajectories**: Plots showing trade-offs between coverage and density as boxes evolve.
- **Comparison Figures**: Combined plot comparing peeling trajectories between PRIM and PCA–PRIM.
- **CART Box Output**: Full plot of decision regions learned by the CART algorithm.

## Getting Started

Open [`visualisation_scenario_methods_demo.ipynb`](./visualisation_scenario_methods_demo.ipynb).  
Adjust sliders to generate datasets, run scenario discovery methods, and save results to `/_data/scenario_methods_demo_outputs/`.


## Implementation Highlights

- **CART (via [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html))**: Extracts axis-aligned decision boxes from the input space.
- **Principal Component Analysis (PCA)**: Rotates the input space for PCA-PRIM to perform oblique cuts.
- **Matplotlib patches**: Uses [`Rectangle`](https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html) and [`Polygon`](https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Polygon.html) for plotting regions.
- **Colormap interpolation**: Implements custom `interpolate_color` to create a visual gradient over the box evolution.
- **Flexible plotting abstraction**: All plots are constructed via a shared [`generic_plot`](plotting/base_plots.py) interface.
- **Structured output saving**: All figures and trajectories are exported using consistent folder hierarchies and naming patterns.
- **Notebook display logic**: Inline tables and printouts use [`IPython.display`](https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html) for clarity.
- **Cross-platform paths**: Paths are constructed using [`pathlib`](https://docs.python.org/3/library/pathlib.html) for reliability across operating systems.

## Project Structure

This folder has the following structure:

```text
A_scenario_methods_demo/
│
├── analysis.py                      # Runs full PRIM, PCA-PRIM, and CART pipelines
├── prim_module.py                   # Contains functions for PRIM box selection and metrics
├── pca_rotation_module.py           # Contains PCA rotation and inverse rotation for data and boxes
├── cart_module.py                   # Contains CART training and decision box extraction
├── utils.py                         # Contains color utilities and matplotlib setup
├── notebook_helpers.py              # Runs plot updates and figure saving for all methods
├── __init__.py
├── plotting/                        # Contains plotting modules for PRIM, PCA-PRIM, and CART
│   ├── base_plots.py                # Contains generic plotting and sample scatter logic
│   ├── prim_plots.py                # Contains PRIM and PCA-PRIM box evolution visualisations
│   ├── cart_plots.py                # Contains CART box visualisation logic
│   └── __init__.py
└── visualisation_scenario_methods_demo.ipynb # Interactive notebook for scenario discovery methods
```

Each method (PRIM, PCA–PRIM, CART) saves output plots under:

```text
_data/scenario_methods_demo_outputs/
├── PRIM/
│   ├── evolution/
│   │   ├── box_1.pdf
│   │   ├── box_2.pdf
│   │   └── ...
│   ├── peeling_trajectory_with_constraints.pdf
│   ├── prim_box_evolution.pdf
│   └── prim_peeling_trajectory.pdf
├── PCA_PRIM/
│   ├── evolution/
│   │   ├── box_1.pdf
│   │   ├── box_2.pdf
│   │   └── ...
│   ├── pcaprim_original_box_evolution.pdf
│   ├── pcaprim_peeling_trajectory.pdf
│   ├── pcaprim_rotated_box_evolution.pdf
│   ├── pcaprim_rotated_data.pdf
│   └── peeling_trajectory_with_constraints.pdf
├── CART/
│   └── cart_plot.pdf
├── data_plot.pdf
└── peeling_trajectory_prim_vs_pca_prim.pdf
```

---
