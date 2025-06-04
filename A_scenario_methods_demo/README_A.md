# A - Scenario Discovery Methods Demo

This module from my thesis visually explores how three scenario discovery algorithms - **PRIM**, **PCA-PRIM**, and **CART** - identify and isolate regions of interest in a synthetic 2D input space. It is designed to demonstrate how these methods behave when the relationship between inputs and the output of interest is **not axis-aligned**, which is a known limitation of many traditional scenario discovery techniques.

By visualizing their outputs, users can see how each algorithm **trades off coverage vs. density** and observe how decision boundaries evolve over **iterative refinement steps**.

**Author**: Jasper ter Horst  

---

## Overview

This module is part of a larger integration framework and is mainly accessed via the notebook:

📓 [visualisation_scenario_methods_demo.ipynb](./visualisation_scenario_methods_demo.ipynb)

The notebook allows you to:
- **Generate synthetic 2D datasets** with specific inclusion zones
- **Run PRIM, PCA-PRIM, and CART analyses**
- **Save visual outputs** of each method's results

---

## Key Visual Outputs

All output figures are saved in [`../_data/scenario_methods_demo_outputs/`](../_data/scenario_methods_demo_outputs/). These plots visually explain how the methods work and evolve:

- **Data Distribution Plot**  
  Displays the 2D synthetic dataset with regions labeled as “of interest” or “not of interest”.

- **PRIM & PCA-PRIM Box Evolution**  
  Step-by-step visualizations of how each method refines its decision boxes. Includes both original and PCA-rotated views for PCA-PRIM.

- **PRIM & PCA-PRIM Peeling Trajectories**  
  Plots showing the trade-off between **coverage** and **density** over successive box refinements. Includes a constraint-aware variant and a comparative PRIM vs. PCA-PRIM plot.

- **CART Decision Regions**  
  Final classification map produced by the CART decision tree, showing axis-aligned partitions in the input space.

---

## Getting Started
For instructions on how to set up the environment, see [README.md](../README.md) in the root of the project.

Open the notebook using:

```bash
jupyter notebook A_visualisation_scenario_methods_demo.ipynb
```

Then:
1. Use the sliders to define the region of interest.
2. Adjust parameters like **peel fraction** and **mass thresholds**.
3. Click buttons to generate and save plots.

**All results are saved under** [`../_data/scenario_methods_demo_outputs/`](../_data/scenario_methods_demo_outputs/).

---

## Project Structure

```text
A_scenario_methods_demo/
│
├── analysis.py                             # Runs PRIM, PCA-PRIM, and CART experiments from start to finish
├── prim_module.py](./prim_module.py)       # Core PRIM logic
├── pca_rotation_module.py                  # PCA transformation handling
├── cart_module.py                          # CART classifier setup
├── utils.py                                # Colormap helpers, utilities
├── notebook_helpers.py                     # Widget + save logic
├── __init__.py  
│                           
├── plotting/                               # Visualization Logic by Method
│   ├── base_plots.py                       # Base plotting routines: generic plot layout and sample distribution
│   ├── prim_plots.py                       # PRIM and PCA-PRIM specific plotting
│   ├── cart_plots.py                       # CART-specific plotting
│   └── __init__.py
└── visualisation_scenario_methods_demo.ipynb
```

---

## Output Folder Structure

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
│
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
│
├── CART/
│   └── cart_plot.pdf
│
├── data_plot.pdf
└── peeling_trajectory_prim_vs_pca_prim.pdf
```

---