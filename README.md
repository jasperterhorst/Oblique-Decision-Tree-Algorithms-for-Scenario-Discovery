# Oblique Decision Trees for Scenario Discovery

This repository implements, benchmarks, and evaluates decision tree algorithms for scenario discovery. It includes both 
axis-aligned methods (PRIM and CART), globally rotated methods (PCA-PRIM), and adapted oblique decision tree algorithms. 
These methods are tested on a set of synthetic benchmark datasets and a high-dimensional European Union energy transition model.

The codebase is organised around distinct phases of the research, with each capturing a specific part of the 
research, from initial demonstrations and shape generation to algorithm benchmarking and policy model application.

Further explanation and the complete research can be found in the accompanying MSc thesis:

Ter Horst, J. T. (2025). Thinking Outside the Box: A Critical Evaluation of Oblique Decision Tree Algorithms for Scenario Discovery. MSc Thesis, TU Delft.
https://repository.tudelft.nl/record/uuid:ae251f06-e4bf-444f-b3f2-c7e1a6f8254f

---

## Python Compatibility

This project is tested and stable under Python 3.11.

---

## Quick Start

To get started locally:

1. Clone the repository:

    ```
    git clone https://github.com/jasperterhorst/Oblique-Decision-Tree-Algorithms-for-Scenario-Discovery.git
    cd Oblique-Decision-Tree-Algorithms-for-Scenario-Discovery
    ```

2. Set up a virtual environment and activate it:

    ```
    python3.11 -m venv .venv
    source .venv/bin/activate       # on Linux/macOS
    .venv\Scripts\activate          # on Windows
    ```

3. Install the required packages:

    ```
    pip install uv
    uv pip install -r requirements.txt
    ```
   
    If you encounter issues with uv pip, you can use the standard pip command:

    ```
    pip install -r requirements.txt
    ```

4. You are set up and can now run all code in this repository.

---

## Repository Modules

The repository is organised into several modules, each serving a specific purpose in the research process. 
Keep in mind each of these separate modules has its own `README_{letter}.md` file with more detailed information.

### A - Scenario Methods Demo
This module provides an interactive demonstration of scenario discovery using three established methods: PRIM, PCA-PRIM, 
and CART. The focus is on visualising how each method generates subspaces on a single synthetic dataset (a rotated 
quadrilateral). The notebook interface allows users to interactively change the shape and inspect subspace coverage and 
density, observe rules that are created, and save outputs automatically. 

### B - Test Shape Generation
This module is responsible for generating a set of synthetic classification problems used throughout the study. 
It includes both 2D and 3D benchmark shapes, such as barbell, radial segment, and rectangle classes. These datasets are 
parametrised by sample size, shape-specific parameters, label noise, and irrelevant dimensional noise, allowing for 
systematic control over the problem. All benchmark datasets follow a consistent naming convention and are stored 
for reuse across benchmarking workflows.

### C - Oblique Decision Tree Benchmark
This module benchmarks a range of oblique decision tree algorithms, such as MOC1, RandCART, RidgeCART, WODT, and HHCART 
variants, on the synthetic benchmark shapes. Each algorithm is tested across increasing tree depth, boundary noise, 
and dimensional noise. The benchmarking system includes batch runners for both local execution and large-scale parallel 
experiments using SLURM on DelftBlue. Output metrics include classification accuracy, subspace coverage and density, 
average number of features used per split, sparsity of decision vectors, and runtime.

### D - Testing HHCART(D) Regularisation
This module investigates how the HHCART(D) algorithm responds to regularisation constraints, specifically the min_purity 
and mass_min parameters. These constraints control when nodes are allowed to split and affect both the number and shape 
of discovered subspaces. The experiments systematically vary these parameters and evaluate the resulting impact on tree 
depth, subspace complexity, and performance metrics. All model outputs are saved using the HHCART_SD format, which includes 
both metric logs and annotated decision trees.

### E - Comparison on Benchmark Shapes
This module visualises and compares the subspaces identified by PRIM, PCA-PRIM, CART, and HHCART(D) on the same benchmark 
shapes. Each method is applied independently, and their results are visualised as overlaid box regions or oblique 
partitions in the input space. The comparison highlights structural differences between axis-aligned and oblique 
approaches in terms of boundary alignment, rule compactness, and coverage–density trade-offs. The module includes shared 
plotting utilities and supports reproducible generation of all comparison figures.

### F - Testing on Policy Model
This module applies the PRIM, PCA-PRIM, CART, and HHCART(D) algorithms to a high-dimensional EU energy transition model 
originally developed by Hamarat et al. (2013) The experiments assess the coverage, density and interpretability of different 
algorithms when applied to realistic policy data. For the HHCART(D), the model uses Extra Trees-based feature importance 
scores to rank input variables and run targeted experiments with top-k feature subsets. The outputs include visualised 
decision trees, subspace diagnostics, and comparative metrics across selected configurations. This module represents 
the final application phase of the research.

### _adopted_oblique_trees – Adapted Algorithm Implementations
This package contains standardised and revised implementations of all oblique decision tree algorithms used in the 
research up to C. Each algorithm has been partly aligned with its original academic formulation and revised where 
necessary to ensure compatibility with the benchmarking framework and for scenario discover (for example in terms of 
runtime tractability (see thesis)). Included models are CART, RandCART, RidgeCART, WODT, MOC1 (modified OC1), and 
both HHCART(A) and HHCART(D). Reproducibility is ensured through consistent use of random_state controls for seeding.

### HHCART_SD – Modular Oblique Tree Framework
This module contains the full implementation of HHCART(D) used from D onwards. It exposes a high-level HHCartD class 
with methods for training, inspection, visualisation, and exporting results. It includes both oblique and axis-aligned 
splitting strategies, supports depth-wise tree construction, and logs key scenario discovery metrics such as coverage 
and density. Visual outputs include decision boundaries, region overlays, trade-off curves, and performance traces 
across tree depths.

### src – Shared Configuration and Utilities
This module provides shared utilities for path handling, plotting configuration, and shape loading. It ensures consistent 
saving, styling, and access to datasets across all other modules.

---


## Outputs and File Types

Output files are saved in the global `_data/` folder (for experiments from A-C) and local `data/` folder (for modules D–F).  
Common formats include
- `.pdf`: For all visualisations
- `.csv`: Tabular outputs for features, metrics, and input data
- `.json`: Metadata about models and parameters
- `.pkl`: Saved decision tree objects in unified format


