# C - Oblique Decision Tree Benchmark

This module provides a unified benchmarking framework for comparing oblique decision trees with the axis-aligned CART 
method on synthetic classification datasets created in [B_test_shape_generation](../B_test_shape_generation). 
It automates execution, metric evaluation, and visualization for various algorithms under controlled experimental 
conditions such as label noise, sample size, and input dimensionality.

The module is designed for scalability, offering both local execution and SLURM-based parallelization on high-performance 
computing clusters like **DelftBlue**.

---

## Overview

The goal of this module is to systematically evaluate how different tree-based classifiers perform on noisy, 
geometric classification problems. This includes both **axis-aligned** (e.g., CART) and **oblique** 
(e.g., MOC1, WODT, RidgeCART, HHCART(A), HHCART(D), RandCART) decision trees.

Three types of benchmark sweeps are supported:
- **Boundary/Label Noise Sweep**: Evaluates model robustness as labels are flipped near shape boundaries.
- **Dimensionality Sweep**: Adds increasing numbers of irrelevant features to test how trees handle high-dimensional noise.
- **Sample Size Sweep**: Tests how performance scales with the amount of training data.

---

## Recommended Workflow

1. **Run Benchmarks on DelftBlue**
   Navigate to [`running_on_delftblue`](./running_on_delftblue/) folder and subsequent subfolders and submit SLURM jobs for each benchmark type using the appropriate scripts.
   - Install packages Python server you are using (see main README.md file of this project).
   
2. **Merge Results Locally**
   After jobs complete, use [`merge_delft_blue_runs.py`](../_data/depth_sweep_batch_results/merge_delft_blue_runs.py) 
   in the `_data` folder to merge the output CSVs into one file per benchmark type.

3. **Visualize the Results**
   Run the plotting functions in the notebooks or use the scripts in the `visualization/` folder. The plots will be saved 
   under `depth_sweep_batch_results/plots/`.

---

## Project Structure

```text
C_oblique_decision_tree_benchmark/
├── README_C.md
├── __init__.py
├── converters/                                     # Wrappers and format converters for tree algorithms (CART, OC1, etc.)
│   ├── __init__.py
│   ├── cart_converter.py
│   ├── co2_converter.py
│   ├── dispatcher.py
│   ├── hhcart_converter.py
│   ├── oc1_converter.py
│   ├── randcart_converter.py
│   ├── ridgecart_converter.py
│   └── wodt_converter.py
├── core/                                           # Core oblique decision tree structures and logic
│   ├── __init__.py
│   └── tree.py
├── evaluation/                                     # Tools for running benchmarks and computing evaluation metrics
│   ├── __init__.py
│   ├── benchmark_runner.py
│   ├── evaluator.py
│   ├── io_utils.py
│   └── metrics.py
├── notebooks/                                      # Jupyter notebooks for analyzing and visualizing results
│   ├── batch_runner_evaluation_notebook.ipynb
│   └── evaluate_single_run.ipynb
├── running_on_delftblue/                           # SLURM scripts for DelftBlue job submission
│   ├── __init__.py
│   ├── dimensionality_runs/
│   │   ├── __init__.py
│   │   ├── generate_job_list_dimensionality.py
│   │   ├── job_list_dimensionality.txt
│   │   └── run_dimensionality_array.sh
│   ├── label_noise_runs/
│   │   ├── __init__.py
│   │   ├── generate_job_list_label_noise.py
│   │   ├── job_list_label_noise.txt
│   │   ├── logs/
│   │   └── run_label_noise_array.sh
│   ├── sample_size_runs/
│   │   ├── __init__.py
│   │   ├── generate_job_list_sample_size.py
│   │   ├── job_list_sample_size.txt
│   │   ├── logs/
│   │   └── run_sample_size_array.sh
│   ├── local_batch_runner.py
│   ├── run_depth_sweep_benchmark.py
│   └── run_depth_sweep_parallel.sh
└── visualization/                                  # Plotting utilities for single run and batch result visualizations
    ├── __init__.py
    ├── batch_results_plots.py
    └── single_run_plots.py
```

---

## Output Folder Structure

Benchmark results are saved under the [`../_data/depth_sweep_batch_results/`](../_data/depth_sweep_batch_results/) directory. 

This directory contains all outputs from SLURM-based batch runs on DelftBlue, it contains:

- **Benchmark CSVs**: Each folder inside [`depth_sweep_batch_results`](../_data/depth_sweep_batch_results/)-folder
(e.g. `delftblue_sample_size_runs/`) contains individual `.csv` files for every run (per dataset, model, seed, etc.)  
  File format example:  
  `radial_segment_3d_label_noise_003_10000_samples_dim_03_hhcart_d_seed0.csv`

- **Merged Result Files**:  
  The `merge_delft_blue_runs.py` script consolidates each folder’s runs into a single CSV:
  - `concatenated_delftblue_sample_size_runs.csv`
  - `concatenated_delftblue_label_noise_runs.csv`
  - `concatenated_delftblue_dimensionality_runs.csv`

- **Plots Folder**:  
  Visualizations from `visualization/` scripts (e.g. performance over depth, runtime analysis) are saved in the `plots/` subdirectory, grouped by plot type.

These outputs are automatically created after running the SLURM jobs using the batch runners in `running_on_delftblue/`. 
The merging and visualization steps can then be executed locally via notebooks or Python scripts.

```text
└── depth_sweep_batch_results/
    ├── delftblue_label_noise_runs/                     
    ├── delftblue_sample_size_runs/
    ├── delftblue_dimensionality_runs/
    ├── concatenated_delftblue_label_noise_runs.csv
    ├── concatenated_delftblue_sample_size_runs.csv
    ├── concatenated_delftblue_dimensionality_runs.csv
    ├── merge_delft_blue_runs.py
    └── plots/                             
        ├── depth_trajectories/
        ├── performance_by_label_noise/
        ├── runtime_by_n_samples/
        ├── runtime_by_data_dim/
        ├── active_features_by_data_dim/
        └── plot_over_depth/
```
