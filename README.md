# Oblique Decision Tree Algorithms for Scenario Discovery

Welcome to the repository for **Oblique Decision Tree Algorithms for Scenario Discovery**. This project presents a 
modular, extensible framework for experimenting with **oblique decision trees**, particularly for use in **scenario 
discovery** and **decision-making under deep uncertainty**.

The repository contains a diverse set of models, tools for synthetic data generation, visualization notebooks, 
and benchmarks. It also integrates and builds upon existing work in the field, most notably ensemble-based oblique 
decision trees.

---

## 📦 Repository Overview

The repository is organized into modular folders. Each serves a distinct role in supporting experimentation, algorithm 
development, and scenario discovery workflows:

### `A_scenario_methods_demo/` – 📊 Scenario Discovery Methods
Implements and demonstrates the use of scenario discovery techniques:
- **PRIM** (Patient Rule Induction Method)
- **PCA-PRIM** (a PCA-enhanced version of PRIM)
- **CART** (Classification and Regression Trees)

Includes plotting utilities and helper functions to visualize boxes and peeling trajectories. This module is ideal 
mainly used for visualization of scenario discovery techniques and their limitations in the thesis.

---

### `B_test_shape_generation/` – 🧪 Shape Generator
A tool to generate **synthetic 2D and 3D geometric shapes** like bars, radial segments, and stars. These synthetic 
datasets are used to:
- Test model robustness
- Provide interpretable visual feedback
- Simulate scenario boundaries

Includes an **interactive Jupyter notebook** interface to explore and modify shape characteristics.

---

### `C_test_shape_experiments/` – 🧬 Shape-based Experiments
Builds on synthetic shapes from module B to:
- Run scenario discovery algorithms
- Validate tree models on well-defined decision boundaries
- Compare algorithmic behavior on complex vs. simple scenarios

This is where theory meets controlled benchmarking.

---

### `C_oblique_decision_trees/` – 🌲 Core Tree Implementations
Implements several types of **oblique decision trees**, such as:
- **OC1**
- **Householder CART (HHCART)**
- **Randomized CART (RandCART)**
- **Weighted Oblique Decision Trees (WODT)**
- **CO2 Trees**

These are ran and using converters put into a standard tree structure. So that the different trees can be compared
evaluated and ran against each other on different datasets.


[//]: # (### `E_TAO_algorithm/` – 🔧 TAO Optimization)

[//]: # (Implements **Tree Alternating Optimization &#40;TAO&#41;** for training oblique decision trees using:)

[//]: # (- Custom loss functions)

[//]: # (- Regularizers)

[//]: # (- Gradient-based optimization logic)

[//]: # ()
[//]: # (Provides a notebook to showcase how TAO improves oblique split quality over traditional methods.)

---

### `Ensembles_of_Oblique_Decision_Trees/` – 🤝 External Module Integration
This folder integrates the work from **[Ensembles of Oblique Decision Trees](https://github.com/jasperterhorst/Oblique-Decision-Tree-Algorithms-for-Scenario-Discovery/tree/main/Ensembles_of_Oblique_Decision_Trees)** 
by Torsha Majumder.

It includes:
- Implementations of ensemble methods for oblique decision trees (OC1, HHCART, RandCART, WODT, CO2 Trees)
- Several classic datasets like *breast cancer*, *diabetes*, *glass*, *vehicle*, etc.

> 🧾 **Citation:**  
> Majumder, T. (2020). *Ensembles of Oblique Decision Trees* [Master's Thesis, University of Texas, Dallas]. UTD Theses and Dissertations.

---

### `_data/` – 📁 Datasets and Outputs
This folder includes:
- Generated 2D/3D shape files (CSV and PDF)
- Model outputs and evaluation results
- Pickled tree models
- Visual output figures (e.g., box evolution, peeling trajectories)

Organized by experiment or shape type.

---

### `notebooks/` – 📓 Notebooks & Demos
Jupyter notebooks that walk through:
- Scenario method visualizations
- Interactive shape generation
- Model behavior explanations

These are perfect for learning, teaching, and sharing your findings.

---

### `src/` – ⚙️ Utilities and Configuration
Contains:
- Configuration settings (paths, colors, plotting)
- Data loading utilities

Used across the entire project for clean code reuse.

---

### `requirements.txt`
Defines the required Python dependencies. Install with:

pip install -r requirements.txt


---

## 🚀 Quick Start

To get started:

1. Clone the repository:
    ```
    git clone https://github.com/jasperterhorst/Oblique-Decision-Tree-Algorithms-for-Scenario-Discovery.git
    cd Oblique-Decision-Tree-Algorithms-for-Scenario-Discovery
    ```

2. Install the dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Launch notebooks from the `notebooks/` directory:

## 📬 Contact

Questions, feedback, or collaboration ideas?  
Open an issue or connect via GitHub.

