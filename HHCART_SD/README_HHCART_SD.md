# HHCART Oblique Tree Wrapper

This module provides a Python wrapper for the **HHCART(D) oblique decision tree algorithm** adapted for scenario discovery
and integrating it with standardized dataset and result handling pipelines. It is designed to support reproducible 
experimentation.

**Author**: Jasper ter Horst

---

## Overview

This package allows you to use HHCART(D) via Python or command line. It handles data loading, model training, evaluation, 
and outputting predictions and metadata.

All outputs are generated in a consistent format for each experiment and can be merged automatically after large batch runs.

---


## How to Use the Package

Below is an example of how to use the package:
```python
import pandas as pd
import numpy as np
from HHCART_SD.core import HHCartD

X = pd.DataFrame(np.random.rand(100, 5))
y = np.random.randint(0, 2, size=100)

hh = HHCartD(X, y, max_depth=6, min_purity=0.95)
hh.build_tree()
hh.select(depth=3)
hh.inspect()
hh.plot_tree_structure(depth=3)
```

### Visualization Methods

Once the tree is built, the following methods can be accessed:

- `plot_tree_structure()` – Shows a node structure of a certain tree of depth of choice
- `plot_clipped_boundaries()` – Creates a multi-panel plot with oblique splits at different depths
- `plot_oblique_regions()` – Creates a multi-panel plot with regions classifications at different depths
- `plot_tradeoff_path()` – Creates a 1x1 plot showing the coverage vs. density across tree depths
- `plot_metrics_over_depth()` – Accuracy, coverage, and density
- `plot_node_size_distribution()` – Distribution of samples per tree depth

---

## Project Structure

```
HHCART_SD/
├── core.py                                 # High-level interface (HHCartD) for training, inspecting, and managing oblique decision trees
├── evaluator.py                            # Metric aggregation across depths using the metrics module
├── HHCartDPruning.py                       # Implements tree construction and pruning logic using HouseHolder reflection based oblique splits
├── metrics.py                              # Functions for accuracy, coverage, density, purity, etc.
├── tree.py                                 # Defines core tree and node classes, traversal logic, and split application
├── segmentor.py                            # CARTSegmentor: classic axis-aligned split generator using midpoints
├── split_criteria.py                       # Implementation of Gini impurity and other split evaluation functions
├── metrics.py                              # Standalone metric functions: accuracy, coverage, purity, density, etc.
├── io/
│   └── save_load.py                        # Serialize and load trained HHCartD model objects
├── visualisation/
│   ├── clipped_boundaries.py               # Visualize per-depth oblique splits in the 2D input space
│   ├── coverage_density_path.py            # Trade-off curves of coverage vs density across depths
│   ├── node_size_distribution.py           # Node sample distribution per depth
│   ├── performance_metrics_depthwise.py    # Accuracy, coverage, and density over depth
│   ├── regions.py                          # Visualize per-depth regions and there classification in 2D input space
│   ├── tree_structure.py                   # Graph-style tree structure visualization
│   └── base/
│       ├── save_figure.py                  # Generic export to PDF helper
│       ├── plot_settings.py                # Global styling for matplotlib plots
│       └── colors.py                       # Class palettes and color utilities
```

---

## Output Format

Each run of the HHCART-D algorithm creates a local `data/` directory (if it doesn't exist) in the execution path. 
Inside a folder is created for the respective tree build, this folder can be manually named or is named using the 
current configuration (e.g., `barbell_2d_label_noise_000_depth_8_p_0_95`).

Inside this folder:

- `model/`
  - `trees.pkl` - Serialized tree object for the fitted HHCART-D model
  - `X.csv` - Features used during training
  - `y.csv` - Corresponding binary labels
  - `metrics.csv` - Coverage, density, accuracy, depth, and node count
  - `metadata.json` - Configuration details, parameter values, and run ID

Next to the model folder:

- One or more `.pdf` plots are saved to visualize:
  - The decision boundaries
  - The oblique region structures
  - The trade-off curve between coverage and density
  - Node size distributions
  - Performance trajectories across depths