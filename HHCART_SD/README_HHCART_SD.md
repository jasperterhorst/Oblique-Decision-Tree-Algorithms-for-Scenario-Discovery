
# HHCART_SD: Oblique Decision Trees for Scenario Discovery

## Project Description

**HHCART_SD** is a customized implementation of oblique decision trees, tailored specifically for scenario discovery applications. It is based on the original HHCART(D) method proposed by Wickramarachchi et al. (2016), which introduces the use of Householder transformations for constructing oblique hyperplane splits. The base algorithm was sourced from Majumder's open-source implementation, and then extensively adapted and extended to try to enable interpretable scenario discovery.

### Key Extensions

- **Minimum Purity Threshold**: Introduced a `min_purity` parameter to control node splitting and overfitting, improving interpretability.
- **Depth-wise Saving and Selection**: Stores all trained trees by depth, allowing inspection and structured comparison.
- **Scenario Discovery Metrics**: Integrated coverage and density metrics to evaluate decision regions.
- **Custom Visualizations**: Includes various types of plots, including trade-off plot and multi-panel plots for visualising evolution of oblique splits.

## How to Use the Package

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

## Features

- Oblique tree building with Householder reflections
- Scenario-relevant metrics and thresholds
- Structured training and saving at multiple depths
- Integrated model evaluation and plotting

## Folder Structure
```
HHCART_SD/
├── core.py                   # High-level interface (HHCartD) for training, inspecting, and managing oblique decision trees
├── evaluator.py              # Metric aggregation across depths using the metrics module
├── HHCartDPruning.py         # Implements tree construction and pruning logic using HouseHolder reflection based oblique splits
├── metrics.py                # Functions for accuracy, coverage, density, purity, etc.
├── tree.py                   # Defines core tree and node classes, traversal logic, and split application
├── segmentor.py              # CARTSegmentor: classic axis-aligned split generator using midpoints
├── split_criteria.py         # Implementation of Gini impurity and other split evaluation functions
├── metrics.py                # Standalone metric functions: accuracy, coverage, purity, density, etc.
├── io/
│   └── save_load.py          # Serialize and load trained HHCartD model objects
├── visualisation/
│   ├── clipped_boundaries.py               # Visualize per-depth oblique splits in the 2D input space
│   ├── coverage_density_path.py            # Trade-off curves of coverage vs density across depths
│   ├── node_size_distribution.py           # Node sample distribution per depth
│   ├── performance_metrics_depthwise.py    # Accuracy, coverage, and density over depth
│   ├── regions.py                          # Visualize per-depth regions and there classification in 2D input space
│   ├── tree_structure.py                   # Graph-style tree structure visualization
│   └── base/
│       ├── save_figure.py           # Generic export to PDF helper
│       ├── plot_settings.py         # Global styling for matplotlib plots
│       └── colors.py                # Class palettes and color utilities
```

## Credits

**Jasper ter Horst** – Developed and adapted the HHCART_SD package for scenario discovery, as part of a thesis project at Delft University of Technology.

Based on original research and software by:
> Wickramarachchi, D. C., Robertson, B. L., Reale, M., Price, C. J., & Brown, J. (2016). HHCART: An oblique decision tree. *Computational Statistics & Data Analysis*, 96, 12–23. https://doi.org/10.1016/j.csda.2015.11.006  
> Majumder, T. (2020b). *Ensembles of Oblique Decision Trees*. GitHub repository. https://github.com/TorshaMajumder/Ensembles-of-Oblique-Decision-Trees

## License
⚠️ This project adapts code originally authored by Torsha Majumder, who did not specify a license. The reuse is solely for academic purposes and no commercial redistribution is permitted. If you are the original author and would like your code removed or properly licensed, please contact me.