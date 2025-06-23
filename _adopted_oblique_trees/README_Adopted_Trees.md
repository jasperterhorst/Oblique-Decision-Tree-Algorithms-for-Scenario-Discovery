# `_adopted_oblique_trees`

This directory contains adapted and revised implementations of various oblique decision tree algorithms used for 
scenario discovery, with a strong focus on preserving interpretability, aligning with original papers and keeping runtime
tractable.

These implementations are based on or inspired by the open-source projects 
[Ensembles of Oblique Decision Trees](https://github.com/TorshaMajumder/Ensembles_of_Oblique_Decision_Trees) or
[DecisionTreeBaseline](https://github.com/maoqiangqiang/DecisionTreeBaseline), or created by myself. Each algorithm has 
been evaluated and revised where necessary to ensure consistency with its original academic formulation.

---

## Overview of Included Algorithms and Changes

The table below shows an overview of included algorihtmms, their original sources, whether they have been adapted, and any notes on the changes made.


For an elaborate overview see appendix of Ter Horst (2025).

| Algorithm              | Original Source                                                               | Adapted | Notes                                                                                                                          |
|------------------------|-------------------------------------------------------------------------------|---------|--------------------------------------------------------------------------------------------------------------------------------|
| **CART**               | Self-implemented                                                              | ✅      | Axis-aligned baseline algorithm for comparison.                                                                                |
| **RandCART**           | [EODT](https://github.com/TorshaMajumder/Ensembles_of_Oblique_Decision_Trees) | 🔁 Minor      | Oblique baseline algorithm, slight restructuring for framework compatibility & added reproducibility control.                  |
| **HHCART (A & D)**     | [EODT](https://github.com/TorshaMajumder/Ensembles_of_Oblique_Decision_Trees) | ✅ | Major theoretical & structural adaptations to align with Wickramarachchi et al. (2016).                                        |
| **OC1**                | [EODT](https://github.com/TorshaMajumder/Ensembles_of_Oblique_Decision_Trees) | ✅      | Adapted for runtime tractability and alignment with OC1 by Murthy et al. (1994), creating the MOC1 variant used in the thesis. |
| **WODT**               | [EODT](https://github.com/TorshaMajumder/Ensembles_of_Oblique_Decision_Trees) | 🔁 Minor | Light restructuring for framework compatibility & added reproducibility control.                                               |
| **RidgeCART**          | [DecisionTreeBaseline](https://github.com/maoqiangqiang/DecisionTreeBaseline) | ✅ | Fundamentally adapted from a regression to a classification model to suit the scenario discovery task.                         |

---

## 📁 Folder Structure

```
_adopted_oblique_trees/
│
├── CART.py                              # Implementation of a standard Classification and Regression Tree (CART).
├── HouseHolder_CART.py                  # Implementation of HHCART(A) and HHCART(D) based on Wickramarachchi et al.
├── Modified_Oblique_Classifier_1.py     # Implementation of the MOC1 algorithm used in the thesis, adapted for performance.
├── RandCART.py                          # Implementation of a randomized oblique CART using random projections.
├── RidgeCART.py                         # Implementation of RidgeCART, adapted for classification from the DecisionTreeBaseline repo.
├── WODT.py                              # Implementation of the Weighted Oblique Decision Tree (WODT).
├── OC1_tree_structure.py                # Tree structure used by MOC1 algorithm
├── segmentor.py                         # Defines thresholding strategies, including the exhaustive CARTSegmentor.
├── split_criteria.py                    # Defines impurity calculation methods (e.g., Gini Index, Entropy).
```

## References
- Torsha Majumder — [Ensembles_of_Oblique_Decision_Trees](https://github.com/TorshaMajumder/Ensembles_of_Oblique_Decision_Trees)
- Qiangqiang Mao — [DecisionTreeBaseline](https://github.com/maoqiangqiang/DecisionTreeBaseline)
- Murthy et al. (1994) — OC1 Algorithm
- Wickramarachchi et al. (2015) — HHCART
- Yang et al., 2019 — WODT
- Ter Horst (2025) — Thinking Outside the Box: A Critical Evaluation of Oblique Decision Tree Algorithms for Scenario Discovery

---

## Reproducibility

Most models have been updated with `random_state` parameters and isolated random generators to ensure reproducibility and experimental consistency.

---