
# 📂 `_adopted_oblique_trees`

This directory contains adapted and revised implementations of various oblique decision tree algorithms used for 
scenario discovery, with a strong focus on preserving interpretability and aligning with original papers on the algorithms.

These implementations are based on or inspired by the open-source projects 
[Ensembles of Oblique Decision Trees](https://github.com/TorshaMajumder/Ensembles_of_Oblique_Decision_Trees) or
[DecisionTreeBaseline](https://github.com/maoqiangqiang/DecisionTreeBaseline), or created by myself. Each algorithm has 
been evaluated and revised where necessary to ensure consistency with its original academic formulation.

---

## 🧠 Overview of Included Algorithms

| Algorithm       | Original Source                                                                    | Adapted | Notes |
|----------------|------------------------------------------------------------------------------------|---------|-------|
| **CART**        | Self-implemented                                                                   | ✅      | Baseline for comparison |
| **CO2**         | [EODT repo](https://github.com/TorshaMajumder/Ensembles_of_Oblique_Decision_Trees) | ❌      | Included as-is |
| **HouseHolder CART (A & D)** | [EODT repo](https://github.com/TorshaMajumder/Ensembles_of_Oblique_Decision_Trees) | ✅ | Major theoretical & structural adaptations |
| **OC1**         | [EODT repo](https://github.com/TorshaMajumder/Ensembles_of_Oblique_Decision_Trees) | ✅      | Reinstated original Murthy et al. (1994) logic |
| **RandCART**    | [EODT repo](https://github.com/TorshaMajumder/Ensembles_of_Oblique_Decision_Trees) | ✅      | Added reproducibility control |
| **WODT**        | [EODT repo](https://github.com/TorshaMajumder/Ensembles_of_Oblique_Decision_Trees) | 🔁 Minor | Slight restructuring & seeding |
| **RidgeCART**   | [DecisionTreeBaseline repo](https://github.com/maoqiangqiang/DecisionTreeBaseline) | ✅ | Fully restructured and integrated |
| **Segmentor**   | [EODT repo](https://github.com/TorshaMajumder/Ensembles_of_Oblique_Decision_Trees) | ✅      | Rewritten to reflect algorithmic needs |
| **Split Criteria** | [EODT repo](https://github.com/TorshaMajumder/Ensembles_of_Oblique_Decision_Trees) | ✅      | Rewritten and generalized |

---

## 📁 Folder Structure

```
_adopted_oblique_trees/
│
├── CART.py                     # Standard CART tree for benchmarking
├── CO2.py                      # CO2: Classic Oblique Decision Tree
├── HouseHolder_CART.py         # HHCART(A) & HHCART(D)
├── Oblique_Classifier_1.py     # (Legacy or wrapper interface)
├── OC1_tree_structure.py       # Rewritten OC1 with coordinate-wise perturbation
├── RandCART.py                 # Randomized Oblique CART tree
├── RidgeCART.py                # RidgeCART Oblique defrom DecisionTreeBaseline
├── segmentor.py                # Updated segmentor for thresholding
├── split_criteria.py           # Updated impurity calculation methods
├── WODT.py                     # Yang et al.’s WODT with minimal edits
├── README.md                   # This file
└── README_original.md          # Original readme from EODT repo
```

---

## 🔍 Key Adaptations

### 📌 OC1 (Oblique Classifier 1)
- Reinstated coordinate-wise perturbations
- Introduced probabilistic acceptance of non-improving splits
- Global impurity minimization with random restarts
- Bias perturbation enabled via `bias_steps`
- Fully deterministic via local RNG and `random_state`

### 📌 HHCART (A & D)
- Uses **class-specific covariance matrices** (vs PCA)
- Applies **all eigenvectors** as possible reflection axes
- Replaces `MeanSegmentor` with a full `CARTSegmentor`
- Global impurity-minimizing split selection
- Hyperplane construction uses Householder reflection logic

### 📌 RandCART
- Core logic preserved
- Added `random_state` for reproducibility

### 📌 WODT
- Added seeding, minor cleanup
- Algorithmic structure unchanged (Bin-Bin Yang et al., 2019)

### 📌 RidgeCART
- Adapted from external repo
- Full code rewrite for compatibility and readability

---

## 🚫 Excluded Algorithms

**NDT** and **DNDT** were excluded due to their **nonlinear decision boundaries** or **soft partitioning** mechanisms, which fall outside the scope of interpretable linear decision trees.  
(Source: Ittner et al., 1996)

---

## 📚 References

- Murthy et al., 1994 — OC1 Algorithm
- Wickramarachchi et al., 2015 — HHCART
- Yang et al., 2019 — WODT
- Original EODT Repo — [TorshaMajumder/Ensembles_of_Oblique_Decision_Trees](https://github.com/TorshaMajumder/Ensembles_of_Oblique_Decision_Trees)
- RidgeCART — [maoqiangqiang/DecisionTreeBaseline](https://github.com/maoqiangqiang/DecisionTreeBaseline)

---

## 🧪 Reproducibility

Most models have been updated with `random_state` parameters and isolated random generators to ensure reproducibility and experimental consistency.

---

## 🏁 Usage

To use the pipeline:

```bash
python pipeline_for_decision_trees.py --model OC1 --dataset your_data.csv
```

Each model script can be used standalone or integrated via the pipeline for batch evaluation and scenario discovery.

---


---

## 🛠 Detailed Change Logs

### 🔧 OC1 (Murthy et al., 1994)

The original OC1 file from the EODT repository diverged from key components of the OC1 algorithm as described in Murthy et al. (1994). Major revisions include:

- ✅ Replaced global weight vector perturbation with **coordinate-wise updates**.
- ✅ Reinstated **probabilistic acceptance** of non-improving steps via `Pupdate = exp(-k)`.
- ✅ Introduced **random restarts** (`n_restarts`) for global impurity minimization.
- ✅ Added **bias term perturbation** controlled by `bias_steps`.
- ✅ Ensured full **reproducibility** using a seeded `RandomState`.
- ✅ Switched from greedy to **global best-split selection**.

These changes substantially improved impurity minimization behavior and reproducibility.

---

### 🔧 HouseHolder CART-A & CART-D (Wickramarachchi et al., 2015)

- ✅ Replaced PCA-based reflection logic with **class-specific covariance matrices**.
- ✅ Considered **all eigenvectors** per class as reflection axes.
- ✅ Switched from `MeanSegmentor` to `CARTSegmentor` to **enumerate all feature midpoints**.
- ✅ Global selection of best reflection/split pair based on impurity.
- ✅ Corrected decision hyperplane construction using Householder logic.

This brought the implementation closer to the original HHCART formulation and improved geometric consistency.

---

### 🔧 RandCART

- ✅ Preserved the original logic: evaluates 10 random oblique splits, selects the best.
- ✅ Introduced a `random_state` parameter for reproducibility.
- 🔍 Serves as a **stochastic baseline**.

---

### 🔧 WODT (Yang et al., 2019)

- ✅ Minor code restructuring and added **random seed control**.
- ✅ Algorithmic logic left untouched due to prior fidelity to publication.

---

### 🔧 RidgeCART

- ✅ Adapted from [DecisionTreeBaseline](https://github.com/maoqiangqiang/DecisionTreeBaseline).
- ✅ Code fully rewritten for clarity, reproducibility, and pipeline integration.

---

### 🔧 Segmentor & Split Criteria

- ✅ `segmentor.py`: replaced original logic with **CART-style thresholding** (used in HHCART).
- ✅ `split_criteria.py`: updated impurity computation for general use across all algorithms.

---

### ⛔ Skipped: NDT & DNDT

These models use **nonlinear or soft boundaries** via neural network-like architectures. Since this work focuses on **interpretable oblique trees**, they are excluded from evaluation (see Ittner et al., 1996).

---

## 📌 Implementation Notes

- `CARTSegmentor` was introduced and used in place of `MeanSegmentor` in HHCART and other variants.
  This enables more exhaustive and meaningful split searches by checking **all feature midpoints**, which aligns with CART's philosophy and improves decision quality.
- `CART.py` was implemented from scratch to serve as a **baseline** against which all oblique trees can be compared.

---

---

## 📌 Implementation Notes

- `CARTSegmentor` was introduced to replace the simpler `MeanSegmentor`. Unlike the original, which split based on feature means, `CARTSegmentor` tests **all midpoints between sorted feature values**, yielding more accurate and expressive splits. This is especially beneficial in methods like HHCART, where precise thresholding in the reflected space is critical.
  
- `CART.py` was written from scratch to serve as a **standard benchmark** for evaluating the effectiveness of oblique models. It follows traditional axis-aligned split logic and provides a baseline against which oblique decision boundaries can be compared.

---

