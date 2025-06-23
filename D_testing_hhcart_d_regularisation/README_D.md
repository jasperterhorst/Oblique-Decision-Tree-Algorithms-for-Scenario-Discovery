# D - HHCART(D) Regularisation Testing

This module evaluates how the **HHCART(D)** oblique decision tree responds to variations in the `min_purity` and 
`mass_min` regularisation parameters. The goal is to analyse how these constraints affect subspace complexity and 
scenario interpretability and is tested on the benchmark shapes.

Main notebook: [`HHCART_D_testing_min_purity_and_samples.ipynb`](./HHCART_D_testing_min_purity_and_samples.ipynb)

---

## Folder Structure

All outputs are saved in the local `data/` directory in this folder under subfolders named by configuration. 
Each folder contains visualisations of coverage–density trade-offs and oblique splits and the saved model.

```text
D_testing_hhcart_d_regularisation/
├── HHCART_D_testing_min_purity_and_samples.ipynb
├── README_D.md
└── data/
    ├── barbell_2d_label_noise_000_depth_8_p_0.9_s_2/
    │   ├── model/
    │   │   ├── metadata.json
    │   │   ├── metrics.csv
    │   │   ├── trees.pkl
    │   │   ├── X.csv
    │   │   └── y.csv
    │   ├── cvd_cls1_*.pdf
    │   ├── cvd_depth_*.pdf
    │   └── ...
    ├── barbell_2d_label_noise_000_depth_8_p_0.95_s_2/
    └── ...
```