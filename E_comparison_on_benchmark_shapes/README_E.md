# E - Comparison of PRIM, PCA-PRIM, CART, and HHCART(D) on Benchmark Shapes

This module compares four scenario discovery techniques, **PRIM**, **PCA-PRIM**, **CART**, and **HHCART(D)**, applied 
to benchmark shapes. It focuses on the trade-off between coverage, density, and visual interpretability by extracting 
and overlaying decision regions.

---

## Key Scripts

- `extract_prim_boxes.py`: Extracts PRIM boxes with coverage, density, and polygon corner coordinates.
- `extract_pca_prim_boxes.py`: Transforms and extracts PCA-PRIM boxes with coverage, density, and polygon corner coordinates.
- `extract_cart_boxes.py`: Extracts CART boxes predicting class 1 with corner coordinates.
- `box_plotter.py`: Visualises 2D benchmark data with overlaid PRIM, CART, and PCA-PRIM boxes.
- `box_utils.py`: Provides geometric utilities for computing box corners and plotting inputs.
- `pca_preprocess_local.py`: Performs PCA with mean and standard deviation tracking for accurate inverse transformation in `extract_pca_prim_boxes.py`.

---

## Folder Structure

```text
E_comparison_on_benchmark_shapes/
├── cart/
│   ├── cart_boxes_on_barbell_2d.pdf
│   └── cart_boxes_on_rectangle_2d.pdf
├── prim/
│   ├── prim_boxes_on_barbell_2d.pdf
│   └── prim_boxes_on_rectangle_2d.pdf
├── pca_prim/
│   ├── pca_prim_boxes_on_barbell_2d.pdf
│   └── pca_prim_boxes_on_rectangle_2d.pdf
├── hhcart_d/
│   ├── data/
│   ├── hhcart_d_on_2d_barbell.ipynb
│   └── hhcart_d_on_2d_rectangle.ipynb
├── visualisation_prim_cart_pca_prim/
│   └── (combined overlay plots for comparison of PRIM, PCA-PRIM, and CART,
│       see key scripts before this section.)
```