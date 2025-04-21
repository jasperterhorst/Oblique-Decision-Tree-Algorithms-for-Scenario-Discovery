import os
import numpy as np
import pandas as pd
from src.load_shapes import load_shape_dataset

# Load all datasets from fuzziness_000
datasets = load_shape_dataset("fuzziness_000")

# Filter to only 3D datasets
datasets_3d = {k: v for k, v in datasets.items() if "3d" in k.lower()}
print(f"Loaded 3D datasets: {list(datasets_3d.keys())}")

# Dimensions to add
dims_to_add = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]

# Output base directory
output_base = os.path.join("..", "_data", "shapes", "fuzziness_000_increased_dimensionality")
os.makedirs(output_base, exist_ok=True)

# Process each dataset
for name, (X, y) in datasets_3d.items():
    dataset_dir = os.path.join(output_base, name)
    os.makedirs(dataset_dir, exist_ok=True)

    for extra_dims in dims_to_add:
        noise = np.random.uniform(0.0, 1.0, size=(X.shape[0], extra_dims))
        X_aug = np.hstack([X, noise])

        new_name = f"{name}_plus_{extra_dims}dims"
        x_path = os.path.join(dataset_dir, new_name + "_x.csv")
        y_path = os.path.join(dataset_dir, new_name + "_y.csv")

        pd.DataFrame(X_aug).to_csv(x_path, index=False)
        pd.DataFrame(y).to_csv(y_path, index=False)
        print(f"✅ Saved: {x_path}, {y_path}")
