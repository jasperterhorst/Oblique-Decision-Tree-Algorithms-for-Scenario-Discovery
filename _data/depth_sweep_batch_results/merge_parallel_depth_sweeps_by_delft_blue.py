import pandas as pd
import glob
import os

# Find all relevant CSV files in the current directory
csv_files = sorted(glob.glob("depth_sweep_*.csv"))

if not csv_files:
    raise FileNotFoundError("No files matching 'depth_sweep_*.csv' found.")

# Merge them
dfs = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(dfs, ignore_index=True)

# Define preferred column order
preferred_order = [
    'dataset', 'data_dim', 'seed', 'algorithm', 'depth', 'splits', 'leaves',
    'accuracy', 'coverage', 'density', 'f_score', 'training_time',
    'avg_active_feature_count', 'feature_utilisation_ratio',
    'tree_level_sparsity_index', 'composite_interpretability_score'
]

# Retain only the columns that exist in the merged DataFrame
final_order = [col for col in preferred_order if col in merged_df.columns]
merged_df = merged_df[final_order]

# Save the merged and reordered result
output_file = "depth_sweep_multiple_seeds.csv"
merged_df.to_csv(output_file, index=False)

print(f"Merged {len(csv_files)} files into '{output_file}' with {len(merged_df)} rows.")
