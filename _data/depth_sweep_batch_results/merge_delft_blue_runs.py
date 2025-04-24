import pandas as pd
from pathlib import Path

# 1. Define current and input dirs
current_dir = Path.cwd()
input_dir = current_dir / "delftblue_dimensionality_runs"

# 2. Gather CSVs
csv_files = sorted(input_dir.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in '{input_dir}'.")

# 3. Read & concat
dfs = [pd.read_csv(f) for f in csv_files]
merged_df = pd.concat(dfs, ignore_index=True)

# 4. Reorder columns if they exist
preferred_order = [
    "seed", "dataset", "data_dim", "algorithm", "depth", "accuracy", "coverage",
    "density", "f_score", "gini_coverage_all_leaves", "gini_density_all_leaves",
    "splits", "leaves", "runtime"
]
final_order = [c for c in preferred_order if c in merged_df.columns]
merged_df = merged_df[final_order]

# 5. Save to current dir
output_path = current_dir / "all_algorithms_all_datasets.csv"
merged_df.to_csv(output_path, index=False)

print(f"Merged {len(csv_files)} CSV files from '{input_dir}' into '{output_path}' ({len(merged_df)} rows).")
