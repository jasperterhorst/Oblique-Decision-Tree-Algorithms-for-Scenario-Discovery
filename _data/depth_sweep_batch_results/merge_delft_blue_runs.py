import pandas as pd
from pathlib import Path

# 1. Set current directory
current_dir = Path.cwd()

# 2. Loop through all subdirectories
for folder in current_dir.iterdir():
    if folder.is_dir():
        csv_files = sorted(folder.glob("*.csv"))
        if not csv_files:
            print(f"[SKIPPED] No CSV files found in '{folder.name}'")
            continue

        # 3. Read and concatenate
        dfs = [pd.read_csv(f) for f in csv_files]
        merged_df = pd.concat(dfs, ignore_index=True)

        # 4. Save with foldername_concatenated.csv in current dir
        output_filename = f"concatenated_{folder.name}.csv"
        output_path = current_dir / output_filename
        merged_df.to_csv(output_path, index=False)

        print(f"[DONE] Merged {len(csv_files)} CSV files from '{folder.name}' → '{output_filename}' ({len(merged_df)} rows)")
