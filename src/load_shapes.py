import os
import pandas as pd


def load_all_shape_datasets():
    """
    Loads all CSV files from the '../_data/shapes' folder.
    Assumes that each dataset is split into two files: one with suffix '_x' for features
    and one with suffix '_y' for labels.

    Returns:
        datasets (dict): Mapping from dataset prefix to tuple (X, y) as NumPy arrays.
    """
    data_dir = os.path.join("..", "_data", "shapes")
    print("Starting CSV file loading process from '_data/shapes'...")
    print(f"Data directory set to: {data_dir}\n")
    files = {}

    if os.path.exists(data_dir):
        for subfolder in os.listdir(data_dir):
            subfolder_path = os.path.join(data_dir, subfolder)
            if os.path.isdir(subfolder_path):
                for file in os.listdir(subfolder_path):
                    if file.endswith(".csv"):
                        file_path = os.path.join(subfolder_path, file)
                        try:
                            df = pd.read_csv(file_path)
                            # Store using the file name without extension as key
                            key = os.path.splitext(file)[0].replace(" ", "_")
                            files[key] = df
                            print(f"Loaded '{file}' as '{key}' with shape {df.shape}")
                        except Exception as e:
                            print(f"Error loading '{file}' from '{subfolder_path}': {e}")
    else:
        print("ERROR: Data directory not found.")

    print("\nCSV file loading complete.")

    # Pair files that share the same prefix (before the last underscore)
    datasets = {}
    keys = list(files.keys())
    paired = set()
    for key in keys:
        if key.endswith("_x"):
            prefix = key[:-2]  # remove "_x"
            key_y = prefix + "_y"
            if key_y in files:
                X = files[key].values
                y = files[key_y].values.squeeze()  # ensure y is 1D if possible
                datasets[prefix] = (X, y)
                paired.add(key)
                paired.add(key_y)
                print(f"Paired dataset '{prefix}': X shape {X.shape}, y shape {y.shape}")

    print(f"\nDatasets available: {list(datasets.keys())}\n")
    return datasets