import os
import pandas as pd
from src.config.paths import DATA_DIR


def load_shape_dataset(folder_name=None):
    """
    Loads CSV files from a folder (and its subfolders) inside the shapes directory.
    If a folder name is provided, it will be treated as a subfolder of DATA_DIR/shapes.
    If no folder name is provided, the function loads CSV files directly from DATA_DIR/shapes.

    Assumes that each dataset is split into two CSV files:
      - One with suffix '_x' for features
      - One with suffix '_y' for labels

    Parameters:
        folder_name (str, optional): The name of the subfolder inside DATA_DIR/shapes.
                                     If None, the main shapes folder is used.

    Returns:
        datasets (dict): Mapping from dataset prefix to tuple (X, y) as NumPy arrays.
                         Returns an empty dictionary if no CSV files are found.
    """
    base_path = os.path.join(DATA_DIR, "shapes")
    if folder_name:
        data_dir = os.path.join(base_path, folder_name)
    else:
        data_dir = base_path

    # print(f"Starting CSV file loading process from {data_dir}\n")
    files = {}

    # Walk through the directory and its subdirectories
    if os.path.exists(data_dir):
        for root, dirs, filenames in os.walk(data_dir):
            for file in filenames:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    try:
                        df = pd.read_csv(file_path)
                        # Normalize key: remove extension and replace spaces with underscores.
                        key = os.path.splitext(file)[0].replace(" ", "_")
                        files[key] = df
                        # print(f"Loaded file: {file_path}")
                    except Exception as e:
                        print(f"Error loading '{file}' from '{root}': {e}")
    else:
        print("ERROR: Data directory not found.")
        return {}

    # Pair files ending in '_x' with their corresponding '_y' files.
    datasets = {}
    for key in list(files.keys()):
        if key.endswith("_x"):
            prefix = key[:-2]  # Remove the trailing "_x"
            key_y = prefix + "_y"
            if key_y in files:
                X = files[key].values
                y = files[key_y].values.squeeze()  # Squeeze to reduce to 1D array if possible
                datasets[prefix] = (X, y)
                # print(f"Paired dataset '{prefix}': X shape {X.shape}, y shape {y.shape}")

    print(f"\nPaired Datasets available are: {list(datasets.keys())}\n")
    return datasets
