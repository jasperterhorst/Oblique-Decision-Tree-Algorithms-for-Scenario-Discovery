import os
import pandas as pd
from src.config.paths import DATA_DIR


def load_shape_dataset(folder_name=None):
    """
    Loads CSV files from a folder inside the shapes directory. If folder_name is provided, it
    must contain either '_2D' or '_3D' and will be treated as a subfolder of DATA_DIR/shapes.
    If no folder name is provided, the function loads CSV files directly from DATA_DIR/shapes.

    Assumes that each dataset is split into two CSV files:
      - One with suffix '_x' for features
      - One with suffix '_y' for labels

    Parameters:
        folder_name (str, optional): The name of the subfolder inside DATA_DIR/shapes.
                                     If None, the main shapes folder is used.

    Returns:
        datasets (dict): Mapping from dataset prefix to tuple (X, y) as NumPy arrays.
                         Returns an empty dictionary if the folder is invalid or not found.
    """
    base_path = os.path.join(DATA_DIR, "shapes")
    # If a folder name is provided, validate and build the path accordingly.
    if folder_name:
        if not (("_2D" in folder_name) or ("_3D" in folder_name)):
            print(f"ERROR: Provided folder name '{folder_name}' does not contain '_2D' or '_3D'.")
            return {}
        data_dir = os.path.join(base_path, folder_name)
    else:
        data_dir = base_path

    print(f"Starting CSV file loading process from {data_dir}\n")
    files = {}

    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(".csv"):
                file_path = os.path.join(data_dir, file)
                try:
                    df = pd.read_csv(file_path)
                    # Normalize key: remove extension and replace spaces with underscores.
                    key = os.path.splitext(file)[0].replace(" ", "_")
                    files[key] = df
                except Exception as e:
                    print(f"Error loading '{file}' from '{data_dir}': {e}")
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
                y = files[key_y].values.squeeze()  # Reduce to 1D array if possible
                datasets[prefix] = (X, y)
                print(f"Paired dataset '{prefix}': X shape {X.shape}, y shape {y.shape}")

    print(f"\nDatasets available: {list(datasets.keys())}\n")
    return datasets