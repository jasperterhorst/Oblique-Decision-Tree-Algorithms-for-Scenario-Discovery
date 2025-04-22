import os
import pandas as pd
from src.config.paths import DATA_DIR


def load_shape_dataset(folder_name="shapes", subfolder_name=None):
    """
    Load shape datasets (X and y) from CSV files inside the 'shapes' directory.

    Assumes:
        - Feature files end with '_x.csv'
        - Label files end with '_y.csv'

    Parameters:
    -----------
    subfolder_name : str or None
        Subfolder inside 'shapes'. If None, uses the root shapes directory.

    Returns:
    --------
    datasets : dict
        Mapping from dataset prefix to (X, y) tuples (NumPy arrays).
    """
    base_path = os.path.join(DATA_DIR, folder_name)
    data_dir = os.path.join(base_path, subfolder_name) if subfolder_name else base_path

    files = {}

    if not os.path.exists(data_dir):
        print("ERROR: Data directory not found.")
        return {}

    for root, dirs, filenames in os.walk(data_dir):
        for file in filenames:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path)
                    key = os.path.splitext(file)[0].replace(" ", "_")
                    files[key] = df
                except Exception as e:
                    print(f"Error loading '{file}' from '{root}': {e}")

    # Pair _x with _y
    datasets = {}
    for key in list(files.keys()):
        if key.endswith("_x"):
            prefix = key[:-2]
            key_y = prefix + "_y"
            if key_y in files:
                X = files[key].values
                y = files[key_y].values.squeeze()
                datasets[prefix] = (X, y)

    print(f"\nLoaded {len(datasets)} paired datasets: {list(datasets.keys())}\n")
    return datasets
