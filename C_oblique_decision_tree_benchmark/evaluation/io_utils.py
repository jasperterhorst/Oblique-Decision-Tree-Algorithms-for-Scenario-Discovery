import os
import pickle
import pandas as pd
from src.config.paths import (
    DEPTH_SWEEP_SINGLE_RUN_RESULTS_OUTPUTS_DIR,
    DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR
)

# =============================================================================
# CONFIGURATION HELPERS
# =============================================================================


def get_output_dir(run_type="single", subdir=None):
    """
    Get the output directory for a given run type.

    Args:
        run_type (str): "single" or "batch".
        subdir (str, optional): Optional subfolder inside the run type directory.

    Returns:
        str: Full path to the output directory.
    """
    base = DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR if run_type == "batch" else DEPTH_SWEEP_SINGLE_RUN_RESULTS_OUTPUTS_DIR
    full_path = os.path.join(base, subdir) if subdir else base
    os.makedirs(full_path, exist_ok=True)
    return full_path

# =============================================================================
# TREE DICTIONARY I/O
# =============================================================================


def save_trees_dict(trees_dict, filename="moc1_barbell_2d_no_noise.pkl", run_type="single", subdir=None):
    """
    Save the trees dictionary to a pickle file.

    Args:
        trees_dict (dict): Dictionary of trained decision trees.
        filename (str): Output filename (should end in .pkl).
        run_type (str): "single" or "batch".
        subdir (str, optional): Optional subdirectory inside the output directory.
    """
    output_dir = get_output_dir(run_type, subdir)
    file_path = os.path.join(output_dir, filename)

    with open(file_path, "wb") as f:
        pickle.dump(trees_dict, f)

    print(f"[OK] Saved trees_dict to: {file_path}")


def load_trees_dict(filename="moc1_barbell_2d_no_noise.pkl", run_type="single", subdir=None):
    """
    Load the trees dictionary from a pickle file.

    Args:
        filename (str): Name of the file to load.
        run_type (str): "single" or "batch".
        subdir (str, optional): Subdirectory inside the output folder.

    Returns:
        dict: Loaded trees dictionary.
    """
    input_dir = get_output_dir(run_type, subdir)
    file_path = os.path.join(input_dir, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[✗] No tree file found at: {file_path}")

    with open(file_path, "rb") as f:
        trees_dict = pickle.load(f)

    print(f"[OK] Loaded trees_dict from: {file_path}")

    return trees_dict

# =============================================================================
# DEPTH SWEEP RESULTS I/O
# =============================================================================


def save_depth_sweep_df(df, filename="depth_sweep_result.csv", run_type="single", subdir=None):
    """
    Save a DataFrame of depth sweep results to CSV.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): Filename for the CSV (should end in .csv).
        run_type (str): "single" or "batch".
        subdir (str, optional): Optional subfolder.
    """
    output_dir = get_output_dir(run_type, subdir)
    file_path = os.path.join(output_dir, filename)

    df.to_csv(file_path, index=False)
    print(f"[OK] Saved DataFrame to: {file_path}")


def load_depth_sweep_df(filename="depth_sweep_result.csv", run_type="single", subdir=None):
    """
    Load a saved CSV file of depth sweep results.

    Args:
        filename (str): Filename of the CSV.
        run_type (str): "single" or "batch".
        subdir (str, optional): Optional subdirectory.

    Returns:
        pd.DataFrame: Loaded results DataFrame.
    """
    input_dir = get_output_dir(run_type, subdir)
    file_path = os.path.join(input_dir, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[✗] No CSV file found at: {file_path}")

    df = pd.read_csv(file_path)
    print(f"[OK] Loaded DataFrame from: {file_path}")

    return df
