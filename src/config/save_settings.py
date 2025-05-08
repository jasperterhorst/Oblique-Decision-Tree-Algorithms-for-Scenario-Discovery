"""
save_settings.py

Utility functions for saving figures and data files in a consistent,
cross-platform way. Handles subdirectories, file naming, and full
integration with central project paths using pathlib.
"""

from typing import Literal, Union
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from src.config.paths import DATA_DIR


def save_figure(
    fig: plt.Figure,
    filename: str,
    base_path: Path = DATA_DIR,
    subfolder: str = "",
    subsubfolder: str = "",
    dpi: int = 300,
    filetype: Literal["pdf", "png", "svg"] = "pdf"
) -> str:
    """
    Save a Matplotlib figure to a specified folder.

    Parameters:
        fig (plt.Figure): The figure object to save.
        filename (str): File name without extension.
        base_path (Path): Base directory (e.g., DATA_DIR or other configured folder).
        subfolder (str): Optional subdirectory inside base_path (e.g., 'plots').
        subsubfolder (str): Optional nested subdirectory inside subfolder.
        dpi (int): DPI for the saved figure.
        filetype (str): Output format: 'pdf', 'png', or 'svg'.

    Returns:
        str: Full path to the saved figure.
    """
    save_dir = base_path
    if subfolder:
        save_dir /= subfolder
    if subsubfolder:
        save_dir /= subsubfolder
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{filename}.{filetype}"
    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    print(f"[FIGURE SAVED] {save_path}")
    return str(save_path)


def save_dataframe(
    df: pd.DataFrame,
    filename: str,
    base_path: Path = DATA_DIR,
    subfolder: str = "",
    subsubfolder: str = "",
    index: bool = False
) -> str:
    """
    Save a DataFrame to CSV under a specified folder.

    Parameters:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): File name without '.csv' extension.
        base_path (Path): Base directory (e.g., DATA_DIR or other configured folder).
        subfolder (str): Optional subdirectory inside base_path.
        subsubfolder (str): Optional nested subdirectory.
        index (bool): Whether to save the DataFrame index.

    Returns:
        str: Full path to the saved CSV file.
    """
    save_dir = base_path
    if subfolder:
        save_dir /= subfolder
    if subsubfolder:
        save_dir /= subsubfolder
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{filename}.csv"
    df.to_csv(save_path, index=index)
    print(f"[CSV SAVED] {save_path}")
    return str(save_path)
