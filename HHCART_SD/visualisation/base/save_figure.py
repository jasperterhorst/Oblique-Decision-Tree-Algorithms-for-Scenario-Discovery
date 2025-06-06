"""
Figure Saving Utility (save_figure.py)
--------------------------------------
Provides a standardised method to save Matplotlib figures to disk using the
`save_dir` associated with an HHCartD model object.

Features:
- Saves plots in PDF format.
- Filename can be user-defined.
- Can be toggled via a `save` boolean flag.
- Raises informative error if `save_dir` is not available.

Example:
    from save_figure import save_figure
    save_figure(hh, filename="decision_boundary.pdf", save=True)
"""

from pathlib import Path
import matplotlib.pyplot as plt


def save_figure(hh, filename: str, save: bool = False) -> None:
    """
    Save the current Matplotlib figure to the model's output folder as a PDF.

    Args:
        hh (HHCartD): The trained decision tree wrapper containing `save_dir`.
        filename (str): Name of the output file (must end with `.pdf`).
        save (bool, optional): If True, the figure is saved. Defaults to False.

    Raises:
        ValueError: If `save` is True but `hh.save_dir` is not defined.
        ValueError: If `filename` does not end with `.pdf`.
    """
    if not save:
        return

    if hh.save_dir is None:
        raise ValueError("Cannot save figure: `hh.save_dir` is not set. Did you build or load the model?")

    if not filename.lower().endswith(".pdf"):
        raise ValueError("Filename must end with '.pdf'.")

    save_path = Path(hh.save_dir) / filename

    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"[SAVED] Figure saved to: {save_path}")
