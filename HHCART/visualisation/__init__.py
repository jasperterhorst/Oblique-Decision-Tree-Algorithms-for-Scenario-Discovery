from .coverage_density_path import plot_coverage_density_tradeoff_path
from .clipped_boundaries import plot_clipped_oblique_splits as _plot_clipped_boundaries


def bind_plotting_methods(obj):
    """
    Dynamically bind visualisation methods to an HHCartD object instance.

    Adds the following methods to the instance:
    - plot_tradeoff(save_path=None): Coverage and density vs depth (separate lines)
    - plot_tradeoff_path(save_path=None): Coverage–density trajectory (X=coverage, Y=density)
    - plot_clipped_boundaries(save=False, filename=None, title=None): Region plots per depth

    Args:
        obj: An instance of HHCartD.

    Raises:
        AttributeError: If required attributes are missing on the object.
    """
    required_attrs = ["trees_by_depth", "X", "y"]
    for attr in required_attrs:
        if not hasattr(obj, attr):
            raise AttributeError(f"The object must have attribute '{attr}'.")

    def plot_tradeoff_path(save: bool = False, filename: str = None, title: str = None):
        """
        Plot the sequential coverage–density trade-off path across depths.

        Args:
            save (bool): Whether to save the figure.
            filename (str, optional): Output filename (PDF).
            title (str, optional): Optional figure title.
        """
        return plot_coverage_density_tradeoff_path(obj, save=save, filename=filename, title=title)

    def plot_clipped_boundaries(save: bool = False, filename: str = None, title: str = None):
        """
        Plot clipped oblique decision boundaries for all trained depths.

        Args:
            save (bool): Whether to save the figure.
            filename (str, optional): Output filename (PDF).
            title (str, optional): Optional figure title.
        """
        return _plot_clipped_boundaries(obj, save=save, filename=filename, title=title)

    obj.plot_tradeoff_path = plot_tradeoff_path
    obj.plot_clipped_boundaries = plot_clipped_boundaries
