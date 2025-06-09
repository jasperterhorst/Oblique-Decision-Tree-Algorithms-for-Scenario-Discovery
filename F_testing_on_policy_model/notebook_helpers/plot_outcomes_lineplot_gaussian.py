"""
Scenario Discovery Visualisation Utilities (plot_utils.py)
----------------------------------------------------------
Provides visualisation functions for scenario discovery analysis using Hamarat et al.
energy transition model outputs.

Includes:

- plot_fraction_renewables(): plots renewables fraction trajectories and final-year KDE.

Intended for use in exploratory analysis, model inspection, and reporting.
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from src.config.colors_and_plot_styles import PRIMARY_DARK, PRIMARY_LIGHT
from src.config.plot_settings import apply_global_plot_settings, beautify_plot


def plot_fraction_renewables(outcomes: dict, threshold: float = None, save_path: str = None) -> None:
    """
    Plot fraction of renewables over time, with optional threshold-based colouring.

    Applies global plot settings and beautifies the plot using standard utilities.

    Args:
        outcomes (dict): Outcomes dictionary as returned by load_results. Must contain 'fraction_renewables'.
        threshold (float or None): Optional threshold. If provided, runs ending above threshold will be plotted in red,
        others in grey.
        save_path (str or None): If provided, saves the figure to this path (PDF recommended).
    """
    # Apply global plot settings first
    apply_global_plot_settings()

    # Extract fraction_renewables
    fraction_renewables_over_time = outcomes.get("fraction_renewables", None)
    if fraction_renewables_over_time is None:
        raise ValueError("Outcomes does not contain 'fraction_renewables'. Cannot plot.")

    # Validate that it is a 2D array
    if not isinstance(fraction_renewables_over_time, np.ndarray) or fraction_renewables_over_time.ndim != 2:
        raise ValueError("Expected outcomes['fraction_renewables'] to be a 2D array (n_runs, n_timesteps).")

    # Time axis
    time_resolution = 0.0625
    start_year = 2010
    end_year = 2050
    n_timesteps = fraction_renewables_over_time.shape[1]
    years = np.linspace(start_year, end_year, n_timesteps)

    # Final value per run
    final_renewable_share = fraction_renewables_over_time[:, -1]

    # Plot main figure
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=100)

    for i in range(fraction_renewables_over_time.shape[0]):
        color = PRIMARY_DARK if threshold is not None and final_renewable_share[i] > threshold else PRIMARY_LIGHT
        ax.plot(years, fraction_renewables_over_time[i], lw=0.4, alpha=0.5, color=color)

    # Force main y-axis and x-axis limits
    ax.set_ylim(0, 1)
    ax.set_xlim(start_year, end_year)

    # KDE inset
    kde = gaussian_kde(final_renewable_share)
    y_vals = np.linspace(0, 1, 300)
    density = kde(y_vals)

    # Create inset axes with better spacing
    kde_ax = ax.inset_axes([1.15, 0.0, 0.08, 1.0])
    kde_ax.plot(density, y_vals, color=PRIMARY_LIGHT, lw=1, alpha=0.8)
    kde_ax.set_xticks([])
    kde_ax.set_ylabel("Fraction in 2100", fontsize=15)
    kde_ax.tick_params(axis='y', labelsize=12)
    kde_ax.set_xlim(0, np.max(density) * 1.05)

    # Set spines to grey
    for spine in kde_ax.spines.values():
        spine.set_color('grey')

    # Set tick marks and tick labels to grey
    kde_ax.tick_params(axis='y', colors='grey')

    # Force KDE inset y-axis to match main y-axis
    kde_ax.set_ylim(ax.get_ylim())

    # Optional: show threshold line on inset
    if threshold is not None:
        kde_ax.axhline(threshold, color=PRIMARY_DARK, linestyle='--', linewidth=1.5)

    # Use your beautify_plot function
    beautify_plot(
        ax,
        title="Fraction of Renewables Over Time",
        xlabel="Year",
        ylabel="Fraction",
        save_path=save_path
    )

    # Print summary if threshold is given
    if threshold is not None:
        n_total = len(final_renewable_share)
        n_above = np.sum(final_renewable_share > threshold)
        n_below = n_total - n_above

        pct_above = n_above / n_total * 100
        pct_below = n_below / n_total * 100

        threshold_pct = threshold * 100

        print(f"\nAt threshold the threshold of {threshold_pct:.1f}% renewables by {end_year}:\n"
              f"{pct_above:.1f}% of cases exceed threshold ({n_above} of {n_total}), "
              f"{pct_below:.1f}% of cases fall below threshold ({n_below} of {n_total})\n")
