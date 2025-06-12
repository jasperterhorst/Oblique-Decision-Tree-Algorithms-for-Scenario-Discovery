"""
Hamarat Scenario Data Cleaner (load_hamarat_results.py)
------------------------------------------------------
Cleans scenario discovery input data from the Hamarat et al. (2013) energy transition model
as processed through the EMA Workbench.

Provides:
- Cleaned features DataFrame (X)
- Target outcome array (y) containing final-year renewables fraction.
"""

import pandas as pd
import numpy as np


def clean_results(
    experiments: pd.DataFrame,
    outcomes: dict,
    drop_switch: bool = False
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Clean experiments and outcomes from Hamarat LHS results.

    The experiments DataFrame is cleaned and formatted:
    - Column names are capitalised.
    - SWITCH variables are encoded as categorical (unless drop_switch=True).

    The target variable y is extracted from the final year of 'fraction_renewables'.

    Args:
        experiments (pd.DataFrame):
            Raw experiments DataFrame.
        outcomes (dict):
            Outcomes dictionary as returned by load_results().
        drop_switch (bool):
            If True, drops SWITCH variables entirely after processing.

    Returns:
        tuple:
            X (pd.DataFrame): Cleaned input DataFrame with appropriate encoding.
            y (np.ndarray): Final-year renewables fraction per run.

    Side effects:
        None
    """
    # --- Clean experiments DataFrame ---
    # Correct known column name typo if present
    if "SWTICH_preference_carbon_curve" in experiments.columns:
        experiments = experiments.rename(
            columns={"SWTICH_preference_carbon_curve": "SWITCH_preference_carbon_curve"}
        )

    # Drop non-input columns if present
    columns_to_drop = ["model", "policy", "scenario", "year"]
    columns_to_drop = [col for col in columns_to_drop if col in experiments.columns]
    X_df = experiments.drop(columns=columns_to_drop)

    # Identify SWITCH columns (categorical inputs)
    categorical_cols = [col for col in X_df.columns if col.lower().startswith("switch")]
    X_df[categorical_cols] = X_df[categorical_cols].astype("category")

    # --- Clean column names ---
    X_df = X_df.rename(columns=lambda col: " ".join(word.capitalize() for word in col.replace("_", " ").split()))

    # Optionally drop SWITCH variables entirely
    if drop_switch:
        X_df = X_df.drop(columns=categorical_cols)

    # --- Extract target variable y ---
    fraction_renewables_over_time = outcomes.get("fraction_renewables", None)
    if fraction_renewables_over_time is None:
        raise ValueError("Outcomes dictionary does not contain 'fraction_renewables'.")

    if not isinstance(fraction_renewables_over_time, np.ndarray) or fraction_renewables_over_time.ndim != 2:
        raise ValueError("Expected outcomes['fraction_renewables'] to be a 2D array (n_runs, n_timesteps).")

    # Target y = renewables fraction at final year (last time step)
    y = fraction_renewables_over_time[:, -1]

    return X_df, y
