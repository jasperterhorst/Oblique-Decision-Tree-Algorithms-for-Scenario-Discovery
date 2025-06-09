"""
Hamarat Scenario Data Cleaner (load_hamarat_results.py)
------------------------------------------------------
Cleans scenario discovery input data from Hamarat et al. (2013) model from EMA workbench.

Provides cleaned features DataFrame and target outcome for final year.
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

    Args:
        experiments (pd.DataFrame): Raw experiments DataFrame.
        outcomes (dict): Outcomes dictionary as returned by `load_results`.
        drop_switch (bool): Whether to drop SWITCH variables after encoding.

    Returns:
        X (pd.DataFrame): Cleaned input DataFrame with dummy variables.
        y (np.ndarray): Final year (2100) values of fraction_renewables.
    """
    # --- Clean experiments ---
    # Correct column name typo
    if "SWTICH_preference_carbon_curve" in experiments.columns:
        experiments = experiments.rename(
            columns={"SWTICH_preference_carbon_curve": "SWITCH_preference_carbon_curve"}
        )

    # Drop non-input columns
    columns_to_drop = ["model", "policy", "scenario", "year"]
    columns_to_drop = [col for col in columns_to_drop if col in experiments.columns]
    X_df = experiments.drop(columns=columns_to_drop)

    # Categorical encoding for SWITCH columns (do NOT create dummies yet)
    categorical_cols = [col for col in X_df.columns if col.lower().startswith("switch")]
    X_df[categorical_cols] = X_df[categorical_cols].astype("category")

    # --- Clean column names ---
    X_df = X_df.rename(columns=lambda col: " ".join(word.capitalize() for word in col.replace("_", " ").split()))

    # Optionally drop SWITCH variables entirely
    if drop_switch:
        X_df = X_df.drop(columns=categorical_cols)

    # --- Extract target variable y from outcomes ---
    fraction_renewables_over_time = outcomes.get("fraction_renewables", None)
    if fraction_renewables_over_time is None or len(fraction_renewables_over_time.shape) != 2:
        raise ValueError(
            "Expected outcomes['fraction_renewables'] to be a 2D array with time steps."
        )

    y = fraction_renewables_over_time[:, -1]

    return X_df, y
