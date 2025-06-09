import pandas as pd
import numpy as np
from ema_workbench import load_results


def load_and_prepare_hamarat_results(results_path, drop_switch=True):
    """
    Load and clean the Hamarat et al. 10000 LHS results.

    Args:
        results_path (Path or str): Path to the .tar.gz results archive.
        drop_switch (bool): Whether to drop SWITCH categorical variables after encoding.

    Returns:
        experiments_raw (DataFrame): Original experiments DataFrame before dropping columns.
        outcomes (dict): Outcomes dictionary as returned by load_results.
        X (DataFrame): Cleaned input DataFrame with dummy variables (and optionally SWITCH columns removed).
        y (np.ndarray): Final year (2100) values of fraction_renewables.
    """
    # Load data
    experiments, outcomes = load_results(results_path)
    experiments_raw = experiments.copy()

    # Correct column name typo
    if "SWTICH_preference_carbon_curve" in experiments.columns:
        experiments = experiments.rename(columns={"SWTICH_preference_carbon_curve": "SWITCH_preference_carbon_curve"})

    # Drop non-input columns
    columns_to_drop = ["model", "policy", "scenario", "year"]
    columns_to_drop = [col for col in columns_to_drop if col in experiments.columns]
    X_df = experiments.drop(columns=columns_to_drop)

    # Categorical encoding for SWITCH columns
    categorical_cols = [col for col in X_df.columns if col.lower().startswith("switch")]
    X_df[categorical_cols] = X_df[categorical_cols].astype("category")
    X_encoded = pd.get_dummies(X_df, drop_first=True)

    # Optionally drop all SWITCH variables after encoding
    if drop_switch:
        switch_cols = [col for col in X_encoded.columns if col.startswith("SWITCH")]
        X_encoded = X_encoded.drop(columns=switch_cols)

    # Target y = fraction_renewables at final year (2100)
    fraction_renewables_over_time = outcomes["fraction_renewables"]
    y = fraction_renewables_over_time[:, -1]

    # Final cleaned input is called X
    X = X_encoded

    return experiments_raw, outcomes, X, y
