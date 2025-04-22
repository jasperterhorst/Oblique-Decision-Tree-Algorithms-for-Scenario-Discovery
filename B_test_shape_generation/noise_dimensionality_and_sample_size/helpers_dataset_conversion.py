import numpy as np
import os
import pandas as pd
from src.config.paths import DATA_DIR
from src.config.settings import DEFAULT_VARIABLE_SEEDS


from src.config.settings import DEFAULT_VARIABLE_SEEDS
import numpy as np

def add_noise_dimensions(X, num_noise_dims):
    """
    Add a specified number of uniform random noise dimensions to X.
    Noise is generated column by column for reproducibility.

    Parameters:
    -----------
    X : np.ndarray
        Original feature matrix of shape (n_samples, n_features).
    num_noise_dims : int
        Number of noise dimensions to add.

    Returns:
    --------
    X_augmented : np.ndarray
        Feature matrix with (original + noise) dimensions.
    """
    if num_noise_dims <= 0:
        print(f"No noise dimensions added (num_noise_dims={num_noise_dims})")
        return X

    n_samples = X.shape[0]
    seed = DEFAULT_VARIABLE_SEEDS[0]
    rng = np.random.default_rng(seed)

    # Generate noise column by column
    noise_columns = [rng.uniform(0, 1, size=(n_samples, 1)) for _ in range(num_noise_dims)]
    noise_matrix = np.hstack(noise_columns)

    X_augmented = np.hstack([X, noise_matrix])

    print(f"Added {num_noise_dims} noise dimensions: {X.shape[1]} ➔ {X_augmented.shape[1]}")
    return X_augmented


def downsample_dataset(X, y, target_samples):
    """
    Shuffle and randomly downsample X and y to target number of samples using a fixed seed.

    Parameters:
    -----------
    X : np.ndarray
    y : np.ndarray
    target_samples : int

    Returns:
    --------
    X_sampled, y_sampled : np.ndarray, np.ndarray
    """
    n_samples = X.shape[0]
    if target_samples >= n_samples:
        return X, y  # No downsampling needed

    seed = DEFAULT_VARIABLE_SEEDS[0]
    rng = np.random.default_rng(seed)

    # Shuffle indices first
    indices = rng.permutation(n_samples)

    # Select first 'target_samples' after shuffling
    selected_indices = indices[:target_samples]

    return X[selected_indices], y[selected_indices]


def save_dataset(X, y, test_type, shape_name, fuzziness="003", samples=None, dim=None):
    """
    Save dataset (X and y) into the appropriate folder with structured naming.

    Parameters:
    -----------
    X : np.ndarray
    y : np.ndarray
    test_type : str
        Either 'noise_dimensionality' or 'sample_size'.
    shape_name : str
        Shape identifier, e.g., 'radial_segment_2d'.
    fuzziness : str
        Fuzziness level as string, e.g., '003'.
    samples : int
        Number of samples.
    dim : int or None
        Total number of dimensions (only for noise_dimensionality test).
    """
    if test_type == 'noise_dimensionality':
        base_dir = os.path.join(DATA_DIR, "shapes_noise_dimensionality_tests", f"fuzziness_{fuzziness}", shape_name)
        filename = f"{shape_name}_fuzziness_{fuzziness}_{samples}_samples_dim_{str(dim).zfill(2)}.csv"
    elif test_type == 'sample_size':
        base_dir = os.path.join(DATA_DIR, "shapes_sample_size_tests", f"fuzziness_{fuzziness}", shape_name)
        filename = f"{shape_name}_fuzziness_{fuzziness}_{samples}_samples.csv"
    else:
        raise ValueError("Invalid test_type. Use 'noise_dimensionality' or 'sample_size'.")

    os.makedirs(base_dir, exist_ok=True)

    # Save X and y as separate CSVs for compatibility
    df_x = pd.DataFrame(X)
    df_y = pd.DataFrame(y, columns=['label'])

    x_path = os.path.join(base_dir, filename.replace(".csv", "_x.csv"))
    y_path = os.path.join(base_dir, filename.replace(".csv", "_y.csv"))

    df_x.to_csv(x_path, index=False)
    df_y.to_csv(y_path, index=False)

    print(f"Saved: {x_path} and {y_path}")
