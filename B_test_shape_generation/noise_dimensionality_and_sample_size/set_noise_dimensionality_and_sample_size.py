from helpers_dataset_conversion import add_noise_dimensions, downsample_dataset, save_dataset
from src.load_shapes import load_shape_dataset

# 1. Load all datasets from fuzziness_003
all_datasets = load_shape_dataset(folder_name="fuzziness_003")

# 2. Filter for target shapes
target_shapes = ['radial_segment_3d_fuzziness_003']

for shape_key in target_shapes:
    X, y = all_datasets[shape_key]

    # Extract base shape name without the fuzziness suffix for cleaner saving
    base_shape_name = shape_key.replace('_fuzziness_003', '')

    # === Noise Dimensionality Test ===
    for num_noise_dims in [1, 3, 5, 10, 15, 25]:  # Number of noise dimensions to ADD
        X_augmented = add_noise_dimensions(X, num_noise_dims)
        total_dims = X_augmented.shape[1]
        save_dataset(X_augmented, y, test_type='noise_dimensionality',
                     shape_name=base_shape_name, samples=10000, dim=total_dims)

    # === Sample Size Test ===
    for sample_size in [1000, 5000, 10000]:
        X_sampled, y_sampled = downsample_dataset(X, y, sample_size)
        save_dataset(X_sampled, y_sampled, test_type='sample_size',
                     shape_name=base_shape_name, samples=sample_size)
