"""
Batch Shape Generation Script (2D & 3D) with Configurable Fuzziness Levels

This script provides a fast, non-interactive way to generate and save many synthetic
2D and 3D shapes with varying label noise ("fuzziness"). It is intended for use as a batch runner
to complement the interactive shape generation interface.

Functionality:
    - Iterates over multiple predefined shapes and fuzziness levels.
    - Automatically generates labeled samples using shape parameters.
    - Saves the resulting datasets (features and labels) and visualizations (PDF plots) to disk.
    - Organizes output into folders by fuzziness level and shape name.

Shapes:
    - 2D: Rectangle, Radial Segment, Barbell, Sine Wave, Star
    - 3D: Saddle Surface, Radial Segment, Barbell

Fuzziness:
    - Label fuzziness simulates classification uncertainty by flipping a percentage of labels.
    - Settable via `fuzz_levels`, applied to all shapes in batch.
    - Output is organized under the `SHAPES_DIR` path using a suffix based on fuzziness percentage.

Use Case:
    - Rapid dataset generation for model training, testing, and benchmarking.
    - Supports reproducibility and consistent output structure.
    - Designed to run independently of user interaction.

This script is intended to be run as a batch utility when many shape instances are necessary.
"""

# === Imports ===
from B_test_shape_generation.shape_generators import (
    generate_2d_rectangle, generate_2d_radial_segment, generate_2d_barbell,
    generate_2d_sine_wave, generate_2d_star,
    generate_3d_saddle, generate_3d_radial_segment, generate_3d_barbell
)
from B_test_shape_generation.utils import plot_2d_shape, plot_3d_shape, save_data
from src.config.paths import SHAPES_DIR

# === Configuration ===
fuzz_levels = [0.00, 0.03, 0.05, 0.07]
num_samples_2d = 5000
num_samples_3d = 10000

# === Shape Definitions ===
shape_configs = {
    "rectangle_2d": {
        "generator": generate_2d_rectangle,
        "params": {
            "num_samples": num_samples_2d,
            "ribs": (0.5, 0.5),
            "center": (0.5, 0.5),
            "rotation": 45
        }
    },
    "radial_segment_2d": {
        "generator": generate_2d_radial_segment,
        "params": {
            "num_samples": num_samples_2d,
            "center": (0.5, 0.5),
            "outer_radius": 0.4,
            "inner_radius": 0.2,
            "arc_span_degrees": 300,
            "rotation": 90
        }
    },
    "barbell_2d": {
        "generator": generate_2d_barbell,
        "params": {
            "num_samples": num_samples_2d,
            "center": (0.5, 0.5),
            "barbell_length": 0.6,
            "sphere_radius": 0.2,
            "connector_thickness": 0.04,
            "rotation": 50
        }
    },
    "sine_wave_2d": {
        "generator": generate_2d_sine_wave,
        "params": {
            "num_samples": num_samples_2d,
            "x_range": (0.1, 0.9),
            "vertical_offset": 0.5,
            "amplitude": 0.2,
            "frequency": 0.5,
            "thickness": 0.10,
            "rotation": 0
        }
    },
    "star_2d": {
        "generator": generate_2d_star,
        "params": {
            "num_samples": num_samples_2d,
            "center": (0.5, 0.5),
            "num_points": 5,
            "star_size": 1.0,
            "outer_radius": 0.4,
            "inner_radius": 0.2,
            "rotation": 0
        }
    },
    "saddle_3d": {
        "generator": generate_3d_saddle,
        "params": {
            "num_samples": num_samples_3d,
            "center": (0.5, 0.5, 0.5),
            "saddle_height": 0.5,
            "curve_sharpness_x1": 1.0,
            "curve_sharpness_x2": 1.0,
            "surface_thickness": 0.2,
            "rotate_x1_deg": 0,
            "rotate_x2_deg": 0,
            "rotate_x3_deg": 0
        }
    },
    "radial_segment_3d": {
        "generator": generate_3d_radial_segment,
        "params": {
            "num_samples": num_samples_3d,
            "center": (0.5, 0.5, 0.5),
            "outer_radius": 0.4,
            "inner_radius": 0.2,
            "arc_span_degrees": 300,
            "rotation_x1": 35,
            "rotation_x2": 0,
            "rotation_x3": 60
        }
    },
    "barbell_3d": {
        "generator": generate_3d_barbell,
        "params": {
            "num_samples": num_samples_3d,
            "center": (0.5, 0.5, 0.5),
            "barbell_length": 0.8,
            "sphere_radius": 0.25,
            "connector_thickness": 0.1,
            "rotation_angle_x1": 50,
            "rotation_angle_x2": 50,
            "rotation_angle_x3": 0
        }
    }
}

# === Generation Loop ===
for fuzz in fuzz_levels:
    suffix = f"fuzziness_{int(fuzz * 100):03d}"
    out_dir = SHAPES_DIR / suffix
    out_dir.mkdir(exist_ok=True)

    for shape_name, config in shape_configs.items():
        generator = config["generator"]
        params = config["params"].copy()
        # noinspection PyTypeChecker
        params["fuzziness"] = fuzz

        # Generate data
        df_x, y, samples = generator(**params)

        # Define output directory and filename
        shape_folder = out_dir / f"{shape_name}_{suffix}"
        shape_folder.mkdir(parents=True, exist_ok=True)
        full_prefix = f"{shape_name}_{suffix}"

        # Save data to disk
        save_data(df_x, y, file_prefix=full_prefix, save_dir=shape_folder)

        # Prepare plot annotations
        plot_title = shape_name.replace("_", " ").title()
        note = ", ".join(f"{k}: {v}" for k, v in params.items())
        note_formatted = note.replace("_", " ").title()

        # Generate and save plot
        if "2d" in shape_name:
            plot_2d_shape(samples, y, title=plot_title, save_path=shape_folder / f"{full_prefix}.pdf",
                          note=note_formatted)
        else:
            plot_3d_shape(samples, y, title=plot_title, save_path=shape_folder / f"{full_prefix}.pdf",
                          note=note_formatted)
