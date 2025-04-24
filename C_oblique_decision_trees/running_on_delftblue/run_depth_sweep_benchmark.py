"""
run_depth_sweep_benchmark.py

This script executes a depth sweep benchmark for a specified dataset and model.
It is a general-purpose runner used across different experiment types, including:
- Shape baseline sweeps
- Dimensionality increase tests
- Sample size variation tests

Datasets are loaded from configured DATA_DIR paths, and results are saved to
organised subfolders within the batch results directory.
"""

import argparse
import os
import time
import sys

# Add project root to the Python path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)

from src.load_shapes import load_shape_dataset
from C_oblique_decision_trees.evaluation.benchmark_runner import DepthSweepRunner
from src.config.settings import DEFAULT_VARIABLE_SEEDS

# Set number of threads (default to 1 if not specified by SLURM)
os.environ["OMP_NUM_THREADS"] = os.environ.get("SLURM_CPUS_PER_TASK", "1")


def main():
    start_time = time.time()
    print(">>> Starting Depth Sweep Benchmark")
    print(f"OMP_NUM_THREADS set to: {os.environ['OMP_NUM_THREADS']}")

    parser = argparse.ArgumentParser(description="Run depth sweep benchmark on a dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset prefix to run")
    parser.add_argument("--folder-name", type=str, required=True, help="Main folder inside DATA_DIR")
    parser.add_argument("--subfolder-name", type=str, required=True, help="Subfolder path within folder-name")
    parser.add_argument("--seed-index", type=int, required=True, help="Index into DEFAULT_VARIABLE_SEEDS")
    parser.add_argument("--model", type=str, required=True,
                        choices=["hhcart_a", "hhcart_d", "randcart", "oc1", "wodt", "co2", "cart", "ridge_cart"],
                        help="Model to benchmark")
    parser.add_argument("--max-depth", type=int, default=12, help="Maximum tree depth")
    parser.add_argument("--output-subfolder", type=str, default="delftblue", help="Output subfolder")
    parser.add_argument("--output-filename", type=str, required=True, help="Output CSV filename")

    args = parser.parse_args()
    print(f"Arguments: {args}")

    print("Loading dataset...")
    all_data = load_shape_dataset(folder_name=args.folder_name, subfolder_name=args.subfolder_name)
    filtered_data = {k: v for k, v in all_data.items() if k.startswith(args.dataset)}
    if not filtered_data:
        raise ValueError(f"No datasets found starting with '{args.dataset}'")
    print(f"Datasets loaded: {list(filtered_data.keys())}")

    seed = DEFAULT_VARIABLE_SEEDS[args.seed_index]
    print(f"Using seed: {seed}")

    print("Building model registry...")
    registry = {args.model: DepthSweepRunner.build_registry(random_state=seed)[args.model]}

    print("Running depth sweep benchmark...")
    runner = DepthSweepRunner(datasets=filtered_data, max_depth=args.max_depth)
    runner.run(
        auto_export=True,
        filename=args.output_filename,
        tree_dict_filename=args.output_filename.replace(".csv", ".pkl"),
        n_seeds=1,
        fixed_seed=seed,
        registry=registry,
        return_trees=True,
        save_tree_dict=False,
        batch_mode=True,
        output_subfolder=args.output_subfolder
    )

    elapsed = time.time() - start_time
    print(f"[✓] Completed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()


# #!/usr/bin/env python3
# """
# run_depth_sweep_benchmark.py
#
# Runs a depth sweep benchmark on a single dataset.
# Loads the dataset from a specified subfolder (if provided) inside DATA_DIR/shapes.
# The results are saved into a corresponding subfolder under DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR.
# """
#
# import argparse
# import os
# import time
# import sys
#
# # Add project root to the Python path.
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# sys.path.insert(0, project_root)
#
# from src.load_shapes import load_shape_dataset
# from C_oblique_decision_trees.evaluation.benchmark_runner import DepthSweepRunner
# from src.config.settings import DEFAULT_VARIABLE_SEEDS
#
# # Set the number of threads (from SLURM or default to 1)
# os.environ["OMP_NUM_THREADS"] = os.environ.get("SLURM_CPUS_PER_TASK", "1")
#
# start_time = time.time()
# print(">>> Starting Depth Sweep Script")
# print(f"OMP_NUM_THREADS set to: {os.environ['OMP_NUM_THREADS']}")
#
# parser = argparse.ArgumentParser(
#     description="Run depth sweep for a selected dataset, seed, and model."
# )
# parser.add_argument("--dataset", type=str, required=True,
#                     help="Specify one dataset to run")
# parser.add_argument("--folder-name", type=str, required=True,
#                     help="Main folder inside DATA_DIR (e.g., shapes, shapes_noise_dimensionality_tests)")
# parser.add_argument("--subfolder-name", type=str, required=True,
#                     help="Subfolder path within folder-name (e.g., fuzziness_003/radial_segment_3d)")
# parser.add_argument("--seed-index", type=int, required=True,
#                     help="Index into DEFAULT_VARIABLE_SEEDS")
# parser.add_argument("--model", type=str, required=True,
#                     choices=["hhcart_a", "hhcart_d", "randcart", "oc1", "wodt", "co2", "cart", "ridge_cart"],
#                     help="Model to use")
# parser.add_argument("--max-depth", type=int, default=12,
#                     help="Maximum depth to sweep over")
# parser.add_argument("--output-subfolder", type=str, default="delftblue",
#                     help="Subfolder where outputs will be saved (default: delftblue)")
# parser.add_argument("--output-filename", type=str, required=True,
#                     help="Output CSV filename")
# args = parser.parse_args()
# print(f"Arguments received: {args}")
#
# print("Loading datasets...")
# all_data = load_shape_dataset(folder_name=args.folder_name, subfolder_name=args.subfolder_name)
# filtered_data = {k: v for k, v in all_data.items() if k.startswith(args.dataset)}
# if not filtered_data:
#     raise ValueError(f"No datasets found starting with '{args.dataset}'.")
# all_datasets_dict = filtered_data
# print(f"Datasets selected: {list(all_datasets_dict.keys())}")
#
# try:
#     seed = DEFAULT_VARIABLE_SEEDS[args.seed_index]
# except IndexError:
#     raise ValueError(f"Seed index {args.seed_index} out of range.")
# print(f"Using seed: {seed}")
#
# # Build model-specific registry
# print("Preparing model registry...")
# full_registry = DepthSweepRunner.build_registry(random_state=seed)
# registry = {args.model: full_registry[args.model]}
#
# print("Initializing DepthSweepRunner...")
# runner = DepthSweepRunner(datasets=all_datasets_dict, max_depth=args.max_depth)
#
# print("Running benchmark sweep...")
# depth_sweep_df, _ = runner.run(
#     auto_export=True,
#     filename=args.output_filename,
#     tree_dict_filename=args.output_filename.replace(".csv", ".pkl"),
#     n_seeds=1,
#     fixed_seed=seed,
#     registry=registry,
#     return_trees=True,
#     save_tree_dict=False,
#     batch_mode=True,
#     output_subfolder=args.output_subfolder
# )
#
# elapsed = time.time() - start_time
# print(f"[✓] Script completed in {elapsed:.2f} seconds")
#
# #     output_subfolder=args.folder
