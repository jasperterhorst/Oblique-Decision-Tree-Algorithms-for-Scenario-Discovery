#!/usr/bin/env python3
"""
run_depth_sweep_parallel.py

Runs a depth sweep benchmark on a single dataset.
Loads the dataset from a specified subfolder (if provided) inside DATA_DIR/shapes.
The results are saved into a corresponding subfolder under DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR.
"""

import argparse
import os
import time
import sys

# Add project root to the Python path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)

from src.load_shapes import load_shape_dataset
from D_oblique_decision_trees.evaluation.benchmark_runner import DepthSweepRunner
from src.config.settings import DEFAULT_VARIABLE_SEEDS

# Set the number of threads (from SLURM or default to 1)
os.environ["OMP_NUM_THREADS"] = os.environ.get("SLURM_CPUS_PER_TASK", "1")

start_time = time.time()
print(">>> Starting Depth Sweep Script")
print(f"OMP_NUM_THREADS set to: {os.environ['OMP_NUM_THREADS']}")

parser = argparse.ArgumentParser(
    description="Run depth sweep for a selected dataset, seed, and model."
)
parser.add_argument("--dataset", type=str, required=True,
                    help="Specify one dataset to run")
parser.add_argument("--folder", type=str, required=True,
                    help="Subfolder inside DATA_DIR/shapes to load (e.g., no_noise, 5_percent_noise, etc.)")
parser.add_argument("--seed-index", type=int, required=True,
                    help="Index into DEFAULT_VARIABLE_SEEDS")
parser.add_argument("--model", type=str, required=True,
                    choices=["hhcart", "randcart", "oc1", "wodt"],
                    help="Model to use")
parser.add_argument("--max-depth", type=int, default=20,
                    help="Maximum depth to sweep over")
parser.add_argument("--output-filename", type=str, required=True,
                    help="Output CSV filename")
args = parser.parse_args()
print(f"Arguments received: {args}")

print("Loading datasets...")
all_data = load_shape_dataset(folder_name=args.folder)
filtered_data = {k: v for k, v in all_data.items() if k.startswith(args.dataset)}
if not filtered_data:
    raise ValueError(f"No datasets found starting with '{args.dataset}'.")
all_datasets_dict = filtered_data
print(f"Datasets selected: {list(all_datasets_dict.keys())}")

try:
    seed = DEFAULT_VARIABLE_SEEDS[args.seed_index]
except IndexError:
    raise ValueError(f"Seed index {args.seed_index} out of range.")
print(f"Using seed: {seed}")

# Build model-specific registry
print("Preparing model registry...")
full_registry = DepthSweepRunner.build_registry(random_state=seed)
registry = {args.model: full_registry[args.model]}

print("Initializing DepthSweepRunner...")
runner = DepthSweepRunner(datasets=all_datasets_dict, max_depth=args.max_depth)

print("Running benchmark sweep...")
depth_sweep_df, _ = runner.run(
    auto_export=True,
    filename=args.output_filename,
    tree_dict_filename=args.output_filename.replace(".csv", ".pkl"),
    n_seeds=1,
    fixed_seed=seed,
    registry=registry,
    return_trees=True,
    save_tree_dict=True,
    batch_mode=True,
    output_subfolder=args.folder
)

elapsed = time.time() - start_time
print(f"[✓] Script completed in {elapsed:.2f} seconds")


# #!/usr/bin/env python3
# """
# run_depth_sweep_parallel.py
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
# from D_oblique_decision_trees.evaluation.benchmark_runner import DepthSweepRunner
# from src.config.paths import DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR
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
# parser.add_argument("--folder", type=str, default=None,
#                     help="Subfolder inside DATA_DIR/shapes to load (e.g., no_noise, 5_percent_noise, etc.)")
# parser.add_argument("--seed-index", type=int, default=0,
#                     help="Index into DEFAULT_VARIABLE_SEEDS")
# parser.add_argument("--model", type=str, choices=["hhcart", "randcart", "oc1", "wodt"],
#                     help="Model to use")
# parser.add_argument("--max-depth", type=int, default=20,
#                     help="Maximum depth to sweep over")
# parser.add_argument("--output-filename", type=str, required=True,
#                     help="Output CSV filename")
# args = parser.parse_args()
# print(f"Arguments received: {args}")
#
# print("Loading datasets...")
# all_data = load_shape_dataset(folder_name=args.folder)
# print(f"Datasets loaded: {list(all_data.keys())}")
#
# # Use a filter to collect all keys that start with the provided dataset name.
# filtered_data = {k: v for k, v in all_data.items() if k.startswith(args.dataset)}
# if not filtered_data:
#     raise ValueError(f"No datasets found starting with '{args.dataset}'.")
# all_datasets_dict = filtered_data
# print(f"Datasets selected: {list(all_datasets_dict.keys())}")
#
# # if args.dataset not in all_data:
# #     raise ValueError(f"Dataset '{args.dataset}' not found.")
# # all_datasets_dict = {args.dataset: all_data[args.dataset]}
# # print(f"Dataset selected: {args.dataset}")
#
# try:
#     seed = DEFAULT_VARIABLE_SEEDS[args.seed_index]
# except IndexError:
#     raise ValueError(f"Seed index {args.seed_index} out of range.")
# print(f"Using seed: {seed}")
#
# print("Initializing DepthSweepRunner...")
# runner = DepthSweepRunner(datasets=all_datasets_dict, max_depth=args.max_depth)
# print("Running benchmark sweep...")
# depth_sweep_df = runner.run(auto_export=True, filename=args.output_filename, n_seeds=1, fixed_seed=seed)
#
# if args.model:
#     print(f"Filtering by model: {args.model}")
#     depth_sweep_df = depth_sweep_df[depth_sweep_df["algorithm"] == args.model]
#
# # Determine results folder based on the provided folder argument.
# if args.folder:
#     results_dir = os.path.join(DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR, args.folder)
#     os.makedirs(results_dir, exist_ok=True)
#     results_file = os.path.join(str(results_dir), args.output_filename)
# else:
#     results_file = os.path.join(DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR, args.output_filename)
#
# depth_sweep_df.to_csv(str(results_file), index=False)
# print(f"\nFiltered results saved to: {results_file}")
#
# elapsed = time.time() - start_time
# print(f"Script completed in {elapsed:.2f} seconds")


# # ==============================
# # run_depth_sweep_parallel.py
# # ==============================
#
# import argparse
# import os
# import time
# import sys
#
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# sys.path.insert(0, project_root)
#
# from src.load_shapes import load_shape_dataset
# from D_oblique_decision_trees.evaluation.benchmark_runner import DepthSweepRunner
# from src.config.paths import DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR
# from src.config.settings import DEFAULT_VARIABLE_SEEDS
#
# os.environ["OMP_NUM_THREADS"] = os.environ.get("SLURM_CPUS_PER_TASK", "1")
#
# start_time = time.time()
# print(">>> Starting Depth Sweep Script")
# print(f"OMP_NUM_THREADS set to: {os.environ['OMP_NUM_THREADS']}")
#
# parser = argparse.ArgumentParser(description="Run depth sweep for selected dataset, seed, and model.")
# parser.add_argument("--dataset", type=str, required=True, help="Specify one dataset to run")
# parser.add_argument("--seed-index", type=int, default=0, help="Index into DEFAULT_VARIABLE_SEEDS")
# parser.add_argument("--model", type=str, choices=["hhcart", "randcart", "oc1", "wodt"], help="Model to use")
# parser.add_argument("--max-depth", type=int, default=20, help="Maximum depth to sweep over")
# parser.add_argument("--output-filename", type=str, required=True, help="Output CSV filename")
# args = parser.parse_args()
# print(f"Arguments received: {args}")
#
# print("Loading all datasets...")
# all_data = load_shape_dataset()
# print(f"Datasets loaded: {list(all_data.keys())}")
#
# if args.dataset not in all_data:
#     raise ValueError(f"Dataset '{args.dataset}' not found.")
# all_datasets_dict = {args.dataset: all_data[args.dataset]}
# print(f"Dataset selected: {args.dataset}")
#
# try:
#     seed = DEFAULT_VARIABLE_SEEDS[args.seed_index]
# except IndexError:
#     raise ValueError(f"Seed index {args.seed_index} out of range.")
# print(f"Using seed: {seed}")
#
# print("Initializing DepthSweepRunner...")
# runner = DepthSweepRunner(datasets=all_datasets_dict, max_depth=args.max_depth)
# print("Running benchmark sweep...")
# depth_sweep_df = runner.run(auto_export=True, filename=args.output_filename, n_seeds=1, fixed_seed=seed)
#
# if args.model:
#     print(f"Filtering by model: {args.model}")
#     depth_sweep_df = depth_sweep_df[depth_sweep_df["algorithm"] == args.model]
#
# results_file = os.path.join(DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR, args.output_filename)
# depth_sweep_df.to_csv(str(results_file), index=False)
# print(f"\nFiltered results saved to: {results_file}")
#
# elapsed = time.time() - start_time
# print(f"Script completed in {elapsed:.2f} seconds")
