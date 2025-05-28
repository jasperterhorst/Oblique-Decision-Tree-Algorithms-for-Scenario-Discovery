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
from C_oblique_decision_tree_benchmark.evaluation.benchmark_runner import DepthSweepRunner
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
                        choices=["hhcart_a", "hhcart_d", "randcart", "moc1", "wodt", "co2", "cart", "ridge_cart"],
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
        output_subfolder=args.output_subfolder,
    )

    elapsed = time.time() - start_time
    print(f"[✓] Completed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
