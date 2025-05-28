"""
Cross-platform file paths for _data and outputs.
All paths are relative to the project root.
"""

from pathlib import Path

# Project root (two levels up from this file)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data folders
DATA_DIR = BASE_DIR / "_data"
SHAPES_DIR = DATA_DIR / "shapes"
SCENARIO_METHODS_DEMO_OUTPUTS_DIR = DATA_DIR / "scenario_methods_demo_outputs"
DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR = DATA_DIR / "depth_sweep_batch_results"
DEPTH_SWEEP_SINGLE_RUN_RESULTS_OUTPUTS_DIR = DATA_DIR / "depth_sweep_single_run_results"
HAMARAT_DATA_DIR = DATA_DIR / "hamarat_et_al_2013"

# Ensure required directories exist
for path in [
    DATA_DIR,
    SHAPES_DIR,
    SCENARIO_METHODS_DEMO_OUTPUTS_DIR,
    DEPTH_SWEEP_BATCH_RESULTS_OUTPUTS_DIR,
    DEPTH_SWEEP_SINGLE_RUN_RESULTS_OUTPUTS_DIR
]:
    path.mkdir(parents=True, exist_ok=True)
