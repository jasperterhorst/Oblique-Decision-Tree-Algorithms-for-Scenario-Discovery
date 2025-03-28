"""
Cross-platform file paths for _data and outputs.
All paths are relative to the project root.
"""

import os

# Project root (two levels up from this file)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Data folders
DATA_DIR = os.path.join(BASE_DIR, "_data")
SHAPES_DIR = os.path.join(DATA_DIR, "shapes")
SCENARIO_METHODS_DEMO_OUTPUTS_DIR = os.path.join(DATA_DIR, "scenario_methods_demo_outputs")

# # Logging folder
# LOG_DIR = os.path.join(BASE_DIR, "logs")

# Ensure required directories exist
for path in [DATA_DIR, SHAPES_DIR, SCENARIO_METHODS_DEMO_OUTPUTS_DIR]:
    os.makedirs(path, exist_ok=True)
