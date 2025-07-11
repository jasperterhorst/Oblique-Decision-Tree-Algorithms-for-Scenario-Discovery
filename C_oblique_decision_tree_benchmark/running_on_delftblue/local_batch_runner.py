import subprocess
import os
import sys
import time
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # Add tqdm for progress bar

# ==== Config ====
DATASETS = [
    "barbell_2d", "sine_wave_2d", "star_2d", "radial_segment_2d",
    "rectangle_2d", "barbell_3d", "radial_segment_3d", "saddle_3d"
]
NOISE_FOLDERS = ["label_noise_000", "label_noise_003", "label_noise_005", "label_noise_007"]
MODELS = ["hhcart_a", "hhcart_d", "randcart", "moc1", "wodt", "co2", "cart", "ridge_cart"]
SEED_INDICES = [0, 1, 2]
MAX_PARALLEL = 12

os.makedirs("logs", exist_ok=True)

# ==== Task list ====
combinations = list(product(DATASETS, NOISE_FOLDERS, SEED_INDICES, MODELS))


def run_job(dataset, noise, seed_index, model):
    filename = f"{dataset}_{noise}_{model}_seed{seed_index}.csv"
    logfile = f"logs/{dataset}_{noise}_{model}_seed{seed_index}.log"

    cmd = [
        sys.executable,
        "run_depth_sweep_benchmark.py",
        "--dataset", dataset,
        "--folder-name", "shapes",
        "--subfolder-name", noise,
        "--seed-index", str(seed_index),
        "--model", model,
        "--output-filename", filename
    ]

    with open(logfile, "w") as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        process.wait()

    # Try to extract the saved path from the log (optional)
    save_path = ""
    try:
        with open(logfile, "r") as log:
            lines = log.readlines()
            for line in reversed(lines):
                if "Final CSV saved to:" in line:
                    save_path = line.strip()
                    break
    except Exception:
        pass

    return f"[OK] {filename} → {save_path}" if save_path else f"[OK] {filename} (log saved)"


# ==== Parallel Execution with Progress Bar and Timer ====
if __name__ == "__main__":
    total_jobs = len(combinations)
    print(f"Running {total_jobs} combinations with up to {MAX_PARALLEL} workers...\n")

    start = time.time()
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = {executor.submit(run_job, *combo): combo for combo in combinations}
        with tqdm(total=total_jobs, desc="Progress", unit="job") as pbar:
            for future in as_completed(futures):
                result = future.result()
                print(result)
                pbar.update(1)

    elapsed = time.time() - start
    print(f"\n[OK] All jobs completed in {elapsed:.2f} seconds.")
