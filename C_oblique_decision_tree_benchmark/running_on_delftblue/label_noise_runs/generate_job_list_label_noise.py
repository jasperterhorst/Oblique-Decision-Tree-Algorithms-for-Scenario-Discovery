import os


DATASETS = [
    "barbell_2d", "sine_wave_2d", "star_2d", "radial_segment_2d",
    "rectangle_2d", "barbell_3d", "radial_segment_3d", "saddle_3d"
]
FOLDER = "shapes"
NOISE_FOLDERS = ["label_noise_000", "label_noise_003", "label_noise_005", "label_noise_007"]
MODELS = ["hhcart_a", "hhcart_d", "randcart", "moc1", "wodt", "cart", "ridge_cart"]

DEFAULT_SEEDS = [0, 1, 2, 3, 4]

# Custom seeds per model (override default)
MODEL_SEEDS = {
    "hhcart_a": [0],    # Run only seed 0 for HHCART(A)
    "hhcart_d": [0],    # Run only seed 0 for HHCART(D)
    "cart": [0],        # Run only seed 0 for CART
    "ridge_cart": [0],  # Run only seed 0 for RidgeCART
}

OUTPUT_JOBLIST = "job_list_label_noise.txt"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

with open(OUTPUT_JOBLIST, "w", newline="\n") as f:
    for dataset in DATASETS:
        for folder in NOISE_FOLDERS:
            for model in MODELS:
                seed_indices = MODEL_SEEDS.get(model, DEFAULT_SEEDS)
                for seed_index in seed_indices:
                    output_file = f"{dataset}_{folder}_{model}_seed{seed_index}.csv"
                    log_file = f"{LOG_DIR}/{dataset}_{folder}_{model}_seed{seed_index}.log"

                    cmd = (
                        f"python ../run_depth_sweep_benchmark.py "
                        f"--dataset {dataset} "
                        f"--folder-name {FOLDER} "
                        f"--subfolder-name {folder} "
                        f"--seed-index {seed_index} "
                        f"--model {model} "
                        f"--output-filename {output_file} "
                        f"--output-subfolder delftblue_label_noise_runs > {log_file} 2>&1"
                    )
                    f.write(cmd + "\n")

print(f"[OK] Generated {OUTPUT_JOBLIST}.")