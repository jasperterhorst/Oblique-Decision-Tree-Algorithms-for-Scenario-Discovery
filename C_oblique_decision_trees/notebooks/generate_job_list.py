# generate_job_list.py

DATASETS = [
    "barbell_2d", "sine_wave_2d", "star_2d", "radial_segment_2d",
    "rectangle_2d", "barbell_3d", "radial_segment_3d", "saddle_3d"
]
NOISE_FOLDERS = ["fuzziness_000", "fuzziness_003", "fuzziness_005", "fuzziness_007"]
MODELS = ["hhcart_a", "hhcart_d", "randcart", "oc1", "wodt", "co2", "cart", "ridge_cart"]
SEED_INDICES = [0, 1, 2]

with open("job_list.txt", "w", newline="\n") as f:
    for dataset in DATASETS:
        for folder in NOISE_FOLDERS:
            for seed_index in SEED_INDICES:
                for model in MODELS:
                    output_file = f"{dataset}_{folder}_{model}_seed{seed_index}.csv"
                    log_file = f"logs/{dataset}_{folder}_{model}_seed{seed_index}.log"
                    cmd = (
                        f"mkdir -p logs && "
                        f"python run_depth_sweep_parallel.py "
                        f"--dataset {dataset} "
                        f"--folder {folder} "
                        f"--seed-index {seed_index} "
                        f"--model {model} "
                        f"--output-filename {output_file} > {log_file} 2>&1"
                    )
                    f.write(cmd + "\n")
