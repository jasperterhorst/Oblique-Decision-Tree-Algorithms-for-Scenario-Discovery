from src.load_shapes import load_shape_dataset


# Configuration
FOLDER_NAME = "shapes_sample_size_tests"
SUBFOLDER_NAME = "label_noise_003"
SEED_INDICES = [0, 1, 2]
MODELS_WITH_SEEDS = ["moc1"]
MODELS_SINGLE_RUN = ["hhcart_d"]

# Initialise job list with standard logs folder creation
job_lines = ["mkdir -p logs\n"]

# Load datasets
all_datasets = load_shape_dataset(folder_name=FOLDER_NAME, subfolder_name=SUBFOLDER_NAME)
dataset_prefixes = sorted(all_datasets.keys())

# Generate job commands
for dataset_prefix in dataset_prefixes:
    # MOC1 runs with varying seeds
    for seed in SEED_INDICES:
        output_file = f"{dataset_prefix}_moc1_seed{seed}.csv"
        log_file = f"logs/{dataset_prefix}_moc1_seed{seed}.log"
        cmd = (
            f"python ../run_depth_sweep_benchmark.py "
            f"--dataset {dataset_prefix} "
            f"--folder-name {FOLDER_NAME} "
            f"--subfolder-name {SUBFOLDER_NAME} "
            f"--seed-index {seed} "
            f"--model moc1 "
            f"--output-filename {output_file} "
            f"--output-subfolder delftblue_sample_size_runs > {log_file} 2>&1"
        )
        job_lines.append(cmd + "\n")

    # HHCart-D single run
    output_file = f"{dataset_prefix}_hhcart_d_seed0.csv"
    log_file = f"logs/{dataset_prefix}_hhcart_d_seed0.log"
    cmd = (
        f"python ../run_depth_sweep_benchmark.py "
        f"--dataset {dataset_prefix} "
        f"--folder-name {FOLDER_NAME} "
        f"--subfolder-name {SUBFOLDER_NAME} "
        f"--seed-index 0 "
        f"--model hhcart_d "
        f"--output-filename {output_file} "
        f"--output-subfolder delftblue_sample_size_runs > {log_file} 2>&1"
    )
    job_lines.append(cmd + "\n")

# Save job list in the correct folder
with open("job_list_sample_size.txt", "w", newline="\n") as f:
    f.writelines(job_lines)

print(f"[OK] Generated job_list_sample_size.txt with {len(job_lines)-1} jobs.")