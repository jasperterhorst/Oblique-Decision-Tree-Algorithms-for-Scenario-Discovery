import os

# Constants
BASE_FOLDER = "fuzziness_000_increased_dimensionality"
DATA_PATH = os.path.join("..", "..", "..", "_data", "shapes", BASE_FOLDER)
MODELS = ["hhcart_d", "oc1"]
SEED_INDICES = [0, 1, 2]
LOG_DIR = "logs_dim_sweep"
JOB_LIST_PATH = "job_list_dimensionality.txt"

# Step 1: Gather dataset variants (from subfolders)
dataset_prefixes = []
for folder in sorted(os.listdir(DATA_PATH)):
    folder_path = os.path.join(DATA_PATH, folder)
    if not os.path.isdir(folder_path):
        continue
    for file in os.listdir(folder_path):
        if file.endswith("_x.csv"):
            prefix = file.replace("_x.csv", "")
            dataset_prefixes.append((folder, prefix))  # (subfolder, dataset_name)

print("Detected datasets:", dataset_prefixes)

# Step 2: Write job list
with open(JOB_LIST_PATH, "w") as f:
    for folder, dataset in dataset_prefixes:
        for model in MODELS:
            if model == "hhcart_d":
                log_file = f"{LOG_DIR}/{dataset}_{folder}_{model}.log"
                output_file = f"{dataset}_{folder}_{model}.csv"
                cmd = (
                    f"mkdir -p {LOG_DIR} && "
                    f"python run_depth_sweep_parallel.py "
                    f"--dataset {dataset} "
                    f"--folder {os.path.join(BASE_FOLDER, folder)} "
                    f"--model {model} "
                    f"--output-filename {output_file} > {log_file} 2>&1"
                )
                f.write(cmd + "\n")

            elif model == "oc1":
                for seed_index in SEED_INDICES:
                    log_file = f"{LOG_DIR}/{dataset}_{folder}_{model}_seed{seed_index}.log"
                    output_file = f"{dataset}_{folder}_{model}_seed{seed_index}.csv"
                    cmd = (
                        f"mkdir -p {LOG_DIR} && "
                        f"python run_depth_sweep_parallel.py "
                        f"--dataset {dataset} "
                        f"--folder {os.path.join(BASE_FOLDER, folder)} "
                        f"--model {model} "
                        f"--seed-index {seed_index} "
                        f"--output-filename {output_file} > {log_file} 2>&1"
                    )
                    f.write(cmd + "\n")

print(f"✅ Job list written to {JOB_LIST_PATH}")
