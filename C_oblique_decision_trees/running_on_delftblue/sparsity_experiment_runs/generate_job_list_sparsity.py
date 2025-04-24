import itertools

FOLDER_NAME = "shapes_noise_dimensionality_tests"
SUBFOLDER_NAME = "fuzziness_003"

DATASET = "radial_segment_3d_fuzziness_003_10000_samples_dim_30"
OUTPUT_FOLDER = "delftblue_sparsity_runs"
SCRIPT = "../run_depth_sweep_benchmark.py"

oc1_lambda = [0.0, 0.025, 0.050, 0.075, 0.1]
oc1_threshold = [0.0, 0.01, 0.02, 0.03, 0.04]
oc1_seeds = [0, 1, 2]

hhcart_alpha = [0.0, 0.1, 0.5, 1.0, 2.0]
hhcart_seeds = [0]

jobs = []

# OC1 Jobs
for lambda_reg, threshold_value, seed in itertools.product(oc1_lambda, oc1_threshold, oc1_seeds):
    cmd = (
        f"python {SCRIPT} "
        f"--dataset {DATASET} "
        f"--folder-name {FOLDER_NAME} "
        f"--subfolder-name {SUBFOLDER_NAME} "
        f"--algorithm OC1 "
        f"--lambda_reg {lambda_reg} "
        f"--threshold_value {threshold_value} "
        f"--seed {seed} "
        f"--output_folder {OUTPUT_FOLDER}"
    )
    jobs.append(cmd)

# HHCART(D) Jobs
for alpha, seed in itertools.product(hhcart_alpha, hhcart_seeds):
    cmd = (
        f"python {SCRIPT} "
        f"--dataset {DATASET} "
        f"--folder-name {FOLDER_NAME} "
        f"--subfolder-name {SUBFOLDER_NAME} "
        f"--algorithm HHCART_D "
        f"--alpha {alpha} "
        f"--seed {seed} "
        f"--output_folder {OUTPUT_FOLDER}"
    )
    jobs.append(cmd)

with open("job_list_sparsity.txt", "w", newline="\n") as f:
    for job in jobs:
        f.write(job + "\n")

print(f"Generated {len(jobs)} jobs in job_list_sparsity.txt")
