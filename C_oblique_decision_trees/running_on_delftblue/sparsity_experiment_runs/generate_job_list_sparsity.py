import itertools

FOLDER_NAME = "shapes_noise_dimensionality_tests"
SUBFOLDER_NAME = "fuzziness_003"

DATASET = "radial_segment_3d_fuzziness_003_10000_samples_dim_20"
OUTPUT_FOLDER = "delftblue_sparsity_runs"
SCRIPT = "../run_depth_sweep_benchmark.py"

oc1_lambda = [0.0, 0.050, 0.1, 0.2, 0.5]
oc1_threshold = [0.0, 0.025, 0.050, 0.075, 0.1]
oc1_seeds = [0, 1, 2]

hhcart_alpha = [0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
hhcart_seeds = [0]

jobs = []


def format_decimal(value):
    return str(value).replace('.', '')


# OC1 Jobs
for lambda_reg, threshold_value, seed in itertools.product(oc1_lambda, oc1_threshold, oc1_seeds):
    lambda_str = format_decimal(lambda_reg)
    threshold_str = format_decimal(threshold_value)
    output_filename = f"{DATASET}_oc1_seed{seed}_l{lambda_str}_t{threshold_str}"
    log_file = f"logs/{DATASET}_oc1_seed{seed}_l{lambda_str}_t{threshold_str}.log"

    cmd = (
        f"python {SCRIPT} "
        f"--dataset {DATASET} "
        f"--folder-name {FOLDER_NAME} "
        f"--subfolder-name {SUBFOLDER_NAME} "
        f"--model oc1 "
        f"--lambda_reg {lambda_reg} "
        f"--threshold_value {threshold_value} "
        f"--seed-index {seed} "
        f"--output-filename {output_filename} "
        f"--output-subfolder {OUTPUT_FOLDER} > {log_file} 2>&1 "
    )
    jobs.append(cmd)

# HHCART(D) Jobs
for alpha, seed in itertools.product(hhcart_alpha, hhcart_seeds):
    alpha_str = format_decimal(alpha)
    output_filename = f"{DATASET}_hhcart_d_seed{seed}_a{alpha_str}"
    log_file = f"logs/{DATASET}_hhcart_d_seed{seed}_a{alpha_str}.log"

    cmd = (
        f"python {SCRIPT} "
        f"--dataset {DATASET} "
        f"--folder-name {FOLDER_NAME} "
        f"--subfolder-name {SUBFOLDER_NAME} "
        f"--model hhcart_d "
        f"--alpha {alpha} "
        f"--seed-index {seed} "
        f"--output-filename {output_filename} "
        f"--output-subfolder {OUTPUT_FOLDER} > {log_file} 2>&1 "
    )
    jobs.append(cmd)

with open("job_list_sparsity.txt", "w", newline="\n") as f:
    for job in jobs:
        f.write(job + "\n")

print(f"Generated {len(jobs)} jobs in job_list_sparsity.txt")