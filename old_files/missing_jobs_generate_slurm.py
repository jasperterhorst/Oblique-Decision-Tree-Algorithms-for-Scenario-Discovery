import os

INPUT_FILE = "missing_jobs.txt"
OUTPUT_SCRIPT = "run_sparsity_array.sh"

# Load all lines
with open(INPUT_FILE, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

# Extract unique datasets
datasets = sorted(set(line.split(",")[0] for line in lines))
print(f"[✓] Found {len(datasets)} unique datasets.")

total_cpus = 16 * len(datasets)

# Group jobs by dataset
grouped_jobs = {ds: [] for ds in datasets}
for line in lines:
    try:
        dataset, folder, seed, model = line.split(",")
        grouped_jobs[dataset].append((dataset, folder, seed, model))
    except ValueError:
        print(f"[!] Skipping malformed line: {line}")
        continue

# Write rerun script
with open(OUTPUT_SCRIPT, "w") as f:
    f.write("#!/bin/bash\n")
    f.write("#SBATCH --job-name=RerunMissingJobs\n")
    f.write("#SBATCH --time=12:00:00\n")
    f.write("#SBATCH --ntasks=1\n")
    f.write(f"#SBATCH --cpus-per-task={total_cpus}\n")
    f.write("#SBATCH --partition=compute\n")
    f.write("#SBATCH --mem-per-cpu=1GB\n")
    f.write("#SBATCH --account=education-tpm-msc-epa\n")
    f.write(f"#SBATCH --array=0-{len(datasets)-1}\n\n")

    f.write("export OMP_NUM_THREADS=1\n\n")
    f.write("module load 2023r1\n")
    f.write("module load openmpi\n")
    f.write("module load python\n")
    f.write("module load py-pip\n\n")

    f.write("DATASETS=(" + " ".join(datasets) + ")\n")
    f.write("TARGET_DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}\n\n")
    f.write("echo \"Running dataset: $TARGET_DATASET\"\n")
    f.write("mkdir -p logs\n\n")

    f.write("MAX_PARALLEL=16\n\n")
    f.write("case \"$TARGET_DATASET\" in\n")
    for ds in datasets:
        f.write(f"  {ds})\n")
        for dataset, folder, seed, model in grouped_jobs[ds]:
            out_file = f"{dataset}_{folder}_{model}_seed{seed}.csv"
            log_file = f"logs/{dataset}_{folder}_{model}_seed{seed}.log"
            f.write(f"    echo \"Launching: {out_file}\"\n")
            f.write(f"    srun -N1 -n1 python run_depth_sweep_parallel.py "
                    f"--dataset {dataset} --folder {folder} "
                    f"--seed-index {seed} --model {model} "
                    f"--output-filename {out_file} > {log_file} 2>&1 &\n")
            f.write("    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do\n")
            f.write("      wait -n\n")
            f.write("    done\n")
        f.write("    wait\n")
        f.write("    ;;\n")
    f.write("esac\n")

print(f"[✓] SLURM array rerun script saved to: {OUTPUT_SCRIPT}")
#
#
# import os
#
# INPUT_FILE = "missing_jobs.txt"
# OUTPUT_SCRIPT = "run_sparsity_array.sh"
#
# # Load all lines
# with open(INPUT_FILE, "r") as f:
#     lines = [line.strip() for line in f if line.strip()]
#
# # Extract unique datasets
# datasets = sorted(set(line.split(",")[0] for line in lines))
# print(f"[✓] Found {len(datasets)} unique datasets.")
#
# # Group jobs by dataset
# grouped_jobs = {ds: [] for ds in datasets}
# for line in lines:
#     try:
#         dataset, folder, seed, model = line.split(",")
#         grouped_jobs[dataset].append((dataset, folder, seed, model))
#     except ValueError:
#         print(f"[!] Skipping malformed line: {line}")
#         continue
#
# # Write rerun script
# with open(OUTPUT_SCRIPT, "w") as f:
#     f.write("#!/bin/bash\n")
#     f.write("#SBATCH --job-name=RerunMissingJobs\n")
#     f.write("#SBATCH --time=4:00:00\n")
#     f.write("#SBATCH --ntasks=1\n")
#     f.write("#SBATCH --cpus-per-task=8\n")
#     f.write("#SBATCH --partition=compute\n")
#     f.write("#SBATCH --mem-per-cpu=1GB\n")
#     f.write("#SBATCH --account=education-tpm-msc-epa\n")
#     f.write(f"#SBATCH --array=0-{len(datasets)-1}\n\n")
#
#     f.write("export OMP_NUM_THREADS=1\n\n")
#     f.write("module load 2023r1\n")
#     f.write("module load openmpi\n")
#     f.write("module load python\n")
#     f.write("module load py-pip\n\n")
#
#     f.write("DATASETS=(" + " ".join(datasets) + ")\n")
#     f.write("TARGET_DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}\n\n")
#     f.write("echo \"Running dataset: $TARGET_DATASET\"\n")
#     f.write("mkdir -p logs\n\n")
#
#     f.write("case \"$TARGET_DATASET\" in\n")
#     for ds in datasets:
#         f.write(f"  {ds})\n")
#         f.write("    JOB_COUNT=0\n")
#         f.write("    MAX_PARALLEL=8\n")
#         for dataset, folder, seed, model in grouped_jobs[ds]:
#             out_file = f"{dataset}_{folder}_{model}_seed{seed}.csv"
#             log_file = f"logs/{dataset}_{folder}_{model}_seed{seed}.log"
#             f.write(f"    echo \"Launching: {out_file}\"\n")
#             f.write(f"    srun -N1 -n1 --exclusive python run_depth_sweep_benchmark.py "
#                     f"--dataset {dataset} --folder {folder} "
#                     f"--seed-index {seed} --model {model} "
#                     f"--output-filename {out_file} > {log_file} 2>&1 &\n")
#             f.write("    (( JOB_COUNT++ ))\n")
#             f.write("    if (( JOB_COUNT % MAX_PARALLEL == 0 )); then\n")
#             f.write("      wait\n")
#             f.write("    fi\n")
#         f.write("    wait\n")
#         f.write("    ;;\n")
#     f.write("esac\n")
#
# print(f"[✓] SLURM array rerun script saved to: {OUTPUT_SCRIPT}")
