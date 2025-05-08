#!/bin/bash
#SBATCH --job-name=DepthSweepLabelNoise
#SBATCH --time=20:00:00
#SBATCH --array=0-671%80
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1G
#SBATCH --account=Research-TPM-MAS

module load 2023r1
module load python
module load py-pip

export OMP_NUM_THREADS=1

# Read the specific command for this array index
JOB_CMD=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" job_list_label_noise.txt)

echo "Running job: $JOB_CMD"
bash -c "$JOB_CMD"

echo "Job $SLURM_ARRAY_TASK_ID completed."
