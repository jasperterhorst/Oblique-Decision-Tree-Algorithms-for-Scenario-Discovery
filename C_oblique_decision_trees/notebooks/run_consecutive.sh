#!/bin/bash
#SBATCH --job-name=DepthSweepArray
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1G
#SBATCH --account=education-tpm-msc-epa
#SBATCH --array=0-767

module load 2023r1
module load python
module load py-pip

export OMP_NUM_THREADS=1

# Pull the correct command from job_list.txt (line = array ID + 1)
JOB=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" job_list.txt)

echo "=== SLURM JOB $SLURM_ARRAY_TASK_ID STARTED ==="
echo "Running command: $JOB"
eval $JOB
