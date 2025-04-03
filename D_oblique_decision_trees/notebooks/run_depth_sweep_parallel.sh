#!/bin/bash

#SBATCH --job-name="DepthSweep"
#SBATCH --time=00:02:00
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=2
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=education-tpm-msc-epa
#SBATCH --array=0-7

export OMP_NUM_THREADS=1

echo "=== SLURM JOB STARTED ==="
echo "Running on $(hostname)"
echo "Date: $(date)"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"

# Load modules
module load 2023r1
module load openmpi
module load python
module load py-pip

# Dataset array setup
DATASETS=(barbell_2d sine_wave_2d star_2d radial_segment_2d rectangle_2d barbell_3d radial_segment_3d saddle_3d)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

# Ensure logs directory exists
mkdir -p logs

echo "Dataset selected: $DATASET"

# Loop over all 5 seeds and launch each as a background process
for SEED_INDEX in {0..4}; do
  echo "Launching seed $SEED_INDEX for dataset $DATASET"
  srun --exclusive -N1 -n1 python3 run_depth_sweep_parallel.py \
    --dataset $DATASET \
    --seed-index $SEED_INDEX \
    --output-filename depth_sweep_${DATASET}_seed${SEED_INDEX}.csv \
    > logs/depth_sweep_${DATASET}_seed${SEED_INDEX}.log 2>&1 &
done

# Wait for all background jobs to complete
wait

echo "=== JOB COMPLETE for dataset: $DATASET ==="
