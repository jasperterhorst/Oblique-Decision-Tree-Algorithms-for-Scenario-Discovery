#!/bin/bash
#SBATCH --job-name="DepthSweep"
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=education-tpm-msc-epa
#SBATCH --array=0-7

export OMP_NUM_THREADS=1

echo "=== SLURM JOB STARTED ==="
echo "Running on $(hostname)"
echo "Date: $(date)"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"

module load 2023r1
module load openmpi
module load python
module load py-pip

# Dataset list (one per SLURM array index)
DATASETS=(barbell_2d sine_wave_2d star_2d radial_segment_2d rectangle_2d barbell_3d radial_segment_3d saddle_3d)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

# Noise folders, models, and seeds
NOISE_FOLDERS=(fuzziness_000 fuzziness_003 fuzziness_005 fuzziness_007)
MODELS=(hhcart randcart oc1 wodt)
SEED_INDICES=(0 1 2 3 4)

mkdir -p logs

echo "Dataset selected: $DATASET"

# Max number of concurrent jobs within this SLURM task
MAX_PARALLEL=8
JOB_COUNT=0

for FOLDER in "${NOISE_FOLDERS[@]}"; do
  for SEED_INDEX in "${SEED_INDICES[@]}"; do
    for MODEL in "${MODELS[@]}"; do

      FILENAME="${DATASET}_${FOLDER}_${MODEL}_seed${SEED_INDEX}.csv"
      LOGFILE="logs/${DATASET}_${FOLDER}_${MODEL}_seed${SEED_INDEX}.log"

      echo "Launching: $FILENAME"
      srun --exclusive -N1 -n1 python run_depth_sweep_parallel.py \
        --dataset "$DATASET" \
        --folder "$FOLDER" \
        --seed-index "$SEED_INDEX" \
        --model "$MODEL" \
        --output-filename "$FILENAME" > "$LOGFILE" 2>&1 &

      (( JOB_COUNT++ ))
      if (( JOB_COUNT % MAX_PARALLEL == 0 )); then
        wait
      fi

    done
  done
done

wait
echo "=== JOB COMPLETE for dataset: $DATASET ==="
