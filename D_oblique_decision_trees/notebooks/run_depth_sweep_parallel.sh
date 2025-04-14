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
NOISE_FOLDERS=(no_noise 5_percent_noise 15_percent_noise 25_percent_noise)
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



##!/bin/bash
##SBATCH --job-name="DepthSweep"
##SBATCH --time=00:05:00
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=4
##SBATCH --partition=compute
##SBATCH --mem-per-cpu=2GB
##SBATCH --account=education-tpm-msc-epa
##SBATCH --array=0-7
#
#export OMP_NUM_THREADS=1
#
#echo "=== SLURM JOB STARTED ==="
#echo "Running on $(hostname)"
#echo "Date: $(date)"
#echo "SLURM job ID: $SLURM_JOB_ID"
#echo "Array task ID: $SLURM_ARRAY_TASK_ID"
#echo "CPUs per task: $SLURM_CPUS_PER_TASK"
#
#module load 2023r1
#module load openmpi
#module load python
#module load py-pip
#
## Define dataset list (one per SLURM array index)
#DATASETS=(barbell_2d sine_wave_2d star_2d radial_segment_2d rectangle_2d barbell_3d radial_segment_3d saddle_3d)
#DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
#
## Noise folders and models
#NOISE_FOLDERS=(no_noise 5_percent_noise 15_percent_noise 25_percent_noise)
#MODELS=(hhcart randcart oc1 wodt)
#
## Number of seeds to loop through (based on your DEFAULT_VARIABLE_SEEDS)
#SEED_INDICES=(0 1 2 3 4)
#
#echo "Dataset selected: $DATASET"
#mkdir -p logs
#
## Loop over all combinations
#for FOLDER in "${NOISE_FOLDERS[@]}"; do
#  for SEED_INDEX in "${SEED_INDICES[@]}"; do
#    for MODEL in "${MODELS[@]}"; do
#      FILENAME="${DATASET}_${FOLDER}_${MODEL}_seed${SEED_INDEX}.csv"
#      echo "Running: $DATASET | $FOLDER | $MODEL | seed=$SEED_INDEX"
#      srun python run_depth_sweep_parallel.py \
#        --dataset "$DATASET" \
#        --folder "$FOLDER" \
#        --seed-index "$SEED_INDEX" \
#        --model "$MODEL" \
#        --output-filename "$FILENAME"
#    done
#  done
#done



##!/bin/bash
##SBATCH --job-name="DepthSweep"
##SBATCH --time=08:00:00
##SBATCH --ntasks=5
##SBATCH --cpus-per-task=4
##SBATCH --partition=compute
##SBATCH --mem-per-cpu=2GB
##SBATCH --account=education-tpm-msc-epa
##SBATCH --array=0-7
#
#export OMP_NUM_THREADS=1
#
#echo "=== SLURM JOB STARTED ==="
#echo "Running on $(hostname)"
#echo "Date: $(date)"
#echo "CPUs per task: $SLURM_CPUS_PER_TASK"
#echo "SLURM job ID: $SLURM_JOB_ID"
#echo "Array task ID: $SLURM_ARRAY_TASK_ID"
#
#module load 2023r1
#module load openmpi
#module load python
#module load py-pip
#
## Array of dataset names (same as before)
#DATASETS=(barbell_2d sine_wave_2d star_2d radial_segment_2d rectangle_2d barbell_3d radial_segment_3d saddle_3d)
#DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
#
## Array of noise folder names you want to evaluate sequentially.
#NOISE_FOLDERS=(no_noise 5_percent_noise 15_percent_noise 25_percent_noise)
#
#mkdir -p logs
#
#echo "Dataset selected: $DATASET"
#
## Loop over each noise folder
#for NOISE in "${NOISE_FOLDERS[@]}"; do
#    echo "Processing noise level: $NOISE"
#    for SEED_INDEX in {0..4}; do
#        echo "Launching seed $SEED_INDEX for dataset $DATASET at noise level $NOISE"
#        srun --exclusive -N1 -n1 python3 run_depth_sweep_parallel.py \
#            --dataset $DATASET \
#            --folder $NOISE \
#            --seed-index $SEED_INDEX \
#            --output-filename depth_sweep_${DATASET}_${NOISE}_seed${SEED_INDEX}.csv \
#            > logs/depth_sweep_${DATASET}_${NOISE}_seed${SEED_INDEX}.log 2>&1 &
#    done
#    wait
#    echo "Finished seeds for $DATASET at noise level: $NOISE"
#done
#
#echo "=== JOB COMPLETE for dataset: $DATASET ==="






##!/bin/bash
#
##SBATCH --job-name="DepthSweep"
##SBATCH --time=02:00:00
##SBATCH --ntasks=5
##SBATCH --cpus-per-task=4
##SBATCH --partition=compute
##SBATCH --mem-per-cpu=2GB
##SBATCH --account=education-tpm-msc-epa
##SBATCH --array=0-7
#
#export OMP_NUM_THREADS=1
#
#echo "=== SLURM JOB STARTED ==="
#echo "Running on $(hostname)"
#echo "Date: $(date)"
#echo "CPUs per task: $SLURM_CPUS_PER_TASK"
#echo "SLURM job ID: $SLURM_JOB_ID"
#echo "Array task ID: $SLURM_ARRAY_TASK_ID"
#
#module load 2023r1
#module load openmpi
#module load python
#module load py-pip
#
#DATASETS=(barbell_2d sine_wave_2d star_2d radial_segment_2d rectangle_2d barbell_3d radial_segment_3d saddle_3d)
#DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
#
#mkdir -p logs
#
#echo "Dataset selected: $DATASET"
#
#for SEED_INDEX in {0..4}; do
#  echo "Launching seed $SEED_INDEX for dataset $DATASET"
#  srun --exclusive -N1 -n1 python3 run_depth_sweep_parallel.py \
#    --dataset $DATASET \
#    --seed-index $SEED_INDEX \
#    --output-filename depth_sweep_${DATASET}_seed${SEED_INDEX}.csv \
#    > logs/depth_sweep_${DATASET}_seed${SEED_INDEX}.log 2>&1 &
#done
#
#wait
#echo "Finished seeds for $DATASET"
#
#echo "=== JOB COMPLETE for dataset: $DATASET ==="
