#!/bin/bash
#SBATCH --job-name=RerunMissingJobs
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=education-tpm-msc-epa
#SBATCH --array=0-2

export OMP_NUM_THREADS=1

module load 2023r1
module load openmpi
module load python
module load py-pip

DATASETS=(barbell_3d radial_segment_3d saddle_3d)
TARGET_DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "Running dataset: $TARGET_DATASET"
mkdir -p logs

MAX_PARALLEL=16

case "$TARGET_DATASET" in
  barbell_3d)
    echo "Launching: barbell_3d_fuzziness_000_randcart_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_000 --seed-index 1 --model randcart --output-filename barbell_3d_fuzziness_000_randcart_seed1.csv > logs/barbell_3d_fuzziness_000_randcart_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_000_ridge_cart_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_000 --seed-index 2 --model ridge_cart --output-filename barbell_3d_fuzziness_000_ridge_cart_seed2.csv > logs/barbell_3d_fuzziness_000_ridge_cart_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_003_hhcart_a_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_003 --seed-index 1 --model hhcart_a --output-filename barbell_3d_fuzziness_003_hhcart_a_seed1.csv > logs/barbell_3d_fuzziness_003_hhcart_a_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_003_hhcart_d_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_003 --seed-index 2 --model hhcart_d --output-filename barbell_3d_fuzziness_003_hhcart_d_seed2.csv > logs/barbell_3d_fuzziness_003_hhcart_d_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_003_oc1_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_003 --seed-index 0 --model oc1 --output-filename barbell_3d_fuzziness_003_oc1_seed0.csv > logs/barbell_3d_fuzziness_003_oc1_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_003_oc1_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_003 --seed-index 1 --model oc1 --output-filename barbell_3d_fuzziness_003_oc1_seed1.csv > logs/barbell_3d_fuzziness_003_oc1_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_003_oc1_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_003 --seed-index 2 --model oc1 --output-filename barbell_3d_fuzziness_003_oc1_seed2.csv > logs/barbell_3d_fuzziness_003_oc1_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_003_randcart_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_003 --seed-index 0 --model randcart --output-filename barbell_3d_fuzziness_003_randcart_seed0.csv > logs/barbell_3d_fuzziness_003_randcart_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_003_randcart_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_003 --seed-index 1 --model randcart --output-filename barbell_3d_fuzziness_003_randcart_seed1.csv > logs/barbell_3d_fuzziness_003_randcart_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_003_randcart_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_003 --seed-index 2 --model randcart --output-filename barbell_3d_fuzziness_003_randcart_seed2.csv > logs/barbell_3d_fuzziness_003_randcart_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_003_ridge_cart_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_003 --seed-index 0 --model ridge_cart --output-filename barbell_3d_fuzziness_003_ridge_cart_seed0.csv > logs/barbell_3d_fuzziness_003_ridge_cart_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_003_ridge_cart_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_003 --seed-index 1 --model ridge_cart --output-filename barbell_3d_fuzziness_003_ridge_cart_seed1.csv > logs/barbell_3d_fuzziness_003_ridge_cart_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_003_ridge_cart_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_003 --seed-index 2 --model ridge_cart --output-filename barbell_3d_fuzziness_003_ridge_cart_seed2.csv > logs/barbell_3d_fuzziness_003_ridge_cart_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_003_wodt_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_003 --seed-index 0 --model wodt --output-filename barbell_3d_fuzziness_003_wodt_seed0.csv > logs/barbell_3d_fuzziness_003_wodt_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_003_wodt_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_003 --seed-index 1 --model wodt --output-filename barbell_3d_fuzziness_003_wodt_seed1.csv > logs/barbell_3d_fuzziness_003_wodt_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_003_wodt_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_003 --seed-index 2 --model wodt --output-filename barbell_3d_fuzziness_003_wodt_seed2.csv > logs/barbell_3d_fuzziness_003_wodt_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_005_cart_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_005 --seed-index 0 --model cart --output-filename barbell_3d_fuzziness_005_cart_seed0.csv > logs/barbell_3d_fuzziness_005_cart_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_005_cart_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_005 --seed-index 1 --model cart --output-filename barbell_3d_fuzziness_005_cart_seed1.csv > logs/barbell_3d_fuzziness_005_cart_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_005_cart_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_005 --seed-index 2 --model cart --output-filename barbell_3d_fuzziness_005_cart_seed2.csv > logs/barbell_3d_fuzziness_005_cart_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_005_co2_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_005 --seed-index 0 --model co2 --output-filename barbell_3d_fuzziness_005_co2_seed0.csv > logs/barbell_3d_fuzziness_005_co2_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_005_co2_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_005 --seed-index 1 --model co2 --output-filename barbell_3d_fuzziness_005_co2_seed1.csv > logs/barbell_3d_fuzziness_005_co2_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_005_co2_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_005 --seed-index 2 --model co2 --output-filename barbell_3d_fuzziness_005_co2_seed2.csv > logs/barbell_3d_fuzziness_005_co2_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_005_hhcart_a_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_005 --seed-index 0 --model hhcart_a --output-filename barbell_3d_fuzziness_005_hhcart_a_seed0.csv > logs/barbell_3d_fuzziness_005_hhcart_a_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_005_hhcart_a_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_005 --seed-index 1 --model hhcart_a --output-filename barbell_3d_fuzziness_005_hhcart_a_seed1.csv > logs/barbell_3d_fuzziness_005_hhcart_a_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_005_hhcart_a_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_005 --seed-index 2 --model hhcart_a --output-filename barbell_3d_fuzziness_005_hhcart_a_seed2.csv > logs/barbell_3d_fuzziness_005_hhcart_a_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_005_hhcart_d_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_005 --seed-index 0 --model hhcart_d --output-filename barbell_3d_fuzziness_005_hhcart_d_seed0.csv > logs/barbell_3d_fuzziness_005_hhcart_d_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_005_hhcart_d_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_005 --seed-index 1 --model hhcart_d --output-filename barbell_3d_fuzziness_005_hhcart_d_seed1.csv > logs/barbell_3d_fuzziness_005_hhcart_d_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_005_hhcart_d_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_005 --seed-index 2 --model hhcart_d --output-filename barbell_3d_fuzziness_005_hhcart_d_seed2.csv > logs/barbell_3d_fuzziness_005_hhcart_d_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_005_oc1_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_005 --seed-index 0 --model oc1 --output-filename barbell_3d_fuzziness_005_oc1_seed0.csv > logs/barbell_3d_fuzziness_005_oc1_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_005_oc1_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_005 --seed-index 1 --model oc1 --output-filename barbell_3d_fuzziness_005_oc1_seed1.csv > logs/barbell_3d_fuzziness_005_oc1_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_005_oc1_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_005 --seed-index 2 --model oc1 --output-filename barbell_3d_fuzziness_005_oc1_seed2.csv > logs/barbell_3d_fuzziness_005_oc1_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_005_randcart_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_005 --seed-index 0 --model randcart --output-filename barbell_3d_fuzziness_005_randcart_seed0.csv > logs/barbell_3d_fuzziness_005_randcart_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_005_randcart_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_005 --seed-index 1 --model randcart --output-filename barbell_3d_fuzziness_005_randcart_seed1.csv > logs/barbell_3d_fuzziness_005_randcart_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_005_randcart_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_005 --seed-index 2 --model randcart --output-filename barbell_3d_fuzziness_005_randcart_seed2.csv > logs/barbell_3d_fuzziness_005_randcart_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_005_ridge_cart_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_005 --seed-index 0 --model ridge_cart --output-filename barbell_3d_fuzziness_005_ridge_cart_seed0.csv > logs/barbell_3d_fuzziness_005_ridge_cart_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_005_ridge_cart_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_005 --seed-index 1 --model ridge_cart --output-filename barbell_3d_fuzziness_005_ridge_cart_seed1.csv > logs/barbell_3d_fuzziness_005_ridge_cart_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_005_ridge_cart_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_005 --seed-index 2 --model ridge_cart --output-filename barbell_3d_fuzziness_005_ridge_cart_seed2.csv > logs/barbell_3d_fuzziness_005_ridge_cart_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_005_wodt_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_005 --seed-index 0 --model wodt --output-filename barbell_3d_fuzziness_005_wodt_seed0.csv > logs/barbell_3d_fuzziness_005_wodt_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_005_wodt_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_005 --seed-index 1 --model wodt --output-filename barbell_3d_fuzziness_005_wodt_seed1.csv > logs/barbell_3d_fuzziness_005_wodt_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_005_wodt_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_005 --seed-index 2 --model wodt --output-filename barbell_3d_fuzziness_005_wodt_seed2.csv > logs/barbell_3d_fuzziness_005_wodt_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_007_cart_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_007 --seed-index 0 --model cart --output-filename barbell_3d_fuzziness_007_cart_seed0.csv > logs/barbell_3d_fuzziness_007_cart_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_007_cart_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_007 --seed-index 1 --model cart --output-filename barbell_3d_fuzziness_007_cart_seed1.csv > logs/barbell_3d_fuzziness_007_cart_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_007_cart_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_007 --seed-index 2 --model cart --output-filename barbell_3d_fuzziness_007_cart_seed2.csv > logs/barbell_3d_fuzziness_007_cart_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_007_co2_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_007 --seed-index 0 --model co2 --output-filename barbell_3d_fuzziness_007_co2_seed0.csv > logs/barbell_3d_fuzziness_007_co2_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_007_co2_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_007 --seed-index 1 --model co2 --output-filename barbell_3d_fuzziness_007_co2_seed1.csv > logs/barbell_3d_fuzziness_007_co2_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_007_co2_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_007 --seed-index 2 --model co2 --output-filename barbell_3d_fuzziness_007_co2_seed2.csv > logs/barbell_3d_fuzziness_007_co2_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_007_hhcart_a_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_007 --seed-index 0 --model hhcart_a --output-filename barbell_3d_fuzziness_007_hhcart_a_seed0.csv > logs/barbell_3d_fuzziness_007_hhcart_a_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_007_hhcart_a_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_007 --seed-index 1 --model hhcart_a --output-filename barbell_3d_fuzziness_007_hhcart_a_seed1.csv > logs/barbell_3d_fuzziness_007_hhcart_a_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_007_hhcart_a_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_007 --seed-index 2 --model hhcart_a --output-filename barbell_3d_fuzziness_007_hhcart_a_seed2.csv > logs/barbell_3d_fuzziness_007_hhcart_a_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_007_hhcart_d_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_007 --seed-index 0 --model hhcart_d --output-filename barbell_3d_fuzziness_007_hhcart_d_seed0.csv > logs/barbell_3d_fuzziness_007_hhcart_d_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_007_hhcart_d_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_007 --seed-index 1 --model hhcart_d --output-filename barbell_3d_fuzziness_007_hhcart_d_seed1.csv > logs/barbell_3d_fuzziness_007_hhcart_d_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_007_hhcart_d_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_007 --seed-index 2 --model hhcart_d --output-filename barbell_3d_fuzziness_007_hhcart_d_seed2.csv > logs/barbell_3d_fuzziness_007_hhcart_d_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_007_oc1_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_007 --seed-index 0 --model oc1 --output-filename barbell_3d_fuzziness_007_oc1_seed0.csv > logs/barbell_3d_fuzziness_007_oc1_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_007_oc1_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_007 --seed-index 1 --model oc1 --output-filename barbell_3d_fuzziness_007_oc1_seed1.csv > logs/barbell_3d_fuzziness_007_oc1_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_007_oc1_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_007 --seed-index 2 --model oc1 --output-filename barbell_3d_fuzziness_007_oc1_seed2.csv > logs/barbell_3d_fuzziness_007_oc1_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_007_randcart_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_007 --seed-index 0 --model randcart --output-filename barbell_3d_fuzziness_007_randcart_seed0.csv > logs/barbell_3d_fuzziness_007_randcart_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_007_randcart_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_007 --seed-index 1 --model randcart --output-filename barbell_3d_fuzziness_007_randcart_seed1.csv > logs/barbell_3d_fuzziness_007_randcart_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_007_randcart_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_007 --seed-index 2 --model randcart --output-filename barbell_3d_fuzziness_007_randcart_seed2.csv > logs/barbell_3d_fuzziness_007_randcart_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_007_ridge_cart_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_007 --seed-index 0 --model ridge_cart --output-filename barbell_3d_fuzziness_007_ridge_cart_seed0.csv > logs/barbell_3d_fuzziness_007_ridge_cart_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_007_ridge_cart_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_007 --seed-index 1 --model ridge_cart --output-filename barbell_3d_fuzziness_007_ridge_cart_seed1.csv > logs/barbell_3d_fuzziness_007_ridge_cart_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_007_ridge_cart_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_007 --seed-index 2 --model ridge_cart --output-filename barbell_3d_fuzziness_007_ridge_cart_seed2.csv > logs/barbell_3d_fuzziness_007_ridge_cart_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_007_wodt_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_007 --seed-index 0 --model wodt --output-filename barbell_3d_fuzziness_007_wodt_seed0.csv > logs/barbell_3d_fuzziness_007_wodt_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_007_wodt_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_007 --seed-index 1 --model wodt --output-filename barbell_3d_fuzziness_007_wodt_seed1.csv > logs/barbell_3d_fuzziness_007_wodt_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: barbell_3d_fuzziness_007_wodt_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset barbell_3d --folder fuzziness_007 --seed-index 2 --model wodt --output-filename barbell_3d_fuzziness_007_wodt_seed2.csv > logs/barbell_3d_fuzziness_007_wodt_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    wait
    ;;
  radial_segment_3d)
    echo "Launching: radial_segment_3d_fuzziness_000_randcart_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_000 --seed-index 2 --model randcart --output-filename radial_segment_3d_fuzziness_000_randcart_seed2.csv > logs/radial_segment_3d_fuzziness_000_randcart_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_000_ridge_cart_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_000 --seed-index 0 --model ridge_cart --output-filename radial_segment_3d_fuzziness_000_ridge_cart_seed0.csv > logs/radial_segment_3d_fuzziness_000_ridge_cart_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_000_ridge_cart_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_000 --seed-index 1 --model ridge_cart --output-filename radial_segment_3d_fuzziness_000_ridge_cart_seed1.csv > logs/radial_segment_3d_fuzziness_000_ridge_cart_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_000_ridge_cart_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_000 --seed-index 2 --model ridge_cart --output-filename radial_segment_3d_fuzziness_000_ridge_cart_seed2.csv > logs/radial_segment_3d_fuzziness_000_ridge_cart_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_003_randcart_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_003 --seed-index 1 --model randcart --output-filename radial_segment_3d_fuzziness_003_randcart_seed1.csv > logs/radial_segment_3d_fuzziness_003_randcart_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_003_randcart_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_003 --seed-index 2 --model randcart --output-filename radial_segment_3d_fuzziness_003_randcart_seed2.csv > logs/radial_segment_3d_fuzziness_003_randcart_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_003_ridge_cart_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_003 --seed-index 0 --model ridge_cart --output-filename radial_segment_3d_fuzziness_003_ridge_cart_seed0.csv > logs/radial_segment_3d_fuzziness_003_ridge_cart_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_003_ridge_cart_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_003 --seed-index 1 --model ridge_cart --output-filename radial_segment_3d_fuzziness_003_ridge_cart_seed1.csv > logs/radial_segment_3d_fuzziness_003_ridge_cart_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_003_ridge_cart_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_003 --seed-index 2 --model ridge_cart --output-filename radial_segment_3d_fuzziness_003_ridge_cart_seed2.csv > logs/radial_segment_3d_fuzziness_003_ridge_cart_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_005_cart_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_005 --seed-index 0 --model cart --output-filename radial_segment_3d_fuzziness_005_cart_seed0.csv > logs/radial_segment_3d_fuzziness_005_cart_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_005_cart_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_005 --seed-index 1 --model cart --output-filename radial_segment_3d_fuzziness_005_cart_seed1.csv > logs/radial_segment_3d_fuzziness_005_cart_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_005_cart_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_005 --seed-index 2 --model cart --output-filename radial_segment_3d_fuzziness_005_cart_seed2.csv > logs/radial_segment_3d_fuzziness_005_cart_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_005_co2_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_005 --seed-index 0 --model co2 --output-filename radial_segment_3d_fuzziness_005_co2_seed0.csv > logs/radial_segment_3d_fuzziness_005_co2_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_005_co2_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_005 --seed-index 1 --model co2 --output-filename radial_segment_3d_fuzziness_005_co2_seed1.csv > logs/radial_segment_3d_fuzziness_005_co2_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_005_co2_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_005 --seed-index 2 --model co2 --output-filename radial_segment_3d_fuzziness_005_co2_seed2.csv > logs/radial_segment_3d_fuzziness_005_co2_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_005_hhcart_a_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_005 --seed-index 0 --model hhcart_a --output-filename radial_segment_3d_fuzziness_005_hhcart_a_seed0.csv > logs/radial_segment_3d_fuzziness_005_hhcart_a_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_005_hhcart_a_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_005 --seed-index 1 --model hhcart_a --output-filename radial_segment_3d_fuzziness_005_hhcart_a_seed1.csv > logs/radial_segment_3d_fuzziness_005_hhcart_a_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_005_hhcart_a_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_005 --seed-index 2 --model hhcart_a --output-filename radial_segment_3d_fuzziness_005_hhcart_a_seed2.csv > logs/radial_segment_3d_fuzziness_005_hhcart_a_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_005_hhcart_d_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_005 --seed-index 0 --model hhcart_d --output-filename radial_segment_3d_fuzziness_005_hhcart_d_seed0.csv > logs/radial_segment_3d_fuzziness_005_hhcart_d_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_005_hhcart_d_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_005 --seed-index 1 --model hhcart_d --output-filename radial_segment_3d_fuzziness_005_hhcart_d_seed1.csv > logs/radial_segment_3d_fuzziness_005_hhcart_d_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_005_hhcart_d_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_005 --seed-index 2 --model hhcart_d --output-filename radial_segment_3d_fuzziness_005_hhcart_d_seed2.csv > logs/radial_segment_3d_fuzziness_005_hhcart_d_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_005_oc1_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_005 --seed-index 0 --model oc1 --output-filename radial_segment_3d_fuzziness_005_oc1_seed0.csv > logs/radial_segment_3d_fuzziness_005_oc1_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_005_oc1_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_005 --seed-index 1 --model oc1 --output-filename radial_segment_3d_fuzziness_005_oc1_seed1.csv > logs/radial_segment_3d_fuzziness_005_oc1_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_005_oc1_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_005 --seed-index 2 --model oc1 --output-filename radial_segment_3d_fuzziness_005_oc1_seed2.csv > logs/radial_segment_3d_fuzziness_005_oc1_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_005_randcart_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_005 --seed-index 0 --model randcart --output-filename radial_segment_3d_fuzziness_005_randcart_seed0.csv > logs/radial_segment_3d_fuzziness_005_randcart_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_005_randcart_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_005 --seed-index 1 --model randcart --output-filename radial_segment_3d_fuzziness_005_randcart_seed1.csv > logs/radial_segment_3d_fuzziness_005_randcart_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_005_randcart_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_005 --seed-index 2 --model randcart --output-filename radial_segment_3d_fuzziness_005_randcart_seed2.csv > logs/radial_segment_3d_fuzziness_005_randcart_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_005_ridge_cart_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_005 --seed-index 0 --model ridge_cart --output-filename radial_segment_3d_fuzziness_005_ridge_cart_seed0.csv > logs/radial_segment_3d_fuzziness_005_ridge_cart_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_005_ridge_cart_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_005 --seed-index 1 --model ridge_cart --output-filename radial_segment_3d_fuzziness_005_ridge_cart_seed1.csv > logs/radial_segment_3d_fuzziness_005_ridge_cart_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_005_ridge_cart_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_005 --seed-index 2 --model ridge_cart --output-filename radial_segment_3d_fuzziness_005_ridge_cart_seed2.csv > logs/radial_segment_3d_fuzziness_005_ridge_cart_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_005_wodt_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_005 --seed-index 0 --model wodt --output-filename radial_segment_3d_fuzziness_005_wodt_seed0.csv > logs/radial_segment_3d_fuzziness_005_wodt_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_005_wodt_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_005 --seed-index 1 --model wodt --output-filename radial_segment_3d_fuzziness_005_wodt_seed1.csv > logs/radial_segment_3d_fuzziness_005_wodt_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_005_wodt_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_005 --seed-index 2 --model wodt --output-filename radial_segment_3d_fuzziness_005_wodt_seed2.csv > logs/radial_segment_3d_fuzziness_005_wodt_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_007_cart_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_007 --seed-index 0 --model cart --output-filename radial_segment_3d_fuzziness_007_cart_seed0.csv > logs/radial_segment_3d_fuzziness_007_cart_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_007_cart_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_007 --seed-index 1 --model cart --output-filename radial_segment_3d_fuzziness_007_cart_seed1.csv > logs/radial_segment_3d_fuzziness_007_cart_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_007_cart_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_007 --seed-index 2 --model cart --output-filename radial_segment_3d_fuzziness_007_cart_seed2.csv > logs/radial_segment_3d_fuzziness_007_cart_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_007_co2_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_007 --seed-index 0 --model co2 --output-filename radial_segment_3d_fuzziness_007_co2_seed0.csv > logs/radial_segment_3d_fuzziness_007_co2_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_007_co2_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_007 --seed-index 1 --model co2 --output-filename radial_segment_3d_fuzziness_007_co2_seed1.csv > logs/radial_segment_3d_fuzziness_007_co2_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_007_co2_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_007 --seed-index 2 --model co2 --output-filename radial_segment_3d_fuzziness_007_co2_seed2.csv > logs/radial_segment_3d_fuzziness_007_co2_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_007_hhcart_a_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_007 --seed-index 0 --model hhcart_a --output-filename radial_segment_3d_fuzziness_007_hhcart_a_seed0.csv > logs/radial_segment_3d_fuzziness_007_hhcart_a_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_007_hhcart_a_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_007 --seed-index 1 --model hhcart_a --output-filename radial_segment_3d_fuzziness_007_hhcart_a_seed1.csv > logs/radial_segment_3d_fuzziness_007_hhcart_a_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_007_hhcart_a_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_007 --seed-index 2 --model hhcart_a --output-filename radial_segment_3d_fuzziness_007_hhcart_a_seed2.csv > logs/radial_segment_3d_fuzziness_007_hhcart_a_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_007_hhcart_d_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_007 --seed-index 0 --model hhcart_d --output-filename radial_segment_3d_fuzziness_007_hhcart_d_seed0.csv > logs/radial_segment_3d_fuzziness_007_hhcart_d_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_007_hhcart_d_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_007 --seed-index 1 --model hhcart_d --output-filename radial_segment_3d_fuzziness_007_hhcart_d_seed1.csv > logs/radial_segment_3d_fuzziness_007_hhcart_d_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_007_hhcart_d_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_007 --seed-index 2 --model hhcart_d --output-filename radial_segment_3d_fuzziness_007_hhcart_d_seed2.csv > logs/radial_segment_3d_fuzziness_007_hhcart_d_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_007_oc1_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_007 --seed-index 0 --model oc1 --output-filename radial_segment_3d_fuzziness_007_oc1_seed0.csv > logs/radial_segment_3d_fuzziness_007_oc1_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_007_oc1_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_007 --seed-index 1 --model oc1 --output-filename radial_segment_3d_fuzziness_007_oc1_seed1.csv > logs/radial_segment_3d_fuzziness_007_oc1_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_007_oc1_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_007 --seed-index 2 --model oc1 --output-filename radial_segment_3d_fuzziness_007_oc1_seed2.csv > logs/radial_segment_3d_fuzziness_007_oc1_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_007_randcart_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_007 --seed-index 0 --model randcart --output-filename radial_segment_3d_fuzziness_007_randcart_seed0.csv > logs/radial_segment_3d_fuzziness_007_randcart_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_007_randcart_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_007 --seed-index 1 --model randcart --output-filename radial_segment_3d_fuzziness_007_randcart_seed1.csv > logs/radial_segment_3d_fuzziness_007_randcart_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_007_randcart_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_007 --seed-index 2 --model randcart --output-filename radial_segment_3d_fuzziness_007_randcart_seed2.csv > logs/radial_segment_3d_fuzziness_007_randcart_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_007_ridge_cart_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_007 --seed-index 0 --model ridge_cart --output-filename radial_segment_3d_fuzziness_007_ridge_cart_seed0.csv > logs/radial_segment_3d_fuzziness_007_ridge_cart_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_007_ridge_cart_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_007 --seed-index 1 --model ridge_cart --output-filename radial_segment_3d_fuzziness_007_ridge_cart_seed1.csv > logs/radial_segment_3d_fuzziness_007_ridge_cart_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_007_ridge_cart_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_007 --seed-index 2 --model ridge_cart --output-filename radial_segment_3d_fuzziness_007_ridge_cart_seed2.csv > logs/radial_segment_3d_fuzziness_007_ridge_cart_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_007_wodt_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_007 --seed-index 0 --model wodt --output-filename radial_segment_3d_fuzziness_007_wodt_seed0.csv > logs/radial_segment_3d_fuzziness_007_wodt_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_007_wodt_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_007 --seed-index 1 --model wodt --output-filename radial_segment_3d_fuzziness_007_wodt_seed1.csv > logs/radial_segment_3d_fuzziness_007_wodt_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: radial_segment_3d_fuzziness_007_wodt_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset radial_segment_3d --folder fuzziness_007 --seed-index 2 --model wodt --output-filename radial_segment_3d_fuzziness_007_wodt_seed2.csv > logs/radial_segment_3d_fuzziness_007_wodt_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    wait
    ;;
  saddle_3d)
    echo "Launching: saddle_3d_fuzziness_000_randcart_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_000 --seed-index 0 --model randcart --output-filename saddle_3d_fuzziness_000_randcart_seed0.csv > logs/saddle_3d_fuzziness_000_randcart_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_000_ridge_cart_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_000 --seed-index 0 --model ridge_cart --output-filename saddle_3d_fuzziness_000_ridge_cart_seed0.csv > logs/saddle_3d_fuzziness_000_ridge_cart_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_003_hhcart_d_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_003 --seed-index 1 --model hhcart_d --output-filename saddle_3d_fuzziness_003_hhcart_d_seed1.csv > logs/saddle_3d_fuzziness_003_hhcart_d_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_003_oc1_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_003 --seed-index 1 --model oc1 --output-filename saddle_3d_fuzziness_003_oc1_seed1.csv > logs/saddle_3d_fuzziness_003_oc1_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_003_oc1_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_003 --seed-index 2 --model oc1 --output-filename saddle_3d_fuzziness_003_oc1_seed2.csv > logs/saddle_3d_fuzziness_003_oc1_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_003_randcart_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_003 --seed-index 0 --model randcart --output-filename saddle_3d_fuzziness_003_randcart_seed0.csv > logs/saddle_3d_fuzziness_003_randcart_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_003_randcart_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_003 --seed-index 1 --model randcart --output-filename saddle_3d_fuzziness_003_randcart_seed1.csv > logs/saddle_3d_fuzziness_003_randcart_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_003_randcart_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_003 --seed-index 2 --model randcart --output-filename saddle_3d_fuzziness_003_randcart_seed2.csv > logs/saddle_3d_fuzziness_003_randcart_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_003_ridge_cart_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_003 --seed-index 0 --model ridge_cart --output-filename saddle_3d_fuzziness_003_ridge_cart_seed0.csv > logs/saddle_3d_fuzziness_003_ridge_cart_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_003_ridge_cart_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_003 --seed-index 1 --model ridge_cart --output-filename saddle_3d_fuzziness_003_ridge_cart_seed1.csv > logs/saddle_3d_fuzziness_003_ridge_cart_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_003_ridge_cart_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_003 --seed-index 2 --model ridge_cart --output-filename saddle_3d_fuzziness_003_ridge_cart_seed2.csv > logs/saddle_3d_fuzziness_003_ridge_cart_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_003_wodt_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_003 --seed-index 0 --model wodt --output-filename saddle_3d_fuzziness_003_wodt_seed0.csv > logs/saddle_3d_fuzziness_003_wodt_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_003_wodt_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_003 --seed-index 1 --model wodt --output-filename saddle_3d_fuzziness_003_wodt_seed1.csv > logs/saddle_3d_fuzziness_003_wodt_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_003_wodt_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_003 --seed-index 2 --model wodt --output-filename saddle_3d_fuzziness_003_wodt_seed2.csv > logs/saddle_3d_fuzziness_003_wodt_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_005_cart_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_005 --seed-index 0 --model cart --output-filename saddle_3d_fuzziness_005_cart_seed0.csv > logs/saddle_3d_fuzziness_005_cart_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_005_cart_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_005 --seed-index 1 --model cart --output-filename saddle_3d_fuzziness_005_cart_seed1.csv > logs/saddle_3d_fuzziness_005_cart_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_005_cart_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_005 --seed-index 2 --model cart --output-filename saddle_3d_fuzziness_005_cart_seed2.csv > logs/saddle_3d_fuzziness_005_cart_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_005_co2_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_005 --seed-index 0 --model co2 --output-filename saddle_3d_fuzziness_005_co2_seed0.csv > logs/saddle_3d_fuzziness_005_co2_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_005_co2_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_005 --seed-index 1 --model co2 --output-filename saddle_3d_fuzziness_005_co2_seed1.csv > logs/saddle_3d_fuzziness_005_co2_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_005_co2_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_005 --seed-index 2 --model co2 --output-filename saddle_3d_fuzziness_005_co2_seed2.csv > logs/saddle_3d_fuzziness_005_co2_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_005_hhcart_a_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_005 --seed-index 0 --model hhcart_a --output-filename saddle_3d_fuzziness_005_hhcart_a_seed0.csv > logs/saddle_3d_fuzziness_005_hhcart_a_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_005_hhcart_a_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_005 --seed-index 1 --model hhcart_a --output-filename saddle_3d_fuzziness_005_hhcart_a_seed1.csv > logs/saddle_3d_fuzziness_005_hhcart_a_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_005_hhcart_a_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_005 --seed-index 2 --model hhcart_a --output-filename saddle_3d_fuzziness_005_hhcart_a_seed2.csv > logs/saddle_3d_fuzziness_005_hhcart_a_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_005_hhcart_d_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_005 --seed-index 0 --model hhcart_d --output-filename saddle_3d_fuzziness_005_hhcart_d_seed0.csv > logs/saddle_3d_fuzziness_005_hhcart_d_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_005_hhcart_d_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_005 --seed-index 1 --model hhcart_d --output-filename saddle_3d_fuzziness_005_hhcart_d_seed1.csv > logs/saddle_3d_fuzziness_005_hhcart_d_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_005_hhcart_d_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_005 --seed-index 2 --model hhcart_d --output-filename saddle_3d_fuzziness_005_hhcart_d_seed2.csv > logs/saddle_3d_fuzziness_005_hhcart_d_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_005_oc1_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_005 --seed-index 0 --model oc1 --output-filename saddle_3d_fuzziness_005_oc1_seed0.csv > logs/saddle_3d_fuzziness_005_oc1_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_005_oc1_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_005 --seed-index 1 --model oc1 --output-filename saddle_3d_fuzziness_005_oc1_seed1.csv > logs/saddle_3d_fuzziness_005_oc1_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_005_oc1_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_005 --seed-index 2 --model oc1 --output-filename saddle_3d_fuzziness_005_oc1_seed2.csv > logs/saddle_3d_fuzziness_005_oc1_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_005_randcart_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_005 --seed-index 0 --model randcart --output-filename saddle_3d_fuzziness_005_randcart_seed0.csv > logs/saddle_3d_fuzziness_005_randcart_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_005_randcart_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_005 --seed-index 1 --model randcart --output-filename saddle_3d_fuzziness_005_randcart_seed1.csv > logs/saddle_3d_fuzziness_005_randcart_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_005_randcart_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_005 --seed-index 2 --model randcart --output-filename saddle_3d_fuzziness_005_randcart_seed2.csv > logs/saddle_3d_fuzziness_005_randcart_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_005_ridge_cart_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_005 --seed-index 0 --model ridge_cart --output-filename saddle_3d_fuzziness_005_ridge_cart_seed0.csv > logs/saddle_3d_fuzziness_005_ridge_cart_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_005_ridge_cart_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_005 --seed-index 1 --model ridge_cart --output-filename saddle_3d_fuzziness_005_ridge_cart_seed1.csv > logs/saddle_3d_fuzziness_005_ridge_cart_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_005_ridge_cart_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_005 --seed-index 2 --model ridge_cart --output-filename saddle_3d_fuzziness_005_ridge_cart_seed2.csv > logs/saddle_3d_fuzziness_005_ridge_cart_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_005_wodt_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_005 --seed-index 0 --model wodt --output-filename saddle_3d_fuzziness_005_wodt_seed0.csv > logs/saddle_3d_fuzziness_005_wodt_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_005_wodt_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_005 --seed-index 1 --model wodt --output-filename saddle_3d_fuzziness_005_wodt_seed1.csv > logs/saddle_3d_fuzziness_005_wodt_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_005_wodt_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_005 --seed-index 2 --model wodt --output-filename saddle_3d_fuzziness_005_wodt_seed2.csv > logs/saddle_3d_fuzziness_005_wodt_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_007_cart_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_007 --seed-index 0 --model cart --output-filename saddle_3d_fuzziness_007_cart_seed0.csv > logs/saddle_3d_fuzziness_007_cart_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_007_cart_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_007 --seed-index 1 --model cart --output-filename saddle_3d_fuzziness_007_cart_seed1.csv > logs/saddle_3d_fuzziness_007_cart_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_007_cart_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_007 --seed-index 2 --model cart --output-filename saddle_3d_fuzziness_007_cart_seed2.csv > logs/saddle_3d_fuzziness_007_cart_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_007_co2_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_007 --seed-index 0 --model co2 --output-filename saddle_3d_fuzziness_007_co2_seed0.csv > logs/saddle_3d_fuzziness_007_co2_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_007_co2_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_007 --seed-index 1 --model co2 --output-filename saddle_3d_fuzziness_007_co2_seed1.csv > logs/saddle_3d_fuzziness_007_co2_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_007_co2_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_007 --seed-index 2 --model co2 --output-filename saddle_3d_fuzziness_007_co2_seed2.csv > logs/saddle_3d_fuzziness_007_co2_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_007_hhcart_a_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_007 --seed-index 0 --model hhcart_a --output-filename saddle_3d_fuzziness_007_hhcart_a_seed0.csv > logs/saddle_3d_fuzziness_007_hhcart_a_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_007_hhcart_a_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_007 --seed-index 1 --model hhcart_a --output-filename saddle_3d_fuzziness_007_hhcart_a_seed1.csv > logs/saddle_3d_fuzziness_007_hhcart_a_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_007_hhcart_a_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_007 --seed-index 2 --model hhcart_a --output-filename saddle_3d_fuzziness_007_hhcart_a_seed2.csv > logs/saddle_3d_fuzziness_007_hhcart_a_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_007_hhcart_d_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_007 --seed-index 0 --model hhcart_d --output-filename saddle_3d_fuzziness_007_hhcart_d_seed0.csv > logs/saddle_3d_fuzziness_007_hhcart_d_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_007_hhcart_d_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_007 --seed-index 1 --model hhcart_d --output-filename saddle_3d_fuzziness_007_hhcart_d_seed1.csv > logs/saddle_3d_fuzziness_007_hhcart_d_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_007_hhcart_d_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_007 --seed-index 2 --model hhcart_d --output-filename saddle_3d_fuzziness_007_hhcart_d_seed2.csv > logs/saddle_3d_fuzziness_007_hhcart_d_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_007_oc1_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_007 --seed-index 0 --model oc1 --output-filename saddle_3d_fuzziness_007_oc1_seed0.csv > logs/saddle_3d_fuzziness_007_oc1_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_007_oc1_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_007 --seed-index 1 --model oc1 --output-filename saddle_3d_fuzziness_007_oc1_seed1.csv > logs/saddle_3d_fuzziness_007_oc1_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_007_oc1_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_007 --seed-index 2 --model oc1 --output-filename saddle_3d_fuzziness_007_oc1_seed2.csv > logs/saddle_3d_fuzziness_007_oc1_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_007_randcart_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_007 --seed-index 0 --model randcart --output-filename saddle_3d_fuzziness_007_randcart_seed0.csv > logs/saddle_3d_fuzziness_007_randcart_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_007_randcart_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_007 --seed-index 1 --model randcart --output-filename saddle_3d_fuzziness_007_randcart_seed1.csv > logs/saddle_3d_fuzziness_007_randcart_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_007_randcart_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_007 --seed-index 2 --model randcart --output-filename saddle_3d_fuzziness_007_randcart_seed2.csv > logs/saddle_3d_fuzziness_007_randcart_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_007_ridge_cart_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_007 --seed-index 0 --model ridge_cart --output-filename saddle_3d_fuzziness_007_ridge_cart_seed0.csv > logs/saddle_3d_fuzziness_007_ridge_cart_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_007_ridge_cart_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_007 --seed-index 1 --model ridge_cart --output-filename saddle_3d_fuzziness_007_ridge_cart_seed1.csv > logs/saddle_3d_fuzziness_007_ridge_cart_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_007_ridge_cart_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_007 --seed-index 2 --model ridge_cart --output-filename saddle_3d_fuzziness_007_ridge_cart_seed2.csv > logs/saddle_3d_fuzziness_007_ridge_cart_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_007_wodt_seed0.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_007 --seed-index 0 --model wodt --output-filename saddle_3d_fuzziness_007_wodt_seed0.csv > logs/saddle_3d_fuzziness_007_wodt_seed0.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_007_wodt_seed1.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_007 --seed-index 1 --model wodt --output-filename saddle_3d_fuzziness_007_wodt_seed1.csv > logs/saddle_3d_fuzziness_007_wodt_seed1.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    echo "Launching: saddle_3d_fuzziness_007_wodt_seed2.csv"
    srun -N1 -n1 python run_depth_sweep_parallel.py --dataset saddle_3d --folder fuzziness_007 --seed-index 2 --model wodt --output-filename saddle_3d_fuzziness_007_wodt_seed2.csv > logs/saddle_3d_fuzziness_007_wodt_seed2.log 2>&1 &
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    wait
    ;;
esac
