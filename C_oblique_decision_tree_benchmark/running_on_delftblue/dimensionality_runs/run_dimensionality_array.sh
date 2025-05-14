#!/bin/bash
#SBATCH --job-name=DepthSweepDimensionality
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=29
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1G
#SBATCH --account=Research-TPM-MAS

module load 2023r1
module load python
module load py-pip

export OMP_NUM_THREADS=1

mkdir -p logs

MAX_PARALLEL=29

while read -r JOB; do
    bash -c "$JOB" &
    while [ "$(jobs -r | wc -l)" -ge "$MAX_PARALLEL" ]; do
        wait -n
    done
done < job_list_dimensionality.txt

wait
echo "=== All dimensionality jobs completed ==="