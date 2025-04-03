#!/bin/bash

#SBATCH --job-name="Open_Exploration"
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=2GB
#SBATCH --account=education-tpm-msc-epa

module load 2023r1
module load openmpi
module load python
module load py-numpy
module load py-scipy
module load py-mpi4py
module load py-pip

# Install required packages
pip install ema_workbench networkx openpyxl xlrd
pip install --user --upgrade ema_workbench

srun python Step1_Simulate_Open_Exploration.py > simulation.log
