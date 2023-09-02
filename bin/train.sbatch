#!/bin/bash --login
#SBATCH --time 10:00:00
#SBATCH --cpus-per-task=4  
#SBATCH --mem-per-cpu=100G 
#SBATCH --constraint=intel
#SBATCH --partition=batch 
#SBATCH --mail-type=ALL
#SBATCH --output=bin/%x-%j-slurm.out
#SBATCH --error=bin/%x-%j-slurm.err

# entire script fails if single command fails
set -e

# activate the conda environment
module purge
export PROJECT_DIR="$PWD"
conda activate /ibex/scratch/medinils/breast_data/env

# launch the training script
python /ibex/scratch/medinils/breast_data/src/train.py
#python /ibex/scratch/medinils/breast_data/src/train_2.py
