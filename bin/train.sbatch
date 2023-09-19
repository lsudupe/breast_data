#!/bin/bash --login
#SBATCH --time=40:00:00
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=64G
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --output=bin/%x/%j-slurm.out
#SBATCH --error=bin/%x/%j-slurm.err

# entire script fails if single command fails
set -e

# activate the conda environment
module purge
conda activate /home/medinils/pytorch-gpu-data-science-project/env
module load cuda/11.7.0

# launch the training script
python /ibex/scratch/medinils/breast_data/src/graph_creation.py
python /ibex/scratch/medinils/breast_data/src/dataset.py
python /ibex/scratch/medinils/breast_data/src/gnn_playing.py
