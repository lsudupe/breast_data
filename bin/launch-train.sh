#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="/ibex/scratch/medinils/breast_data"

# creates a separate directory for each job
JOB_NAME=train-graph-model
mkdir -p /ibex/scratch/medinils/breast_data/results"$JOB_NAME"

# launch the training job
sbatch --job-name "$JOB_NAME" "$breast_dir"/bin/train.sbatch "$breast_dir"/src/train.py 
#sbatch --job-name "$JOB_NAME" "$breast_dir"/bin/train.sbatch "$breast_dir"/src/Node2vec.py

