#!/bin/bash --login

# everything fails if any single command failes
set -e
# creates the conda enviroment
PROJECT_DIR=$PWD
conda env create --prefix $PROJECT_DIR/env --file $PROJECT_DIR/enviroment.yml --force

# activate conda env before installing PyTorch Geometric via pip
conda activate $PROJECT_DIR/env
TORCH=2.0.
CUDA=11.7
python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv --no-cache-dir --no-index --find-links https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
python -m pip install torch-geometric --no-cache-dir
