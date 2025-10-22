#!/bin/bash

# --- Slurm Job Directives ---
#SBATCH --job-name=unet3D_sanity_check
#SBATCH --partition=a100-test          # IMPORTANT: Use the test partition
#SBATCH --time=00:20:00              # IMPORTANT: Set time limit to 20 mins or less
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=l.strietzel@student.uq.edu.au

echo "=========================================================="
echo "Starting UNet3D Sanity Check"
echo "=========================================================="

# 1. Initialize Conda
source /home/Student/s4979785/miniconda/etc/profile.d/conda.sh
conda activate brains-gan-new

# 2. Define the project directory
PROJECT_DIR=$SLURM_SUBMIT_DIR

# 3. Run your training script with SMALLER parameters
python ${PROJECT_DIR}/train.py \
    --epochs 1 \
    --batch-size 4 \
    --data-dir /home/groups/comp3710/HipMRI_Study_open \
    --checkpoint-dir ${PROJECT_DIR}/checkpoints_test

echo "=========================================================="
echo "Job finished"
echo "=========================================================="