#!/bin/bash

# --- Slurm Job Directives ---
#SBATCH --job-name=unet_train
#SBATCH --partition=a100         # Request a powerful A100 GPU partition
#SBATCH --time=08:00:00          # Set a reasonable time limit (e.g., 8 hours)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        # More CPUs for faster data loading
#SBATCH --gres=gpu:1             # Request 1 A100 GPU
#SBATCH --mem=32G                # Request more RAM
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=l.strietzel@student.uq.edu.au

echo "=========================================================="
echo "Starting UNet Training Job"
echo "=========================================================="

source /home/Student/s4979785/miniconda/etc/profile.d/conda.sh
conda activate brains-gan

PROJECT_DIR=$SLURM_SUBMIT_DIR

# 3. Run your training script with absolute paths
python ${PROJECT_DIR}/train.py \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.0001 \
    --data-dir /home/groups/comp3710/OASIS \
    --checkpoint-dir ${PROJECT_DIR}/checkpoints

echo "=========================================================="
echo "Job finished"
echo "=========================================================="