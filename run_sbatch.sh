#!/bin/bash
#SBATCH -A berzelius-2024-110
#SBATCH --job-name=july_30_cifar10_IID
#SBATCH --output=july_30_cifar10_IID.txt
#SBATCH --time=04:00:00  # Job will run for 4 hr 00 min
#SBATCH --gres=gpu:1  # Request 1 GPU

# Load necessary modules or activate your environment
source activate personalized_fl

python main.py --wandb --num_global_iters=100 --local_iters=5 --num_users=100