#!/bin/bash
#SBATCH -A berzelius-2024-110
#SBATCH --job-name=FedAvg-july_31_cifar10_non_IID_50_3_noise_0.8
#SBATCH --output=FedAvg-july_31_cifar10_non_IID_50_3_noise_0.8.txt
#SBATCH --time=04:00:00  # Job will run for 4 hr 00 min
#SBATCH --gres=gpu:1  # Request 1 GPU

# Load necessary modules or activate your environment
source activate personalized_fl

python main.py --fl_algorithm=FedAvg --num_users_perGR=50 --num_labels=3 --num_users=50 --global_iters=200 --noise_level=0.8