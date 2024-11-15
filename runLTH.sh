#!/bin/bash
#SBATCH --job-name=pruning_project          # Job name
#SBATCH --output=logs/%x_%j.out             # Output file for stdout (%x = job-name, %j = job ID)
#SBATCH --error=logs/%x_%j.err              # Error file for stderr (%x = job-name, %j = job ID)
#SBATCH --partition=gorman-gpu                   # Partition (GPU in this case)
#SBATCH --gres=gpu:1                        # Request 1 GPU
#SBATCH --mem=16G                           # Memory per node


source ~/myenv/bin/activate
# module load python/3.8



# Set arguments for script
DATASET="mnist"
ARCH="fc1"
PRUNE_PERCENT=51
PRUNE_ITERATIONS=35
BATCH_SIZE=60
LR=0.0012
VALID_FREQ=1

# Run Python script with wandb project initialization
python3 LTH.py --dataset $DATASET --arch_type $ARCH --prune_percent $PRUNE_PERCENT \
    --prune_iterations $PRUNE_ITERATIONS --batch_size $BATCH_SIZE --lr $LR --valid_freq $VALID_FREQ
