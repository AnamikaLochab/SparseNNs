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
BATCH_SIZE=60
LR=0.002
L1_REG=0.001
VALID_FREQ=1
END_ITER=500
PRUNE_TYPE="l1_reg"

# Run Python script with wandb project initialization
python3 L1.py --dataset $DATASET --arch_type $ARCH --prune_type $PRUNE_TYPE \
 --batch_size $BATCH_SIZE --lr $LR --l1_lambda $L1_REG --end_iter $END_ITER --valid_freq $VALID_FREQ





