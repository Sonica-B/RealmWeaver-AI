#!/bin/bash
#SBATCH --job-name=gameworld_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%j.out

# Load modules
module load anaconda3
module load cuda/12.1

# Activate environment
source activate gameworldgen

# Run script
python $1
