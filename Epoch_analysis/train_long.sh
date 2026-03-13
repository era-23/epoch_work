#!/usr/bin/env bash
#SBATCH --job-name=long_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=nodes
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=era536@york.ac.uk
#SBATCH --time=02-00:00:00
#SBATCH --account=pet-icepic-2024
#SBATCH --output=./long_train_%j.log


echo "Job: long_train started at $(date)"
echo "Requested Time: 02-00:00:00"

# Abort if any command fails
set -e

# purge any existing modules
module purge

# Load modules
module load Python/3.11.3-GCCcore-12.3.0
source .venv/bin/activate

python train.py --epochs 100

echo "Job finished at $(date)"
        