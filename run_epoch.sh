#!/usr/bin/env bash

#SBATCH --job-name=new1ph
#SBATCH --ntasks=300
#SBATCH --partition=nodes
#SBATCH --time=00-00:15:00 # Time limit (DD-HH:MM:SS)
#SBATCH --account=pet-icepic-2024
#SBATCH --output=%x_%A_%a.log
#SBATCH --mail-user=ethan.attwood@york.ac.uk
#SBATCH --mail-type=ALL # Mail events (NONE, BEGIN, END, FAIL, ALL)

# Abort if any command fails
set -e

# Purge any previously loaded modules
module purge

# Load modules
module load OpenMPI/4.1.6-GCC-13.2.0

file_path=/users/era536/scratch/epoch/testing_sep25/1pitch/

srun /users/era536/scratch/epoch/epoch_installation/epoch1d/bin/epoch1d <<< $file_path

# Job completed
echo '\n'Job completed at `date`
