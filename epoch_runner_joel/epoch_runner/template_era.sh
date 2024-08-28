#!/usr/bin/env bash

#SBATCH --job-name={job_name}
#SBATCH --ntasks=96
#SBATCH --partition=nodes
#SBATCH --time=02-00:00:00 # Time limit (DD-HH:MM:SS)
#SBATCH --account=pet-icepic-2024
#SBATCH --output={campaign_path}/%x_%A_%a.log
#SBATCH --mail-user=ethan.attwood@york.ac.uk
#SBATCH --mail-type=ALL # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --array=1-{array_range_max} # Array range

# Abort if any command fails
set -e

# Purge any previously loaded modules
module purge

# Load modules
module load Python/3.11.3-GCCcore-12.3.0
module load OpenMPI/3.1.3-GCC-8.2.0-2.31.1

#source .venv/bin/activate
#python -c 'import os; print(os.environ['VIRTUAL_ENV'])'

# Prestart
cd {epoch_dir}/{epoch_version}
echo Working directory: `pwd`
echo Running job on host:
echo -e '\t'`hostname` at `date`'\n'

file_path={file_path}

# Check if the paths file exists
if [[ ! -f "$file_path" ]]; then
    echo "File not found: $file_path"
    exit 1
fi

# Read the specific line from the file
deck_path=$(awk "NR==$SLURM_ARRAY_TASK_ID" "$file_path")

# Check if the line was successfully read
if [[ -z "$deck_path" ]]; then
    echo "No line found at index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Running deck file: $deck_path"

# Script initialisation
mpiexec -n $SLURM_NTASKS {epoch_dir}/{epoch_version}/bin/{epoch_version} <<< $deck_path

# Job completed
echo '\n'Job completed at `date`