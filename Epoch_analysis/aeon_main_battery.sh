#!/usr/bin/env bash
#SBATCH --job-name=1-conv
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=nodes
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=era536@york.ac.uk
#SBATCH --time=00-23:59:00
#SBATCH --account=pet-icepic-2024
#SBATCH --output=/home/era536/Documents/Epoch/Data/2026_analysis/tsr_aeon/logs/1-conv_%j.log


echo "Job: 1-conv started at $(date)"
echo "Requested Time: 00-23:59:00"

# Abort if any command fails
set -e

# purge any existing modules
module purge

# Load modules
module load Python/3.11.3-GCCcore-12.3.0
source .venv/bin/activate

python /users/era536/scratch/repos/epoch_work/Epoch_analysis/ts_aeon_regression_development.py --dir /users/era536/scratch/epoch/ml_feb26/combined_data_full/ --inputSpectra Magnetic_Field_Bz/power/powerByFrequency Electric_Field_Ex/power/powerByFrequency Electric_Field_Ey/power/powerByFrequency --outputFields B0strength pitch backgroundDensity beamFraction --logFields backgroundDensity beamFraction --algorithms aeon.DummyRegressor aeon.HydraRegressor aeon.RocketRegressor aeon.MiniRocketRegressor aeon.MultiRocketRegressor aeon.MultiRocketHydraRegressor aeon.RandomIntervalSpectralEnsembleRegressor aeon.TimeSeriesForestRegressor aeon.QUANTRegressor aeon.KNeighborsTimeSeriesRegressor aeon.RDSTRegressor --cvStrategy LeaveOneOut --resultsFilepath /users/era536/scratch/epoch/publications/modelling_nl_spectra/tsr_2/full_spectra/aeon_main_battery.json --doPlot --noTitle --nThreads $SLURM_CPUS_PER_TASK

echo "Job finished at $(date)"
        