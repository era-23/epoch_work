import glob
import epydeck
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import percentileofscore
import os

def print_parameters_with_percentiles(statsFolder : Path):

    print(f"Checking {statsFolder} for folders named for runs, each containing an input.deck....")

    folders = glob.glob(str(statsFolder / "run_*"))
    # fnames = [Path(f).name for f in folders]
    allB0 = dict.fromkeys(folders)
    allPitch = dict.fromkeys(folders)
    allDensity = dict.fromkeys(folders)
    allFastFrac = dict.fromkeys(folders)
    
    for folder in folders:       
        inputDeck = {}
        with open(str(os.path.join(folder, "input.deck"))) as id:
            inputDeck = epydeck.loads(id.read())

        allB0[folder] = inputDeck["constant"]["b0_strength"]
        allPitch[folder] = inputDeck["constant"]["pitch"]
        allDensity[folder] = inputDeck["constant"]["background_density"]
        allFastFrac[folder] = inputDeck["constant"]["frac_beam"]
    
    for run in folders:
        runName = Path(run).name
        b0_perc = percentileofscore(list(allB0.values()), allB0[run])
        pitch_perc = percentileofscore(list(allPitch.values()), allPitch[run])
        density_perc = percentileofscore(list(allDensity.values()), allDensity[run])
        fastFrac_perc = percentileofscore(list(allFastFrac.values()), allFastFrac[run])

        density_log_perc = percentileofscore(np.log10(list(allDensity.values())), np.log10(allDensity[run]))
        fastFrac_log_perc = percentileofscore(np.log10(list(allFastFrac.values())), np.log10(allFastFrac[run]))

        print(f"Simulation {runName} values (percentiles): B0 = {allB0[run]} ({b0_perc}), pitch = {allPitch[run]} ({pitch_perc}), density = {allDensity[run]} ({density_perc}), fast frac = {allFastFrac[run]} ({fastFrac_perc})")
        print(f"Simulation {runName} values (percentiles): B0 = {allB0[run]} ({b0_perc}), pitch = {allPitch[run]} ({pitch_perc}), density = {np.log10(allDensity[run])} ({density_log_perc}), fast frac = {np.log10(allFastFrac[run])} ({fastFrac_log_perc})")


if __name__ == "__main__":
    # Run python setup.py -h for list of possible arguments
    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--dir",
        action="store",
        help="Directory containing a folder for each simulation with an input.deck file.",
        required = True,
        type=Path
    )
    args = parser.parse_args()

    directory = args.dir

    print_parameters_with_percentiles(directory)