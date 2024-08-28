from pathlib import Path
import argparse
import sys

import epydeck
import epyscan
import epyfunc

# Some arguments that can be passed into this function via the terminal
# Run python setup.py -h for list of possible arguments
parser = argparse.ArgumentParser("parser")
parser.add_argument(
    "--verbose",
    action="store_true",
    help="Displays verbose output including directory locations.",
)
parser.add_argument(
    "--test",
    action="store_true",
    help="Simply tests that the script can run without executing the simulations.",
)
parser.add_argument(
    "--dir",
    action="store",
    help="Parent directory containing epoch script (and optionally, installation) in subfolders.",
)
parser.add_argument(
    "--epochPath",
    action="store",
    help="Directory of epoch installation, if different to dir",
)
parser.add_argument(
    "--numSimulations",
    action="store",
    type=int,
    help="Number of simulations to run",
    required=True
)
args = parser.parse_args()

# Paths setup
top_level_dir = Path(args.dir) if args.dir is not None else Path("/users/era536/scratch/epoch")
script_path = top_level_dir / "epoch_runner"
template_deck_filename = script_path / "template_era.deck"
template_jobscript_filename = script_path / "template_era.sh"
simulation_path_filename = script_path / "paths.txt"
campaign_path = top_level_dir / "example_campaign"

epoch_version = "epoch1d"
epoch_path = args.epochPath if args.epochPath is not None else Path(top_level_dir / "epoch_installation")

if args.verbose:
    print(f"Script directory: {script_path}")
    print(f"Template deck: {template_deck_filename}")
    print(f"Template jobscript: {template_jobscript_filename}")
    print(f"Simulation paths: {simulation_path_filename}")
    print(f"Campaign folder: {campaign_path}")
    print(f"Epoch version: {epoch_version}")
    print(f"Epoch location: {epoch_path}")

# INITIAL RANDOM SAMPLING
# ----------------------------------------------------------------------------------------
with open(template_deck_filename) as f:
    deck = epydeck.load(f)

# b_pitch_range = 10 # +- range of B angle from perpendicular to x, in degrees
# bx_range = np.cos(np.deg2rad(10)) * 

parameters = {
    "constant:background_density": {"min": 1.0e18, "max": 1.0e20, "log": True},
    "constant:frac_beam": {"min": 1.0e-4, "max": 1.0e-2, "log": True},
    "constant:b_strength": {"min": 0.5, "max": 5.0, "log": False},
    "constant:b_angle": {"min": 80, "max": 100, "log": False}, # Angle of B relative to x
}

# Sets up sampling of simulation and specifies number of times to run each simulation
hypercube_samples = epyscan.LatinHypercubeSampler(parameters).sample(args.numSimulations)

# Takes in the folder and template and starts a counter so each new simulation gets saved to a new folder
campaign = epyscan.Campaign(deck, campaign_path)

# Randomly samples the parameter space and creates folders for each simulation
paths = [campaign.setup_case(sample) for sample in hypercube_samples]

# Save the paths to a file on separate lines
with open(simulation_path_filename, "w") as f:
    [f.write(str(path) + "\n") for path in paths]

if args.test:
    print("Successfully validated and created filepaths")
    sys.exit()

# EXECUTE THE JOB
# ----------------------------------------------------------------------------------------
job = epyfunc.SlurmJob(args.verbose)

job.enqueue_array_job(
    epoch_path=epoch_path,
    epoch_version=epoch_version,
    campaign_path=campaign_path,
    file_path=simulation_path_filename,
    template_path=template_jobscript_filename,
    n_runs=len(paths),
    job_name="setup_run",
)

job.poll_jobs(interval=2)
_, failed_jobs = job.get_job_results()

if failed_jobs:
    print("The following jobs failed", failed_jobs)
    sys.exit("Initial/Setup simulation run failed. See job log files")
