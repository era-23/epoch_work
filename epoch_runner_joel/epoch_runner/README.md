# Epoch Runner
For this script to work correctly you need to have the following folder structure:
```
.
├── epoch_runner
│ ├── setup.py # The script file to run
│ ├── template.deck # The template deck
│ ├── template.sh # The Slurm/Viking template jobscript
│ ├── README.md
| ├── epyfunc # helper functions
│ │   └── __init__.py
│ └── ...
├── epoch
│ ├── epoch1d
│ ├── epoch2d
│ └── epoch3d
└── ...
```
# Setup
- Install all python packages into a venv (sorry there's no requirements.txt yet)
- Run `python setup.py`

# Explanation of setup.py
1. Run `epydeck` and generate the file paths for the input decks
2. Create a new jobscript within the top level parent folder for the campaign and do the following:
	1. Rename the `job_name` of the sbatch job
	2. Rename the `campaign_path` to the location where the campaign runs
	3. Rename the `array_range_max` to the total number of sims to run
	4. Rename the `epoch_dir` to the directory containing all EPOCH. e.g. `~/scratch/epoch`
	5. Rename the `epoch_version` to the epoch version. e.g. `epoch2d`
	6. Rename the `file_path` to location of the txt file containing the deck paths. e.g. `paths.txt`
