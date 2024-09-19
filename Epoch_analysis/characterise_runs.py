import epydeck
import argparse
from pathlib import Path
import os

if __name__ == "__main__":
    # Run python setup.py -h for list of possible arguments
    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--dir",
        action="store",
        help="Directory containing all simulation runs.",
        required = True,
        type=Path
    )
    args = parser.parse_args()

    directory = args.dir

    print("Run,B0,B_angle,Bkgd_density,Frac_beam")
    topLevel = True
    for run, _, _ in os.walk(directory):
        if(topLevel): # Dumb way to exclude top level dir
            topLevel = False
            continue
        with open(Path(run) / "input.deck") as inputDeck:
            input = epydeck.loads(inputDeck.read())
            print(f"{Path(run).name},{input['constant']['b_strength']},{input['constant']['b_angle']},{input['constant']['background_density']},{input['constant']['frac_beam']}")