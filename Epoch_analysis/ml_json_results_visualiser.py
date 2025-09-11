import argparse
from pathlib import Path
import json

def plotResults(resultsFile : Path):
    with open(resultsFile, "r") as f:
        parser = json.load(f)
        print(parser)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--file",
        action="store",
        help="JSON file of results output.",
        required = True,
        type=Path
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot a bar chart of results.",
        required = False
    )

    args = parser.parse_args()

    if args.plot:
        plotResults(args.file)