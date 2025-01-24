import argparse
from pathlib import Path

def collate_growth_rate_data(dataDir : Path, outFile : Path):
    return NotImplementedError()

if __name__ == "__main__":

    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--dir",
        action="store",
        help="Directory containing simulation metadata, plots and calculated data, e.g. growth rates.",
        required = True,
        type=Path
    )

    args = parser.parse_args()