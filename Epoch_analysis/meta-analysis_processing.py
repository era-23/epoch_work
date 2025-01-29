import argparse
# import glob
# import os
from pathlib import Path

def collate_growth_rate_data(dataDir : Path, outFile : Path, inputType : str = "netcdf", outputType : str = "csv"):
    return NotImplementedError()

def collate_peak_power_data(dataDir : Path, outFile : Path, outputType : str = "csv"):

    #stats_files = glob.glob(str(dataDir / "*_stats.nc") + os.path.sep)

    with open(outFile):

        return NotImplementedError

if __name__ == "__main__":

    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--dir",
        action="store",
        help="Directory containing simulation metadata, plots and calculated data, e.g. growth rates.",
        required = True,
        type=Path
    )
    parser.add_argument(
        "--outputDir",
        action="store",
        help="Directory to write output file. If absent, defaults to input dir.",
        required = False,
        type=Path
    )
    parser.add_argument(
        "--inputType",
        action="store",
        help="Input is in \'csv\' or \'netcdf\' format. Defaults to netcdf.",
        required = False,
        type=str
    )
    parser.add_argument(
        "--outputType",
        action="store",
        help="Input is in \'csv\' or \'netcdf\' format. Defaults to csv",
        required = False,
        type=str
    )

    args = parser.parse_args()