import glob
from pathlib import Path
import numpy as np
import argparse
import netCDF4 as nc
import xarray as xr
import epoch_utils as eu

def combine_spectra(dataDirectory : Path, outputDirectory : Path):

    # Get folders for each angle
    angle_folders = glob.glob(str(dataDirectory / "9[0-9]"))
    angles = [Path(f).name for f in angle_folders]
    angle_folders = glob.glob(str(dataDirectory / "9[0-9]/data/"))

    assert len(angle_folders) > 0

    # Get first angle
    first_angle_folder = Path(angle_folders[0])

    data_files = glob.glob(str(first_angle_folder / "*.nc"))
    individual_data_filenames = [Path(n).name for n in data_files]

    # Each simulations (at one angle)
    for filename in individual_data_filenames:
        
        sim_id = filename.removeprefix("run_").removesuffix("_stats.nc")

        all_sim_angle_datapaths = [Path(fp) / filename for fp in angle_folders]

        # Create single NC file for summary of combined spectra
        # Get existing spectra as 
        data_xr = xr.open_dataset(all_sim_angle_datapaths[0], engine="netcdf4")
        newData_xr = data_xr.copy(deep = True)
        newData_xr.drop_attrs("B0angle") # Now multivalued

        for angle_sim_path in all_sim_angle_datapaths:
            angle_data_xr = xr.open_dataset(angle_sim_path, engine="netcdf4")
            assert angle_data_xr.attrs["B0strength"] == newData_xr.attrs["B0strength"]
            assert angle_data_xr.attrs["backgroundDensity"] == newData_xr.attrs["backgroundDensity"]
            assert angle_data_xr.attrs["beamFraction"] == newData_xr.attrs["beamFraction"]

if __name__ == "__main__":
    # Run python setup.py -h for list of possible arguments
    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--dir",
        action="store",
        help="Directory containing analysis of all simulations. Expects folders of angles, each with data and plots folders.",
        required = True,
        type=Path
    )
    parser.add_argument(
        "--outputDir",
        action="store",
        help="Output directory.",
        required = False,
        type=Path
    )
    parser.add_argument(
        "--combineSpectra",
        action="store_true",
        help="Combine Ex, Ey and Bz spectra.",
        required = False
    )
    parser.add_argument(
        "--doIciness",
        action="store_true",
        help="Calculate ICEiness characteristics for combined spectra.",
        required = False
    )
    
    args = parser.parse_args()

    # if args.combineSpectra:
    #     dataFile = combine_spectra(args.dir, args.outputDir)
    # if args.doIciness:
    #     calculateIciness(args.dir, dataFile)

