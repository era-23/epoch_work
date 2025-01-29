import argparse
import glob
import os
import xarray as xr
from pathlib import Path

def process_null_simulations(directory : Path):

    data = xr.open_mfdataset(
        str(directory / "*.nc"),
        data_vars="all",
        combine="nested",
        concat_dim="Energy/time",
        engine="netcdf4"
    )

    print(data.variables)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--dir",
        action="store",
        help="Directory containing netCDF files of simulation output.",
        required = True,
        type=Path
    )

    args = parser.parse_args()

    process_null_simulations(args.dir)