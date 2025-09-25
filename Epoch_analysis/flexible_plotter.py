import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import epoch_utils

def power_spectrum(dataFile : Path, field : str):
    
    data = xr.open_datatree(
            dataFile,
            engine="netcdf4"
    )

    B0 = float(data.attrs["B0strength"])
    s = field.split("/")
    group = "/" if len(s) == 1 else '/'.join(s[:-1])
    fieldName = s[-1]
    power_trace = data[group].variables[fieldName]
    fieldData = power_trace.data.astype("float")
    coords = data[group].coords[power_trace.dims[0]]
    fieldName = epoch_utils.fieldNameToText(field)
    
    fig, axs = plt.subplots(figsize=(15, 10))
    axs.plot(coords, fieldData)
    axs.set_xticks(ticks=np.arange(np.floor(coords[0]), np.ceil(coords[-1])+1.0, 1.0), minor=True)
    axs.grid(which='both', axis='x')
    axs.set_xlabel(r"Frequency [$\omega_{ci}$]")
    axs.set_ylabel(f"Sum of power in {fieldName} over all k [T]")
    plt.show()

    fig, axs = plt.subplots(figsize=(15, 10))
    axs.plot(coords, 10.0 * np.log10(fieldData / B0))
    axs.set_xticks(ticks=np.arange(np.floor(coords[0]), np.ceil(coords[-1])+1.0, 1.0), minor=True)
    axs.grid(which='both', axis='x')
    axs.set_xlabel(r"Frequency [$\omega_{ci}$]")
    axs.set_ylabel(f"Sum of power in {fieldName} over all k [dB]")
    plt.show()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--dataFile",
        action="store",
        help="Filepath of simulation output data to plot, e.g. /data/run_23_stats.nc",
        required = True,
        type=Path
    )
    parser.add_argument(
        "--field",
        action="store",
        help="Field to plot, e.g. /Magnetic_Field_Bz/power/powerByFrequency",
        required = True,
        type=str
    )

    args = parser.parse_args()

    plt.rcParams.update({'axes.titlesize': 26.0})
    plt.rcParams.update({'axes.labelsize': 24.0})
    plt.rcParams.update({'xtick.labelsize': 20.0})
    plt.rcParams.update({'ytick.labelsize': 20.0})
    plt.rcParams.update({'legend.fontsize': 18.0})

    power_spectrum(args.dataFile, args.field)

