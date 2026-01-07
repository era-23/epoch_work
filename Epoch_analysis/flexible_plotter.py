import argparse
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import epoch_utils
import ml_utils

def all_power_spectra(dataFolder : Path, fields : list):

    # Get all datafiles
    data_files = glob.glob(str(dataFolder / "*.nc"))

    allSpectra = []

    allSpectra = ml_utils.read_data(
        data_files, 
        {f : [] for f in fields},
        with_names = True, 
        with_coords=True
    )
    
    b0 = np.array(ml_utils.read_data(
        data_files, 
        {"B0strength" : []},
        with_names = True
    )["B0strength"])
    
    # Check all corrdinates are aligned
    # coords = [v for k,v in allSpectra.items() if k.endswith("_coords")]
    # for i in range(len(coords)-1):
    #     assert np.allclose(coords[i], coords[i+1])

    for field in fields:
        
        # Raw data
        spectra = np.array([np.array(s) for s in allSpectra[field]])
        print(spectra.shape)

        # For each simulation, take log10 of power^2 / B0^2
        # for i in range(spectra.shape[0]):
        #     spectra[i] = np.log10(spectra[i]**2 / b0[i]**2)
        
        plt.subplots(figsize=(12,8))
        heatmap = plt.imshow(spectra, cmap="plasma", aspect='auto', origin='lower')
        # heatmap = plt.imshow(np.log10(spectra), cmap="plasma", aspect='auto', origin='lower')
        plt.title(f"{epoch_utils.fieldNameToText(field)}")
        plt.xlim(allSpectra[field + "_coords"][0][0], allSpectra[field + "_coords"][0][-1])
        plt.ylim(float(allSpectra["sim_ids"][0]), float(allSpectra["sim_ids"][-1]))
        plt.xlabel(r"Frequency/$\omega_\alpha$")
        plt.ylabel("Simulation ID")
        cbar = plt.colorbar(heatmap)
        cbar.ax.set_ylabel("power/T")
        plt.tight_layout()
        plt.show()
        
def power_spectrum(dataFile : Path, field : str, doPlot : bool = True):
    
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
    
    if doPlot:
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

    return fieldData, coords

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--dataFile",
        action="store",
        help="Filepath of simulation output data to plot, e.g. /data/run_23_stats.nc",
        required = False,
        type=Path
    )
    parser.add_argument(
        "--dataFolder",
        action="store",
        help="Filepath of folder of simulation output data to plot, e.g. /data/",
        required = False,
        type=Path
    )
    parser.add_argument(
        "--fields",
        action="store",
        help="Field(s) to plot, e.g. /Magnetic_Field_Bz/power/powerByFrequency",
        required = True,
        type=str,
        nargs="*"
    )

    args = parser.parse_args()

    plt.rcParams.update({'axes.titlesize': 26.0})
    plt.rcParams.update({'axes.labelsize': 24.0})
    plt.rcParams.update({'xtick.labelsize': 20.0})
    plt.rcParams.update({'ytick.labelsize': 20.0})
    plt.rcParams.update({'legend.fontsize': 18.0})

    if args.dataFile:
        for f in args.fields:
            power_spectrum(args.dataFile, f)
    if args.dataFolder:
        all_power_spectra(args.dataFolder, args.fields)

