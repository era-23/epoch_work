import argparse
import glob
import numpy as np
import xarray as xr
import epoch_utils
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.interpolate import make_smoothing_spline
from scipy.signal import find_peaks

def find_phase_thresholds(energyTrace : np.ndarray, timeCoords : np.ndarray) -> tuple:

    threshold_indices = {}
    threshold_indices["MCI_start"] = None
    threshold_indices["peak_growth"] = None
    threshold_indices["linear_saturation"] = None
    threshold_indices["nonlinear_restitution"] = None
    threshold_indices["nonlinear_saturation"] = None
    dE_dt = None

    # Smooth energy function
    smoothDeltaED = make_smoothing_spline(timeCoords, energyTrace, lam = 0.01)
    smoothDeltaData = smoothDeltaED(timeCoords)

    # Return rate of change of function
    dE_dt = np.diff(smoothDeltaData)/np.diff(timeCoords)

    # Find start of instability
    mci_start_idxs = np.nonzero(abs(smoothDeltaData) > 0.05) # Deviation by >0.05%
    if len(mci_start_idxs) > 0 and len(mci_start_idxs[0]) > 0:
        threshold_indices["MCI_start"] = int(mci_start_idxs[0][0])

    if threshold_indices["MCI_start"] is not None:
        # Find peaks (NL energy restitution), troughs (linear saturation)
        # Trough in dE/dt indicates peak MCI growth 
        stationaries = np.where(np.diff(np.sign(dE_dt[threshold_indices["MCI_start"]:])))[0]
        if len(stationaries) > 0:
            threshold_indices["linear_saturation"] = threshold_indices["MCI_start"] + int(stationaries[0])
        if len(stationaries) > 1:
            threshold_indices["nonlinear_restitution"] = threshold_indices["MCI_start"] + int(stationaries[1])

        # dataRange = dE_dt.max() - dE_dt.min()
        # prominence = 0.02 * dataRange # Peak prominence must be at least 2% of the data range
        # dEdt_troughs, _ = find_peaks(-dE_dt, distance=50, prominence=prominence)
        # dEdt_troughs = np.array([int(t) for t in dEdt_troughs if dE_dt[t] < 0.0]) # Filter for only negative troughs

        threshold_indices["peak_growth"] = int(np.argmin(dE_dt))

        # Find saturation region
        if threshold_indices["linear_saturation"] is not None: # If linear saturation was identified (NL restitution is optional)
            for i in range(1, len(smoothDeltaData)):
                if abs(smoothDeltaData[-i:].max() - smoothDeltaData[-i:].min()) > 0.2:
                    threshold_indices["nonlinear_saturation"] = len(smoothDeltaData) - i
                    break

    return threshold_indices, dE_dt

def find_and_plot_phases(
        folder : Path, 
        outputFolder : Path = None,
        maxTime : float = None,
        displayPlots : bool = False,
        doLog : bool = False,
        noTitle : bool = False
):
    plt.rcParams.update({'axes.titlesize': 32.0})
    plt.rcParams.update({'axes.labelsize': 32.0})
    plt.rcParams.update({'xtick.labelsize': 28.0})
    plt.rcParams.update({'ytick.labelsize': 28.0})
    plt.rcParams.update({'legend.fontsize': 20.0})

    angles = glob.glob(str(folder / "9*"))
    energyTraces = {}
    energyFields = [
        "/Energy/backgroundIonMeanEnergyDensity", 
        "/Energy/electronMeanEnergyDensity", 
        "/Energy/magneticFieldMeanEnergyDensity", 
        "/Energy/electricFieldMeanEnergyDensity", 
        "/Energy/fastIonMeanEnergyDensity"
    ]
    timeCoords = None

    # Get all data
    for angle in angles:
        
        # Get simulation stats files
        simFiles = glob.glob(str(Path(angle) / "data" / "*_stats.nc"))

        for s in simFiles:
            stats = xr.open_datatree(
                s,
                engine="netcdf4"
            )
            energyTraces[s] = dict.fromkeys(energyFields)

            if timeCoords is None: # Assuming constant
                timeCoords = stats["/Energy"].coords["time"].data
            
            totalDeltaED = np.zeros_like(stats["/Energy/backgroundIonMeanEnergyDensity"].to_numpy())
            for field in energyFields:
                deltaED = (stats[field] - stats[field].isel({"time" : 0})).to_numpy() / float(stats["/Energy/fastIonMeanEnergyDensity"].isel({"time" : 0}).data)
                totalDeltaED += deltaED
                energyTraces[s][field] = 100.0 * deltaED
            energyTraces[s]["totalMeanEnergyDensity"] = 100.0 * totalDeltaED

    # Plot
    for statsFile, tracesByField in energyTraces.items():
        
        simNumber = Path(statsFile).name.split("_")[1]
        angle = Path(statsFile).parent.parent.name

        print(f"Finding phase transition thresholds and plotting 'sim {simNumber}, angle {angle}'....")

        # Initialise plotting
        fig, ax = plt.subplots(figsize=(15, 8))
        filename = Path(f"run_{simNumber}_angle_{angle}_percentage_energy_change.png")
        totalDeltaED = np.zeros_like(tracesByField["/Energy/backgroundIonMeanEnergyDensity"])
        
        # Iterate deltas
        for variable, percentageED in tracesByField.items(): 
            
            colour = next((epoch_utils.E_TRACE_SPECIES_COLOUR_MAP[c] for c in epoch_utils.E_TRACE_SPECIES_COLOUR_MAP.keys() if c in variable), False)
            ax.plot(timeCoords, percentageED, label=f"{epoch_utils.SPECIES_NAME_MAP[variable]}", color = colour)
            
            if "fastIon" in variable:
                threshold_indices, dE_dt = find_phase_thresholds(percentageED, timeCoords)
                for k,v in threshold_indices.items():
                    if v is not None:
                        ax.scatter(timeCoords[v], percentageED[v], marker="x", s=100.0, label = k)
                ax.plot(timeCoords[1:], dE_dt, linestyle="dotted", color="black", label = "d(FI)/dt")
        
        ax.legend()
        ax.set_xlabel(r"Time [$\tau_{ci}$]")
        ax.set_ylabel("Change in energy density [%]")
        if doLog:
            ax.set_yscale("symlog")
        ax.grid()
        if not noTitle:
            ax.set_title(f"Run {simNumber} Angle {angle}: Change in ED relative to initial FI energy")
        fig.tight_layout()
        if outputFolder is not None:
            fig.savefig(outputFolder / filename)
        if displayPlots:
            plt.show()
        plt.close("all")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--dataFolder",
        action="store",
        help="Filepath of folder of simulation output data to plot, with angles and /data/ as subfolders e.g. .../all_angles_4/",
        required = False,
        type=Path
    )
    parser.add_argument(
        "--outputFolder",
        action="store",
        help="Filepath of folder for plot output.",
        required = False,
        type=Path
    )

    args = parser.parse_args()

    plt.rcParams.update({'axes.titlesize': 26.0})
    plt.rcParams.update({'axes.labelsize': 26.0})
    plt.rcParams.update({'xtick.labelsize': 18.0})
    plt.rcParams.update({'ytick.labelsize': 18.0})
    plt.rcParams.update({'legend.fontsize': 18.0})

    find_and_plot_phases(args.dataFolder, args.outputFolder)