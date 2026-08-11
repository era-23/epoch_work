import argparse
import glob
import os
import numpy as np
import xarray as xr
import epoch_utils
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.interpolate import make_smoothing_spline
from scipy.stats import linregress

def calculate_linear_growth_rate(
        energyTrace : np.ndarray,
        timeCoords : np.ndarray,
        thresholdIndices : dict
    ) -> tuple:

    assert len(energyTrace) == len(timeCoords)

    start_idx = thresholdIndices.get("linear_MCI_growth_start", None)
    if start_idx == None:
        return None, None, None
    else:
        end_idx = thresholdIndices.get("nonlinear_MCI_growth_start", None)
        linReg_result = linregress(timeCoords[start_idx:end_idx], energyTrace[start_idx:end_idx])
        return linReg_result, start_idx, end_idx


def find_phase_thresholds(
        energyTrace : np.ndarray, 
        timeCoords : np.ndarray,
        rollingWindowSize : int = 10,
    ) -> tuple:

    threshold_indices = {}
    threshold_indices["linear_MCI_growth_start"] = None
    threshold_indices["nonlinear_MCI_growth_start"] = None
    threshold_indices["peak_growth"] = None
    threshold_indices["MCI_stationary_point"] = None
    threshold_indices["nonlinear_restitution"] = None
    threshold_indices["nonlinear_saturation"] = None
    dE_dt = None

    # Smooth energy function
    smoothDeltaED = make_smoothing_spline(timeCoords, energyTrace, lam = 0.01)
    smoothDeltaData = smoothDeltaED(timeCoords)

    # Return rate of change of function (used to identify stationary points)
    dE_dt = np.diff(smoothDeltaData)/np.diff(timeCoords)

    # Return ln(abs) of function (used to find growth rates)
    np.seterr(divide = 'ignore') 
    ln_abs = np.log(np.abs(energyTrace))
    np.seterr(divide = 'warn') 

    # Return n-point rolling S.D. of ln(abs(function)) (used to identify inflection points)
    rolling_stdev = epoch_utils.rolling_stdev(np.diff(ln_abs), window=rollingWindowSize)

    # Find start of instability/end of noise region
    threshold = 0.1 * rolling_stdev.max()
    max_from_here_to_end = np.maximum.accumulate(rolling_stdev[::-1])[::-1]
    valid_starts = (max_from_here_to_end <= threshold) & (rolling_stdev > 0.0)
    matching_idxs = np.where(valid_starts)[0]
    mci_start_idx = matching_idxs[0] if matching_idxs.size > 0 else None
    threshold_indices["linear_MCI_growth_start"] = int(mci_start_idx)

    if threshold_indices["linear_MCI_growth_start"] is not None:
        nl_mci_start_idxs = np.nonzero(abs(smoothDeltaData) > 0.01) # Deviation from baseline energy is > 0.01%
        if len(nl_mci_start_idxs) > 0 and len(nl_mci_start_idxs[0]) > 0:
            nl_start = int(nl_mci_start_idxs[0][0]) if int(nl_mci_start_idxs[0][0]) > threshold_indices["linear_MCI_growth_start"] else threshold_indices["linear_MCI_growth_start"] + 10
            threshold_indices["nonlinear_MCI_growth_start"] = nl_start

        # Find peaks (NL energy restitution), troughs (linear saturation)
        # Trough in dE/dt indicates peak MCI growth 
        if threshold_indices["nonlinear_MCI_growth_start"] is not None:
            stationaries = np.where(np.diff(np.sign(dE_dt[threshold_indices["nonlinear_MCI_growth_start"]:])))[0]
            if len(stationaries) > 0:
                threshold_indices["MCI_stationary_point"] = threshold_indices["nonlinear_MCI_growth_start"] + int(stationaries[0])
            if len(stationaries) > 1:
                threshold_indices["nonlinear_restitution"] = threshold_indices["nonlinear_MCI_growth_start"] + int(stationaries[1])
            for i in range(1, len(smoothDeltaData)):
                if abs(smoothDeltaData[-i:].max() - smoothDeltaData[-i:].min()) > 0.2:
                    threshold_indices["nonlinear_saturation"] = len(smoothDeltaData) - i
                    break

        threshold_indices["peak_growth"] = int(np.argmin(dE_dt))

    return threshold_indices, dE_dt, ln_abs, rolling_stdev

def find_and_plot_phases(
        folder : Path, 
        outputFolder : Path = None,
        displayPlots : bool = False,
        doLog : bool = True,
        noTitle : bool = False
):
    plt.rcParams.update({'axes.titlesize': 24.0})
    plt.rcParams.update({'axes.labelsize': 24.0})
    plt.rcParams.update({'xtick.labelsize': 24.0})
    plt.rcParams.update({'ytick.labelsize': 24.0})
    plt.rcParams.update({'legend.fontsize': 16.0})

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

            # s = "/home/era536/Documents/Epoch/Data/2026_analysis/all_angles_4/debug/run_74_stats.nc"

            stats = xr.open_datatree(
                s,
                engine="netcdf4"
            )
            energyTraces = dict.fromkeys(energyFields)

            if timeCoords is None: # Assuming constant
                timeCoords = stats["/Energy"].coords["time"].data
            
            totalDeltaED = np.zeros_like(stats["/Energy/backgroundIonMeanEnergyDensity"].to_numpy())
            for field in energyFields:
                deltaED = (stats[field] - stats[field].isel({"time" : 0})).to_numpy() / float(stats["/Energy/fastIonMeanEnergyDensity"].isel({"time" : 0}).data)
                totalDeltaED += deltaED
                energyTraces[field] = 100.0 * deltaED
            energyTraces["totalMeanEnergyDensity"] = 100.0 * totalDeltaED

            # Plot
            for variable, percentageED in energyTraces.items():
                
                if "fastIon" in variable:
                    simNumber = Path(s).name.split("_")[1]
                    angle = Path(s).parent.parent.name
                    print(f"Finding phase transition thresholds and plotting 'sim {simNumber}, angle {angle}'....")

                    # Initialise plotting
                    fig, ax = plt.subplots(figsize=(16, 8))
                    filename = Path(f"run_{simNumber}_angle_{angle}_percentage_energy_change.png")
                    line_colour = next((epoch_utils.E_TRACE_SPECIES_COLOUR_MAP[c] for c in epoch_utils.E_TRACE_SPECIES_COLOUR_MAP.keys() if c in variable), False)
                    # ax.plot(timeCoords, epoch_utils.signed_ln(percentageED) if doLog else percentageED, label=f"{epoch_utils.SPECIES_NAME_MAP[variable]}", color = colour)

                    stdev_window = 5
                    threshold_indices, dE_dt, ln_abs, rolling_std = find_phase_thresholds(percentageED, timeCoords, rollingWindowSize=stdev_window)
                    l1 = ax.plot(timeCoords, ln_abs if doLog else percentageED, label=f"{epoch_utils.SPECIES_NAME_MAP[variable]} (abs)", color = line_colour)
                    l2 = ax.plot(timeCoords[1:], rolling_std, linestyle="dashed", color="black", label = f"{stdev_window}-point rolling SD(diff)")
                    ax2 = ax.twinx()
                    l3 = ax2.plot(timeCoords, percentageED, label=f"{epoch_utils.SPECIES_NAME_MAP[variable]}", alpha = 0.6, linestyle = "dashed", color = line_colour)
                    l4 = ax2.plot(timeCoords[1:], dE_dt, linestyle="dotted", color="black", label = "d(FI)/dt")
                    l5 = []
                    for k,v in threshold_indices.items():
                        if v is not None:
                            point_colour = next((epoch_utils.E_TRACE_TRANSITION_COLOUR_MAP[c] for c in epoch_utils.E_TRACE_TRANSITION_COLOUR_MAP.keys() if c == k), False)
                            # ax.scatter(timeCoords[v], epoch_utils.signed_ln(percentageED[v]) if doLog else percentageED[v], marker="x", s=100.0, label = k)
                            l5.append(ax2.scatter(timeCoords[v], percentageED[v], marker="x", s=100.0, label = k, color = point_colour))
                    # ax.plot(timeCoords[1:], epoch_utils.signed_ln(dE_dt) if doLog else dE_dt, linestyle="dotted", color="black", label = "d(FI)/dt")
                    
                    # Growth rate
                    linGrowth, start_idx, end_idx = calculate_linear_growth_rate(ln_abs, timeCoords, threshold_indices)
                    l6 = []
                    if linGrowth is not None:
                        l6 = ax.plot(
                            timeCoords[start_idx:end_idx], 
                            (linGrowth.slope * timeCoords[start_idx:end_idx]) + linGrowth.intercept, 
                            linestyle = "dashed", color = "blue", linewidth=2.5, label = r'$\gamma=$' + f"{linGrowth.slope:.4f}")

                    # Record attributes
                    stats["/Energy"].attrs["fastIonEnergyGamma"] = linGrowth.slope
                    stats["/Energy"].attrs["fastIonEnergyGammaUnit"] = "pct/Tci"

                    lines = l1 + l3 + l4 + l5 + l2 + l6
                    labels = [line.get_label() for line in lines]
                    ax.legend(lines, labels, loc = "center left", bbox_to_anchor = (1.2, 0.5))
                    ax.set_xlabel(r"Time [$\tau_{ci}$]")
                    ax.set_ylabel(f"Change in energy density [{'ln(abs(%))' if doLog else '%'}]")
                    ax2.set_ylabel(f"Change in energy density [%]")
                    # if doLog:
                    #     ax.set_yscale("symlog")
                    ax.grid()
                    if not noTitle:
                        ax.set_title(f"Run {simNumber} Angle {angle}: Change in ED relative to initial FI energy")
                    fig.tight_layout()
                    if outputFolder is not None:
                        fig.savefig(outputFolder / filename)
                    if displayPlots:
                        plt.show()
                    plt.close("all")

            stats.close()
            stats.to_netcdf("temp_file.nc", mode="w") # Write temp file
            os.replace("temp_file.nc", s)

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