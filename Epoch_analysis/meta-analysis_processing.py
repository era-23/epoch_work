import argparse
import glob
import xarray as xr
import netCDF4 as nc
import numpy as np
import epoch_utils as e_utils
from pathlib import Path
from scipy.interpolate import make_smoothing_spline
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

def collate_growth_rate_data(dataDir : Path, outputToOriginalFile : bool = True):
    
    dataFiles = glob.glob(str(dataDir / "*.nc"))
    
    for simulation in dataFiles:

        data = xr.open_datatree(
            simulation,
            engine="netcdf4"
        )

        growthRateGroups = [g for g in data.groups if g.endswith("growthRates/positive") or g.endswith("growthRates/negative")]
        fields = {}
        for groupPath in growthRateGroups:
            fieldData = data[groupPath]
            keyGrowthRates = {}
            keyGrowthRateIndices = {}
            keyGrowthRateIndices["maxFoundInSimulation"] = np.argmax(fieldData.growthRate.data) 
            keyGrowthRateIndices["bestInHighestPeakPowerK"] = np.argmax(fieldData.peakPower.data)
            keyGrowthRateIndices["bestInHighestTotalPowerK"] = np.argmax(fieldData.totalPower.data)
            for metric, i in keyGrowthRateIndices.items():
                keyGrowthRates[metric] = e_utils.LinearGrowthRate(
                    gamma = float(fieldData.growthRate[i]),
                    timeMidpoint=float(fieldData.time[i]),
                    yIntercept=float(fieldData.yIntercept[i]),
                    rSquared=float(fieldData.rSquared[i]),
                    wavenumber=float(fieldData.wavenumber[i]),
                    peakPower=float(fieldData.peakPower[i]),
                    totalPower=float(fieldData.totalPower[i])
                )
            fields[groupPath] = keyGrowthRates
        data.close()

        ncData = nc.Dataset(simulation, "a", format="NETCDF4")
        for field, stats in fields.items():
            for metric, metricData in stats.items():
                metricGroup = ncData[field].createGroup(metric)
                metricGroup.growthRate = float(metricData.gamma)
                metricGroup.time=float(metricData.timeMidpoint)
                metricGroup.yIntercept=float(metricData.yIntercept)
                metricGroup.rSquared=float(metricData.rSquared)
                metricGroup.wavenumber=float(metricData.wavenumber)
                metricGroup.peakPower=float(metricData.peakPower)
                metricGroup.totalPower=float(metricData.totalPower)
        ncData.close()

        print(f"Done {simulation}")

def calculate_energy_metadata(dataDir : Path, runNumber = None, bigLabels : bool = False, noTitle : bool = False,  debug = False):

    if bigLabels:
        plt.rcParams.update({'axes.titlesize': 26.0})
        plt.rcParams.update({'axes.labelsize': 24.0})
        plt.rcParams.update({'xtick.labelsize': 20.0})
        plt.rcParams.update({'ytick.labelsize': 20.0})
        plt.rcParams.update({'legend.fontsize': 16.0})
    else:
        plt.rcParams.update({'axes.titlesize': 18.0})
        plt.rcParams.update({'axes.labelsize': 16.0})
        plt.rcParams.update({'xtick.labelsize': 14.0})
        plt.rcParams.update({'ytick.labelsize': 14.0})
        plt.rcParams.update({'legend.fontsize': 14.0})

    dataFiles = glob.glob(str(dataDir / f"{f'run_{runNumber}_stats' if runNumber is not None else '*'}.nc"))

    for simulation in dataFiles:

        print(f"Energy meta-analysis of {simulation}...")

        xData = xr.load_dataset(
            simulation,
            engine="netcdf4",
            group="Energy"
        )

        # View groups for data
        ncData = nc.Dataset(simulation, "a", format="NETCDF4")
        energyGroup = ncData["Energy"]
        
        energyVariables = [v for v in energyGroup.variables.keys() if "energydensity" in v.lower()]

        # Get time coords
        timeCoords = xData["time"]

        # Variables for totals
        totalAbsoluteED = 0.0
        totalDeltaED = 0.0

        deltaEnergies = {}
        pctEnergies = {}

        # Sum up total E and do delta calculations
        for variable in energyVariables:

            # Get energy quantities
            absoluteED = xData[variable]  # J / m^3
            deltaED = absoluteED - absoluteED[0] # J / m^3
            
            # Record totals
            totalAbsoluteED += absoluteED
            totalDeltaED += deltaED

            deltaEnergies[variable] = deltaED

        pctConservation = float(100.0 * ((totalAbsoluteED[-1]-totalAbsoluteED[0])/totalAbsoluteED[0]))

        if debug:
            print(f"---------------------- SIMULATION: {simulation} ---------------------")
            print(f"Total energy start: {float(totalAbsoluteED[0])}")
            print(f"Total energy end: {float(totalAbsoluteED[-1])}")
            print(f"Total energy conservation: {pctConservation}%")
        
        # Write top-level energy attributes
        energyGroup.totalEnergyDensity_start = float(totalAbsoluteED[0])
        energyGroup.totalEnergyDensity_end = float(totalAbsoluteED[-1])
        energyGroup.totalEnergyDensityConservation_pct = pctConservation

        # Establish presence of fast ions for calculation of percentage energy change
        percentageBaseline = 0.0 # Errors will throw if set incorrectly
        fiField = [v for v in energyVariables if "fastion" in v.lower()]
        if fiField:
            percentageBaseline = float(xData[fiField[0]][0]) # J / m^3
        else: 
            percentageBaseline = float(totalAbsoluteED[0])

        maxExtent = np.max([np.array(v) for v in deltaEnergies.values()])
        minExtent = np.min([np.array(v) for v in deltaEnergies.values()])
        dataRange = maxExtent - minExtent
        prominence = 0.01 * dataRange # Peak prominence must be at least 1% of the data range

        maxPeakIndices = {}
        minTroughIndices = {}

        # Iterate deltas
        for variable, deltaED in deltaEnergies.items():
            
            # Smooth curve
            smoothDeltaED = make_smoothing_spline(timeCoords, deltaED, lam=1.0)
            smoothDeltaData = smoothDeltaED(timeCoords)
            
            # Find stationary points
            ed_peaks, _ = find_peaks(smoothDeltaData, distance=50, prominence=prominence)
            ed_troughs, _ = find_peaks(-smoothDeltaData, distance=50, prominence=prominence)
            ed_troughs = np.array([int(t) for t in ed_troughs if smoothDeltaData[t] < 0.0]) # Filter for only negative troughs

            # Calculate percentage change relative to baseline
            percentageED = 100.0 * (deltaED / percentageBaseline) # %
            pctEnergies[variable] = percentageED

            # Record for total
            # Don't think this is used
            #totalPercentageED += percentageED

            # Record
            smoothPctData = 100.0 * (smoothDeltaData / percentageBaseline)
            hasPeaks = bool(ed_peaks.size > 0)
            hasTroughs = bool(ed_troughs.size > 0)

            maxAtSimEnd = bool((len(smoothDeltaData)-1)-np.argmax(smoothDeltaData) < 5)
            minAtSimEnd = bool((len(smoothDeltaData)-1)-np.argmin(smoothDeltaData) < 5)
            energyGroup[variable].maxAtSimEnd = int(maxAtSimEnd)
            energyGroup[variable].minAtSimEnd = int(minAtSimEnd)     

            if debug:
                print(f"Variable: {variable}")
                print(f"Has peaks: {hasPeaks}")
                print(f"Has troughs: {hasTroughs}")
                print(f"Time series length: {len(smoothDeltaData)}, index max: {np.argmax(smoothDeltaData)}, maxAtSimEnd: {maxAtSimEnd}")
                print(f"Time series length: {len(smoothDeltaData)}, index min: {np.argmin(smoothDeltaData)}, minAtSimEnd: {minAtSimEnd}")
                colour = next((e_utils.E_TRACE_SPECIES_COLOUR_MAP[c] for c in e_utils.E_TRACE_SPECIES_COLOUR_MAP.keys() if c in variable), False)
                if colour:
                    plt.plot(timeCoords, percentageED,  label=e_utils.SPECIES_NAME_MAP[variable], alpha=0.4, color = colour)
                    plt.plot(timeCoords, smoothPctData, label=f"Smoothed {e_utils.SPECIES_NAME_MAP[variable]}", linestyle="--", color = colour)
                else:
                    plt.plot(timeCoords, percentageED,  label=e_utils.SPECIES_NAME_MAP[variable], alpha=0.4)
                    plt.plot(timeCoords, smoothPctData, label=f"Smoothed {e_utils.SPECIES_NAME_MAP[variable]}", linestyle="--")
                
            energyGroup[variable].hasPeaks = int(hasPeaks)
            energyGroup[variable].hasTroughs = int(hasTroughs)
            
            if hasPeaks:
                energyGroup[variable].peakIndices = ed_peaks
                energyGroup[variable].peakValues_delta = smoothDeltaData[ed_peaks]
                energyGroup[variable].peakValues_pct = smoothPctData[ed_peaks]
                energyGroup[variable].peakTimes = [float(t) for t in timeCoords[ed_peaks]]
                if debug:
                    print(f"Peaks: {smoothDeltaData[ed_peaks]} ({smoothPctData[ed_peaks]}%) at {ed_peaks}")
                    print(f"Time of peaks: {[float(t) for t in timeCoords[ed_peaks]]}")
                    plt.scatter(timeCoords[ed_peaks], smoothPctData[ed_peaks], marker="x", color="black")
                maxPeakIndices[variable] = ed_peaks[np.argmax([smoothPctData[p] for p in ed_peaks])]
            
            if hasTroughs:
                energyGroup[variable].troughIndices = ed_troughs
                energyGroup[variable].troughValues_delta = smoothDeltaData[ed_troughs]
                energyGroup[variable].troughValues_pct = smoothPctData[ed_troughs]
                energyGroup[variable].troughTimes = [float(t) for t in timeCoords[ed_troughs]]
                if debug:
                    print(f"Troughs: {smoothDeltaData[ed_troughs]} ({smoothPctData[ed_troughs]}%) at {ed_troughs}")
                    print(f"Time of troughs: {[float(t) for t in timeCoords[ed_troughs]]}")
                    plt.scatter(timeCoords[ed_troughs], smoothPctData[ed_troughs], marker="+", color="black")
                    print("...................................................................................")
                minTroughIndices[variable] = ed_troughs[np.argmin([smoothPctData[p] for p in ed_troughs])]
            
        if debug:
            plt.legend()
            plt.xlabel(r"Time [$\tau_{ci}$]")
            plt.ylabel("Change in energy density [%]")
            plt.grid()
            if not noTitle:
                plt.title(f"{simulation.split('/')[-1]} Percentage change in ED relative to fast ion energy")
            plt.show()
        
        # Specific to IRB transfer
        bkgdProtonField = [v for v in energyVariables if "proton" in v.lower()]
        bkgdElectronField = [v for v in energyVariables if "electron" in v.lower()]
        if fiField and bkgdProtonField and bkgdElectronField:
            fastVar = fiField[0]
            protVar = bkgdProtonField[0]
            elecVar = bkgdElectronField[0]
            fast = deltaEnergies[fastVar]
            fast_pct = pctEnergies[fastVar]
            proton = deltaEnergies[protVar]
            proton_pct = pctEnergies[protVar]
            electron = deltaEnergies[elecVar]
            electron_pct = pctEnergies[elecVar]

            hasFastIonGain = bool(fast[-1] > 0.0) or energyGroup[fastVar].hasPeaks
            hasBkgdIonGain = bool(proton[-1] > 0.0)
            hasBkgdElectronGain = bool(electron[-1] > 0.0)

            energyGroup.hasOverallFastIonGain = int(hasFastIonGain)
            energyGroup.hasOverallBkgdIonGain = int(hasBkgdIonGain)
            energyGroup.hasOverallBkgdElectronGain = int(hasBkgdElectronGain)

            if debug:
                print(f"Overall fast ion gain: {hasFastIonGain}") # bool
                print(f"Overall background proton gain: {hasBkgdIonGain}") # bool
                print(f"Overall background electron gain: {hasBkgdElectronGain}") # bool

            if fastVar in minTroughIndices.keys():

                pGainAtFiTrough = float(proton[minTroughIndices[fastVar]])
                pGainAtFiTrough_pct = float(proton_pct[minTroughIndices[fastVar]])
                eGainAtFiTrough = float(electron[minTroughIndices[fastVar]])
                eGainAtFiTrough_pct = float(electron_pct[minTroughIndices[fastVar]])

                energyGroup.bkgdIonChangeAtFastIonTrough = pGainAtFiTrough
                energyGroup.bkgdIonChangeAtFastIonTrough_pct = pGainAtFiTrough_pct
                energyGroup.bkgdElectronChangeAtFastIonTrough = eGainAtFiTrough
                energyGroup.bkgdElectronChangeAtFastIonTrough_pct = eGainAtFiTrough_pct

                if debug:
                    print(f"Fast ion trough: {minTroughIndices[fastVar]}") # numeric and already recorded above
                    print(f"Proton gain at fast ion trough: {pGainAtFiTrough} ({pGainAtFiTrough_pct}%)") # numeric
                    print(f"Electron gain at fast ion trough: {eGainAtFiTrough} ({eGainAtFiTrough_pct}%)")# numeric

            if protVar in maxPeakIndices.keys():
                fiLossAtProtonPeak = float(fast[maxPeakIndices[protVar]])
                fiLossAtProtonPeak_pct = float(fast_pct[maxPeakIndices[protVar]])
                energyGroup.fastIonChangeAtBkgdIonPeak = fiLossAtProtonPeak
                energyGroup.fastIonChangeAtBkgdIonPeak_pct = fiLossAtProtonPeak_pct

                if debug:
                    print(f"Fast ion loss at proton peak: {fiLossAtProtonPeak} ({fiLossAtProtonPeak_pct}%)") # numeric
            
            if elecVar in maxPeakIndices.keys():
                fiLossAtElectronPeak = float(fast[maxPeakIndices[elecVar]])
                fiLossAtElectronPeak_pct = float(fast_pct[maxPeakIndices[elecVar]])
                energyGroup.fastIonChangeAtBkgdElectronPeak = fiLossAtElectronPeak
                energyGroup.fastIonChangeAtBkgdElectronPeak_pct = fiLossAtElectronPeak_pct

                if debug:
                    print(f"Fast ion loss at electron peak: {fiLossAtElectronPeak} ({fiLossAtElectronPeak_pct}%)") # numeric

            if debug:
                print("------------------------------------------------------------------------------------")
            
        ncData.close()

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
        "--growthRates",
        action="store_true",
        help="Calculate and collate growth rate data",
        required = False
    )
    parser.add_argument(
        "--energy",
        action="store_true",
        help="Calculate and collate energy metadata",
        required = False
    )
    parser.add_argument(
        "--outputToOriginalFile",
        action="store_true",
        help="Write output to each original netCDF data file. Else collate in csv",
        required = False
    )
    parser.add_argument(
        "--outputFile",
        action="store",
        help="Optional CSV file to which to write output.",
        required = False,
        type=str
    )
    parser.add_argument(
        "--runNumber",
        action="store",
        help="Optional run number to only process one run.",
        required = False,
        type=str
    )
    parser.add_argument(
        "--bigLabels",
        action="store_true",
        help="Large labels on plots for posters, presentations etc.",
        required = False
    )
    parser.add_argument(
        "--noTitle",
        action="store_true",
        help="No title on plots for posters, papers etc. which will include captions instead.",
        required = False
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Display debug info/plots",
        required = False
    )

    args = parser.parse_args()

    if args.growthRates:
        collate_growth_rate_data(args.dir, args.outputToOriginalFile)
    if args.energy:
        calculate_energy_metadata(args.dir, args.runNumber, args.bigLabels, args.noTitle, args.debug)