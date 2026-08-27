import argparse
import os
import shutil as sh
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import epydeck
import netCDF4 as nc
import numpy as np
import xarray as xr
import xrft  # noqa: E402
from matplotlib import pyplot as plt
from scipy import constants
from scipy.interpolate import make_smoothing_spline
from scipy.signal import find_peaks
from sdf_xarray import SDFPreprocess

import epoch_utils as e_utils

warnings.simplefilter(action='ignore', category=FutureWarning)

def initialise_folder_structure(
        outputDirectory : Path,
        angles : list = ["90", "92", "94", "96", "sum"],
        fields : list = ["Electric_Field_Ex", "Electric_Field_Ey", "Magnetic_Field_Bz", "energy", "growth_rates"]
    ):
    """
    Creates folder structure:
    outputDir
        -> angle, e.g. 90, 92 etc.
            -> data
            -> plots
                -> fields, e.g. Magnetic_Field_Bz, energy 
    """

    if os.path.exists(outputDirectory):
        sh.rmtree(outputDirectory)
    os.mkdir(outputDirectory)

    for angle in angles:
        angle_folder = outputDirectory / angle
        os.mkdir(angle_folder)
        os.mkdir(angle_folder / "data")
        plots_folder = angle_folder / "plots"
        os.mkdir(plots_folder)
        for field in fields:
            os.mkdir(plots_folder / field)

def run_energy_analysis(
    dataset : xr.Dataset,
    inputDeck : dict,
    simName : str,
    savePlotsFolder : Path,
    statsFile : nc.Dataset,
    log : bool = False,
    displayPlots : bool = False,
    noTitle : bool = False,
    noLegend : bool = False,
    backgroundSpeciesName : str = "deuteron",
    fastSpeciesName : str = "alpha",
    mci_NL_threshold_pct = 0.05,
    saturation_variation_threshold_pct = 0.01,
    debug : bool = False
) -> tuple:
    print("Analyzing energy profile...")

    threshold_indices = {
        "MCI_start": None, "MCI_peak_growth": None, "MCI_linear_saturation": None,
        "MCI_nonlinear_restitution": None, "MCI_nonlinear_saturation": None
    }
    dE_dt = None

    constants_deck = inputDeck.get("constant", {})
    beam = constants_deck.get("frac_beam", 0.0) > 0.0

    time_data = dataset.coords["time"].values

    # 1. NetCDF Setup
    if "Energy" not in statsFile.groups:
        energyStats = statsFile.createGroup("Energy")
        energyStats.createDimension("time", time_data.size)
        time_var = energyStats.createVariable("time", "f8", ("time",))
        bed = energyStats.createVariable("backgroundIonMeanEnergyDensity", "f8", ("time",))
        eed = energyStats.createVariable("electronMeanEnergyDensity", "f8", ("time",))
        efd = energyStats.createVariable("electricFieldMeanEnergyDensity", "f8", ("time",))
        mfd = energyStats.createVariable("magneticFieldMeanEnergyDensity", "f8", ("time",))
    else:
        energyStats = statsFile.groups["Energy"]
        time_var = energyStats.variables["time"]
        bed = energyStats.variables["backgroundIonMeanEnergyDensity"]
        eed = energyStats.variables["electronMeanEnergyDensity"]
        efd = energyStats.variables["electricFieldMeanEnergyDensity"]
        mfd = energyStats.variables["magneticFieldMeanEnergyDensity"]
    
    time_var[:] = time_data
    energyStats.long_name = "Particle and field energy data"

    # 2. Physics & Densities
    bkgd_density = constants_deck['background_density']
    frac_beam = constants_deck['frac_beam']
    fast_ion_charge_e = constants_deck['fast_ion_charge_e']
    background_ion_charge_e = constants_deck['background_ion_charge_e']

    background_ion_density = (bkgd_density - (frac_beam * bkgd_density * fast_ion_charge_e)) / background_ion_charge_e
    electron_density = bkgd_density

    # Extract spatial means directly as NumPy arrays (No redundant .load())
    backgroundIonKEdensity_mean = dataset[f'Derived_Average_Particle_Energy_{backgroundSpeciesName}'].mean(dim="x_space").values * background_ion_density
    bed[:] = backgroundIonKEdensity_mean

    electronKEdensity_mean = dataset['Derived_Average_Particle_Energy_electron'].mean(dim="x_space").values * electron_density
    eed[:] = electronKEdensity_mean

    E_sq = dataset['Electric_Field_Ex']**2 + dataset['Electric_Field_Ey']**2 + dataset['Electric_Field_Ez']**2
    electricFieldDensity_mean = ((constants.epsilon_0 * E_sq) / 2.0).mean(dim="x_space").values
    efd[:] = electricFieldDensity_mean

    B_sq = dataset['Magnetic_Field_Bx']**2 + dataset['Magnetic_Field_By']**2 + dataset['Magnetic_Field_Bz']**2
    magneticFieldEnergyDensity_mean = (B_sq / (2.0 * constants.mu_0)).mean(dim="x_space").values
    mfd[:] = magneticFieldEnergyDensity_mean

    # Energy Deltas
    deltaMeanMagneticEnergyDensity = magneticFieldEnergyDensity_mean - magneticFieldEnergyDensity_mean[0]
    deltaMeanElectricEnergyDensity = electricFieldDensity_mean - electricFieldDensity_mean[0]
    deltaBackgroundIonKE_density = backgroundIonKEdensity_mean - backgroundIonKEdensity_mean[0]
    deltaElectronKE_density = electronKEdensity_mean - electronKEdensity_mean[0]

    totalAbsoluteMeanEnergyDensity = backgroundIonKEdensity_mean + electronKEdensity_mean + magneticFieldEnergyDensity_mean + electricFieldDensity_mean
    totalDeltaMeanEnergyDensity = deltaBackgroundIonKE_density + deltaElectronKE_density + deltaMeanMagneticEnergyDensity + deltaMeanElectricEnergyDensity
    timeCoords = time_data

    e_utils.write_stats(energyStats, "backgroundIonEnergyDensity", backgroundIonKEdensity_mean, deltaBackgroundIonKE_density, timeCoords)
    e_utils.write_stats(energyStats, "electronEnergyDensity", electronKEdensity_mean, deltaElectronKE_density, timeCoords)
    e_utils.write_stats(energyStats, "electricFieldEnergyDensity", electricFieldDensity_mean, deltaMeanElectricEnergyDensity, timeCoords)
    e_utils.write_stats(energyStats, "magneticFieldEnergyDensity", magneticFieldEnergyDensity_mean, deltaMeanMagneticEnergyDensity, timeCoords)

    if beam:
        fastIonDensity = bkgd_density * frac_beam
        fastIonKEdensity_mean = dataset[f'Derived_Average_Particle_Energy_{fastSpeciesName}'].mean(dim="x_space").values * fastIonDensity
        
        fed = energyStats.createVariable("fastIonMeanEnergyDensity", "f8", ("time",)) if "fastIonMeanEnergyDensity" not in energyStats.variables else energyStats.variables["fastIonMeanEnergyDensity"]
        fed[:] = fastIonKEdensity_mean
        
        deltaFastIonKE_density = fastIonKEdensity_mean - fastIonKEdensity_mean[0]
        totalAbsoluteMeanEnergyDensity += fastIonKEdensity_mean
        totalDeltaMeanEnergyDensity += deltaFastIonKE_density

        e_utils.write_stats(energyStats, "fastIonEnergyDensity", fastIonKEdensity_mean, deltaFastIonKE_density, timeCoords)

    # Global Conservation
    energyStats.totalEnergyDensity_start = float(totalAbsoluteMeanEnergyDensity[0])
    energyStats.totalEnergyDensity_end = float(totalAbsoluteMeanEnergyDensity[-1])
    pctConservation = float(100.0 * ((totalAbsoluteMeanEnergyDensity[-1] - totalAbsoluteMeanEnergyDensity[0]) / totalAbsoluteMeanEnergyDensity[0]))
    energyStats.totalEnergyDensityConservation_pct = pctConservation

    deltaEnergies = {
        "backgroundIonMeanEnergyDensity": deltaBackgroundIonKE_density,
        "electronMeanEnergyDensity": deltaElectronKE_density,
        "magneticFieldMeanEnergyDensity": deltaMeanMagneticEnergyDensity,
        "electricFieldMeanEnergyDensity": deltaMeanElectricEnergyDensity
    }
    percentageBaseline = float(fastIonKEdensity_mean[0]) if beam else float(totalAbsoluteMeanEnergyDensity[0])
    if beam:
        deltaEnergies["fastIonMeanEnergyDensity"] = deltaFastIonKE_density

    maxPeakIndices, minTroughIndices, pctEnergies = {}, {}, {}
    
    # Calculate Data Prominence across all series
    all_deltas = np.array(list(deltaEnergies.values()))
    prominence = 0.02 * (np.max(all_deltas) - np.min(all_deltas))

    fig, ax = plt.subplots(figsize=(12, 8))
    timeCoords = np.nan_to_num(timeCoords)
    totalED = np.zeros_like(timeCoords)

    for variable, deltaED in deltaEnergies.items():
        deltaED = np.nan_to_num(deltaED)
        totalED += deltaED
        
        smoothDeltaED = make_smoothing_spline(timeCoords, deltaED, lam=0.01)
        smoothDeltaData = smoothDeltaED(timeCoords)
        
        ed_peaks, _ = find_peaks(smoothDeltaData, distance=50, prominence=prominence)
        ed_troughs, _ = find_peaks(-smoothDeltaData, distance=50, prominence=prominence)
        ed_troughs = np.array([int(t) for t in ed_troughs if smoothDeltaData[t] < 0.0])

        percentageED = 100.0 * (deltaED / percentageBaseline)
        pctEnergies[variable] = percentageED
        smoothPctData = 100.0 * (smoothDeltaData / percentageBaseline)

        hasPeaks, hasTroughs = ed_peaks.size > 0, ed_troughs.size > 0
        energyStats[variable].maxAtSimEnd = int((len(smoothDeltaData) - 1) - np.argmax(smoothDeltaData) < 5)
        energyStats[variable].minAtSimEnd = int((len(smoothDeltaData) - 1) - np.argmin(smoothDeltaData) < 5)

        if "fastIon" in variable:
            dE_dt = np.diff(smoothPctData) / np.diff(timeCoords)

            mci_start_idxs = np.nonzero(smoothPctData < -mci_NL_threshold_pct)[0]
            if mci_start_idxs.size > 0:
                threshold_indices["MCI_start"] = int(mci_start_idxs[0])
            
            if threshold_indices["MCI_start"] is not None:
                mci_start = threshold_indices["MCI_start"]
                stationaries = np.where(np.diff(np.sign(dE_dt[mci_start:])))[0]
                if stationaries.size > 0:
                    threshold_indices["MCI_linear_saturation"] = mci_start + int(stationaries[0])
                if stationaries.size > 1:
                    threshold_indices["MCI_nonlinear_restitution"] = mci_start + int(stationaries[1])

                threshold_indices["MCI_peak_growth"] = int(np.argmin(dE_dt))

                # Saturation detection
                if threshold_indices["MCI_linear_saturation"] is not None:
                    rev_data = smoothPctData[::-1]
                    running_max = np.maximum.accumulate(rev_data)
                    running_min = np.minimum.accumulate(rev_data)
                    variation = running_max - running_min
                    
                    sat_mask = np.where(variation > saturation_variation_threshold_pct)[0]
                    if sat_mask.size > 0:
                        threshold_indices["MCI_nonlinear_saturation"] = len(smoothPctData) - int(sat_mask[0])

            # Write NetCDF Attributes
            fast_ion_var = energyStats["fastIonMeanEnergyDensity"]
            for threshold, idx in threshold_indices.items():
                fast_ion_var.setncattr(f"{threshold}_idx", idx if idx is not None else "None")
                fast_ion_var.setncattr(f"{threshold}_time", timeCoords[idx] if idx is not None else "None")
                if idx is not None:
                    ax.scatter(timeCoords[idx], smoothPctData[idx], marker="x", s=100.0, label=threshold)

        # Plot formatting
        colour = next((e_utils.E_TRACE_SPECIES_COLOUR_MAP[c] for c in e_utils.E_TRACE_SPECIES_COLOUR_MAP if c in variable), None)
        lbl = e_utils.SPECIES_NAME_MAP.get(variable, variable)
        ax.plot(timeCoords, percentageED, alpha=0.5, color=colour)
        ax.plot(timeCoords, smoothPctData, label=lbl, linestyle="--", color=colour)

        energyStats[variable].hasPeaks = int(hasPeaks)
        energyStats[variable].hasTroughs = int(hasTroughs)

        if hasPeaks:
            energyStats[variable].peakIndices = ed_peaks
            energyStats[variable].peakValues_delta = smoothDeltaData[ed_peaks]
            energyStats[variable].peakValues_pct = smoothPctData[ed_peaks]
            energyStats[variable].peakTimes = timeCoords[ed_peaks].tolist()
            ax.scatter(timeCoords[ed_peaks], smoothPctData[ed_peaks], marker="x", color="black")
            
            # Vectorized Max Peak Search (Replaces List Comprehension)
            maxPeakIndices[variable] = ed_peaks[np.argmax(smoothPctData[ed_peaks])]

        if hasTroughs:
            energyStats[variable].troughIndices = ed_troughs
            energyStats[variable].troughValues_delta = smoothDeltaData[ed_troughs]
            energyStats[variable].troughValues_pct = smoothPctData[ed_troughs]
            energyStats[variable].troughTimes = timeCoords[ed_troughs].tolist()
            ax.scatter(timeCoords[ed_troughs], smoothPctData[ed_troughs], marker="+", color="black")
            
            # Vectorized Min Trough Search (Replaces List Comprehension)
            minTroughIndices[variable] = ed_troughs[np.argmin(smoothPctData[ed_troughs])]

    # Plot total & regions
    totalPercentage = 100.0 * (totalED / percentageBaseline)
    ax.plot(timeCoords, totalPercentage, label="Total", color="black")

    if threshold_indices["MCI_start"] is not None:
        end_p = timeCoords[-1] if threshold_indices["MCI_linear_saturation"] is None else timeCoords[threshold_indices["MCI_linear_saturation"]]
        ax.axvspan(timeCoords[threshold_indices["MCI_start"]], end_p, color="green", alpha=0.1, label="linear MCI growth")
    if threshold_indices["MCI_linear_saturation"] is not None and threshold_indices["MCI_nonlinear_restitution"] is not None:
        ax.axvspan(timeCoords[threshold_indices["MCI_linear_saturation"]], timeCoords[threshold_indices["MCI_nonlinear_restitution"]], color="blue", alpha=0.1, label="alpha re-energisation")
    if threshold_indices["MCI_nonlinear_saturation"] is not None:
        ax.axvspan(timeCoords[threshold_indices["MCI_nonlinear_saturation"]], timeCoords[-1], color="red", alpha=0.1, label="NL saturation")

    if not noLegend: 
        ax.legend()
    ax.set_xlabel(r"Time [$\tau_{ci}$]")
    ax.set_ylabel("Change in energy density [%]")
    if log: 
        ax.set_yscale("symlog")
    ax.grid()
    if not noTitle:
        ax.set_title(f"{simName}: Percentage change in ED relative to " + ("fast ion energy" if beam else "total starting energy"))
    fig.tight_layout()
    fig.savefig(savePlotsFolder / f"{simName}_percentage_energy_change.png")
    if displayPlots: 
        plt.show()
    plt.close("all")

    # Inter-Regime Energy Transfer Calculations
    if beam:
        fastVar, backIonVar, elecVar = "fastIonMeanEnergyDensity", "backgroundIonMeanEnergyDensity", "electronMeanEnergyDensity"
        energyStats.hasOverallFastIonGain = int(bool(deltaEnergies[fastVar][-1] > 0.0) or energyStats[fastVar].hasPeaks)
        energyStats.hasOverallBkgdIonGain = int(bool(deltaEnergies[backIonVar][-1] > 0.0))
        energyStats.hasOverallBkgdElectronGain = int(bool(deltaEnergies[elecVar][-1] > 0.0))

        if fastVar in minTroughIndices:
            tr_idx = minTroughIndices[fastVar]
            energyStats.bkgdIonChangeAtFastIonTrough = float(deltaEnergies[backIonVar][tr_idx])
            energyStats.bkgdIonChangeAtFastIonTrough_pct = float(pctEnergies[backIonVar][tr_idx])
            energyStats.bkgdElectronChangeAtFastIonTrough = float(deltaEnergies[elecVar][tr_idx])
            energyStats.bkgdElectronChangeAtFastIonTrough_pct = float(pctEnergies[elecVar][tr_idx])

        if backIonVar in maxPeakIndices:
            pk_idx = maxPeakIndices[backIonVar]
            energyStats.fastIonChangeAtBkgdIonPeak = float(deltaEnergies[fastVar][pk_idx])
            energyStats.fastIonChangeAtBkgdIonPeak_pct = float(pctEnergies[fastVar][pk_idx])

        if elecVar in maxPeakIndices:
            el_idx = maxPeakIndices[elecVar]
            energyStats.fastIonChangeAtBkgdElectronPeak = float(deltaEnergies[fastVar][el_idx])
            energyStats.fastIonChangeAtBkgdElectronPeak_pct = float(pctEnergies[fastVar][el_idx])

    # Absolute Energy Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    if beam: 
        ax.plot(timeCoords, deltaFastIonKE_density, label="Fast ion", color="red")
    ax.plot(timeCoords, deltaBackgroundIonKE_density, label="Bkgd ion", color="orange")
    ax.plot(timeCoords, deltaElectronKE_density, label="Bkgd electron", color="blue")
    ax.plot(timeCoords, deltaMeanMagneticEnergyDensity, label="B-field", color="purple", linestyle="--")
    ax.plot(timeCoords, deltaMeanElectricEnergyDensity, label="E-field", color="green", linestyle="--")
    ax.plot(timeCoords, totalDeltaMeanEnergyDensity, label="Total", color="black")
    ax.set_xlabel(r'Time [$\tau_{ci}$]')
    ax.set_ylabel(r"Change in energy density [$J/m^3$]")
    if log: 
        ax.set_yscale("symlog")
    if not noTitle: 
        ax.set_title(f"{simName}: Absolute energy in particles and EM fields", wrap=True)
    if not noLegend: 
        ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(savePlotsFolder / f"{simName}_absolute_energy_change.png")
    if displayPlots: 
        plt.show()
    plt.close("all")

    return threshold_indices, dE_dt

def run_spectral_analysis(
        field_da : xr.DataArray, 
        fieldStats : nc.Dataset, 
        field_name : str,
        field_unit : str,
        maxK : float,
        maxW : float,
        plotFieldFolder : Path,
        sim_name : str,
        inputDeck : dict,
        backgroundIonSpecies : str,
        fastIonSpecies : str,
        displayPlots : bool,
        bispectra : bool,
        growthRates : bool,
        gammaWindowTciMin : float, 
        gammaWindowTciMax : float, 
        saveGrowthRatePlots : bool, 
        numGrowthRatesToPlot : int,
        noTitle : bool,
        timeStart : float = None,
        timeEnd : float = None
    ):

    print(f"Running spectral analysis for {sim_name}...")

    sim_name = f"{sim_name}_time{(timeStart if timeStart is not None else 0.0):.3f}:{f'{timeEnd:.3f}' if timeEnd is not None else 'end'}"
    data = field_da.sel(time=slice(timeStart, timeEnd))
    original_spec = xrft.fft(data, true_amplitude=True, true_phase=True, window=None)
    original_spec = original_spec.rename(freq_time="frequency", freq_x_space="wavenumber")
    original_spec = original_spec.where(original_spec.wavenumber != 0.0, None)

    tk_spec = e_utils.create_t_k_spectrum(sim_name, original_spec, fieldStats, maxK, load=True, debug=debug)
    wavenumberToFrequencyTable = e_utils.create_omega_k_plots(
        original_spec, fieldStats, field_name, field_unit, plotFieldFolder, 
        f"{sim_name}", inputDeck, backgroundIonSpecies, fastIonSpecies, 
        maxK=maxK, maxW=maxW, display=displayPlots, debug=debug
    )
    e_utils.create_power_spectra(data, fieldStats, sim_name, debug)
    e_utils.create_t_k_plot(tk_spec, field_name, field_unit, plotFieldFolder, sim_name, maxK, displayPlots)

    if bispectra:
        e_utils.bispectral_analysis(tk_spec, sim_name, field_name, displayPlots, plotFieldFolder, maxK=maxK)

    if growthRates:
        e_utils.process_growth_rates(
            tk_spec, fieldStats, plotFieldFolder, sim_name, field_name, 
            gammaWindowTciMin, gammaWindowTciMax, saveGrowthRatePlots, 
            numGrowthRatesToPlot, wavenumberToFrequencyTable, displayPlots, noTitle, debug
        )

def process_single_angle(
    angle_name: str,
    sim_folder: Path,
    sim_name: str,
    analysisOutputFolder: Path,
    fastIonSpecies: str,
    backgroundIonSpecies: str,
    fields: list,
    maxK: float,
    maxW: float,
    growthRates: bool,
    bispectra: bool,
    gammaWindowTciMin: float,
    gammaWindowTciMax: float,
    displayPlots: bool,
    saveGrowthRatePlots: bool,
    numGrowthRatesToPlot: int,
    mci_thresholds: dict,
    noTitle: bool,
    noLegend: bool,
    debug: bool
):
    # Read Dataset
    ds = xr.open_mfdataset(
        str(sim_folder / "*.sdf"),
        data_vars='minimal', 
        coords='minimal', 
        compat='override', 
        preprocess=SDFPreprocess()
    )
    # Filter out initial condition in 1 step
    ds = ds.isel(time=slice(1, None))

    # Read input deck
    with open(str(sim_folder / "input.deck")) as id_file:
        inputDeck = epydeck.loads(id_file.read())

    singleSimStats_filepath = analysisOutputFolder / angle_name / "data" / f"{sim_name}_{angle_name}_stats.nc"
    singleSimStats_filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with nc.Dataset(singleSimStats_filepath, "a", format="NETCDF4") as statsRoot:
        ion_gyroperiod, alfven_velocity = e_utils.calculate_simulation_metadata(
            inputDeck, ds, statsRoot, fastIonSpecies, backgroundIonSpecies
        )
        ds = e_utils.normalise_data(ds, f"{sim_name}_angle_{angle_name}", ion_gyroperiod, alfven_velocity)

        plotsFolder = analysisOutputFolder / angle_name / "plots"
        energyPlotFolder = plotsFolder / "energy"
        energyPlotFolder.mkdir(parents=True, exist_ok=True)

        # Energy analysis
        run_energy_analysis(
            ds, 
            inputDeck, 
            f"{sim_name}_angle_{angle_name}", 
            energyPlotFolder, 
            statsRoot, 
            displayPlots=displayPlots, 
            noTitle=noTitle, 
            noLegend=noLegend, 
            backgroundSpeciesName="deuteron" if backgroundIonSpecies == "D+" else "proton",
            fastSpeciesName="alpha" if fastIonSpecies == 'He-4 2+' else "ion_ring_beam",
            mci_NL_threshold_pct=mci_thresholds["mci_threshold_pct"],
            saturation_variation_threshold_pct=mci_thresholds["saturation_variation_threshold_pct"],
            debug=debug
        )

        if "all" in fields:
            em_fields = [str(f) for f in ds.data_vars.keys() if str(f).startswith(("Electric_Field", "Magnetic_Field"))]
        else:
            em_fields = fields

        time_coords = ds.coords['time'].values
        x_coords = ds.coords['x_space'].values
        dx = float(x_coords[2] - x_coords[1])
        dy = float(time_coords[2] - time_coords[1])

        # Spectral analysis on each field
        ds_loaded = ds.load()
        for field in em_fields:
            plotFieldFolder = plotsFolder / field
            plotFieldFolder.mkdir(parents=True, exist_ok=True)

            fieldStats = statsRoot.createGroup(field)
            field_da = ds_loaded[field]
            field_arr = field_da.values
            
            field_unit = field_da.units
            fieldStats.baseUnit = field_unit
            
            field_mag = float(np.sum(np.abs(field_arr)))
            fieldStats.totalMagnitude = field_mag
            
            parseval_field = float(np.sum(field_arr**2)) * dx * dy
            fieldStats.parsevalField = parseval_field
            
            field_mean = float(np.mean(field_arr))
            fieldStats.meanMagnitude = field_mean
            
            delta = np.abs(field_arr - field_mean)
            fieldStats.totalDelta = float(np.sum(delta))
            
            parseval_fieldDelta = float(np.sum(delta**2)) * dx * dy
            fieldStats.parsevalFieldDelta = parseval_fieldDelta
            fieldStats.meanDelta = float(np.mean(delta))
            
            del delta  # Free memory

            # FFT Execution
            run_spectral_analysis(field_da, fieldStats, field, field_unit, maxK, maxW,
                plotFieldFolder, f"{sim_name}_angle_{angle_name}", inputDeck, backgroundIonSpecies, fastIonSpecies,
                displayPlots, bispectra, growthRates, gammaWindowTciMin, gammaWindowTciMax,
                saveGrowthRatePlots, numGrowthRatesToPlot, noTitle)

    return angle_name, ds_loaded

def process_simulation_batch(
    simulationDataFolder : Path,
    analysisOutputFolder : Path,
    runNumber : int = None,
    fields : list = [],
    maxK : float = 100.0,
    maxW : float = None,
    growthRates : bool = False,
    bispectra : bool = False,
    gammaWindowTciMin : float = 0.5,
    gammaWindowTciMax : float = None,
    fastIonSpecies : str = 'He-4 2+',
    backgroundIonSpecies : str = 'D+',
    bigLabels : bool = False,
    noTitle : bool = False,
    noLegend : bool = False,
    displayPlots = False,
    saveGrowthRatePlots = False,
    numGrowthRatesToPlot : int = 0,
    mci_thresholds : dict = None,
    num_workers : int = None,
    debug : bool = False
):
    # MCI thresholds default
    if mci_thresholds is None:
        mci_thresholds = {"mci_threshold_pct": 0.05, "saturation_variation_threshold_pct": 0.01}

    # Dynamically switch backend based on displayPlots flag
    if not displayPlots:
        plt.switch_backend("Agg")  # Non-interactive / headless

    # Resolve angle folders
    angle_folders = list(simulationDataFolder.glob("9*"))
    if not angle_folders:
        raise ValueError("Can't identify angle folders")

    # Determine CPU cores automatically if num_workers is not explicitly provided
    if num_workers is None:
        try:
            # Respect CPU quotas/limits
            num_workers = len(os.sched_getaffinity(0))
        except AttributeError:
            # Fallback for OSs without sched_getaffinity
            num_workers = os.cpu_count() or 1

    print(f"Using {min(num_workers, len(angle_folders))} processing elements...")

    run_folders_dict = {}
    for angle_folder in angle_folders:
        angle_val = angle_folder.name
        if runNumber is None:
            sim_folders = list((angle_folder / "run_0_1000000/run_0_10000/run_0_100").glob("run_*"))
        else:
            sim_folders = list((angle_folder / "run_0_1000000/run_0_10000/run_0_100").glob(f"run_{runNumber}"))
        for sf in sim_folders:
            sim_name = sf.name
            if sim_name not in run_folders_dict:
                run_folders_dict[sim_name] = {}
            run_folders_dict[sim_name][angle_val] = sf

    fontsize_config = {
        'axes.titlesize': 26.0 if bigLabels else 18.0,
        'axes.labelsize': 24.0 if bigLabels else 16.0,
        'xtick.labelsize': 20.0 if bigLabels else 14.0,
        'ytick.labelsize': 20.0 if bigLabels else 14.0,
        'legend.fontsize': 18.0 if bigLabels else 14.0
    }
    plt.rcParams.update(fontsize_config)

    # Phase names
    preMci_name = "pre_MCI"
    linearGrowth_name = "linear_MCI"
    nonlinear_name = "nonlinear_MCI"
    nlSaturation_name = "saturated_MCI"

    if debug:
        times = []
    for sim_name, angles in run_folders_dict.items():

        if debug:
            start = time.perf_counter_ns()
        print(f"Analyzing simulation '{sim_name}' across {len(angles)} angle(s)...")

        angle_datasets = {}

        # Process individual angles in parallel
        with ProcessPoolExecutor(max_workers=min(num_workers, len(angles))) as executor:
            futures = [
                executor.submit(
                    process_single_angle,
                    angle_name, sim_folder, sim_name, analysisOutputFolder,
                    fastIonSpecies, backgroundIonSpecies, fields, maxK, maxW,
                    growthRates, bispectra, gammaWindowTciMin, gammaWindowTciMax,
                    displayPlots, saveGrowthRatePlots, numGrowthRatesToPlot,
                    mci_thresholds, noTitle, noLegend, debug
                )
                for angle_name, sim_folder in angles.items()
            ]
            
            for future in futures:
                angle_name, ds_loaded = future.result()
                angle_datasets[angle_name] = ds_loaded

        # Process combined angles
        combinedStats_filepath = analysisOutputFolder / "sum" / "data" / f"{sim_name}_combined_stats.nc"
        combinedStats_filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with nc.Dataset(combinedStats_filepath, "a", format="NETCDF4") as statsRoot:
            sample_angle = next(iter(angle_datasets.values()))
            sample_folder = next(iter(angles.values()))
            
            with open(str(sample_folder / "input.deck")) as id_file:
                inputDeck = epydeck.loads(id_file.read())

            _, _ = e_utils.calculate_simulation_metadata(
                inputDeck, sample_angle, statsRoot, spaceCoordinateName="x_space", 
                fastSpecies=fastIonSpecies, bkgdSpecies=backgroundIonSpecies
            )
            
            ds_combined = e_utils.combine_angles(sim_name, angle_datasets, statsRoot)

            plotsFolder = analysisOutputFolder / "sum" / "plots"
            energyPlotFolder = plotsFolder / "energy"
            energyPlotFolder.mkdir(parents=True, exist_ok=True)

            transition_indices, dE_dt = run_energy_analysis(
                ds_combined, inputDeck, f"{sim_name}_combined", energyPlotFolder, 
                statsRoot, displayPlots=displayPlots, noTitle=noTitle, noLegend=noLegend, 
                backgroundSpeciesName="deuteron" if backgroundIonSpecies == "D+" else "proton",
                fastSpeciesName="alpha" if fastIonSpecies == 'He-4 2+' else "nbi",
                mci_NL_threshold_pct=mci_thresholds["mci_threshold_pct"],
                saturation_variation_threshold_pct=mci_thresholds["saturation_variation_threshold_pct"]
            )

            time_coords = ds_combined.coords['time'].values
            x_coords = ds_combined.coords['x_space'].values
            dx = float(x_coords[2] - x_coords[1])
            dy = float(time_coords[2] - time_coords[1])

            if "all" in fields:
                target_fields = [str(f) for f in ds_combined.data_vars.keys() if str(f).startswith(("Electric_Field", "Magnetic_Field"))]
            else:
                target_fields = fields

            for field in target_fields:

                ##### Field analysis over all time
                plotFieldFolder = plotsFolder / field
                plotFieldFolder.mkdir(parents=True, exist_ok=True)

                fieldStats = statsRoot.createGroup(field)
                field_da = ds_combined[field]
                field_arr = field_da.values
                
                field_unit = field_da.units
                fieldStats.baseUnit = field_unit
                fieldStats.totalMagnitude = float(np.sum(np.abs(field_arr)))
                fieldStats.parsevalField = float(np.sum(field_arr**2)) * dx * dy
                
                field_mean = float(np.mean(field_arr))
                fieldStats.meanMagnitude = field_mean
                
                delta = np.abs(field_arr - field_mean)
                fieldStats.totalDelta = float(np.sum(delta))
                fieldStats.parsevalFieldDelta = float(np.sum(delta**2)) * dx * dy
                fieldStats.meanDelta = float(np.mean(delta))
                del delta

                # FFT Execution
                run_spectral_analysis(field_da, fieldStats, field, field_unit, maxK, maxW,
                    plotFieldFolder, f"{sim_name}_combined", inputDeck, backgroundIonSpecies, fastIonSpecies,
                    displayPlots, bispectra, growthRates, gammaWindowTciMin, gammaWindowTciMax,
                    saveGrowthRatePlots, numGrowthRatesToPlot, noTitle)
                
                # Need to run this multiple times according to the thresholds identified in the energy analysis (transition_indices)
                # Transition indices are: "MCI_start", "MCI_peak_growth", "MCI_linear_saturation", "MCI_nonlinear_restitution", "MCI_nonlinear_saturation"
                # ### If MCI start:
                # Process "Pre-MCI phase" up to MCI start
                # ### If MCI_start and MCI_linear_saturation:
                # Process "linear MCI growth phase" between MCI_start and MCI_linear_saturation
                # ### If "MCI_linear_saturation" and "MCI_nonlinear_saturation":
                # Process "nonlinear phase" between linear and nonlinear saturation
                # ### If "MCI_nonlinear_saturation":
                # ### Process "nonlinear saturated phase" from nonlinear saturation to end

                # Record idx/times
                for phaseName, phase_idx in transition_indices:
                    setattr(fieldStats, f"{phaseName}_idx", phase_idx)
                    setattr(fieldStats, f"{phaseName}_time", time_coords[phase_idx])

                # Process
                if transition_indices["MCI_start"] is not None:
                    # Process pre-MCI growth phase
                    preMciStats = fieldStats.createGroup(preMci_name)
                    preMciPlotFolder = plotFieldFolder / preMci_name
                    preMciPlotFolder.mkdir(parents=True, exist_ok=True)

                    timeEnd = time_coords[transition_indices["MCI_start"]]

                    run_spectral_analysis(field_da, preMciStats, field, field_unit, maxK, maxW,
                        preMciPlotFolder, f"{sim_name}_combined" + "_preMCI", inputDeck, backgroundIonSpecies, fastIonSpecies,
                        displayPlots, bispectra, growthRates, gammaWindowTciMin, gammaWindowTciMax,
                        saveGrowthRatePlots, numGrowthRatesToPlot, noTitle, timeEnd=timeEnd)

                    if transition_indices["MCI_linear_saturation"] is not None:
                        # Process linear MCI growth phase
                        linearMciStats = fieldStats.createGroup(linearGrowth_name)

                        linearMciPlotFolder = plotFieldFolder / linearGrowth_name
                        linearMciPlotFolder.mkdir(parents=True, exist_ok=True)

                        timeStart = time_coords[transition_indices["MCI_start"]]
                        timeEnd = time_coords[transition_indices["MCI_linear_saturation"]]

                        run_spectral_analysis(field_da, linearMciStats, field, field_unit, maxK, maxW,
                            linearMciPlotFolder, f"{sim_name}_combined" + "_linearMCI", inputDeck, backgroundIonSpecies, fastIonSpecies,
                            displayPlots, bispectra, growthRates, gammaWindowTciMin, gammaWindowTciMax,
                            saveGrowthRatePlots, numGrowthRatesToPlot, noTitle, timeStart=timeStart, timeEnd=timeEnd)

                        if transition_indices["MCI_nonlinear_saturation"] is not None:
                            # Process NL phase
                            nonlinearMciStats = fieldStats.createGroup(nonlinear_name)
                            nonlinearMciPlotFolder = plotFieldFolder / nonlinear_name
                            nonlinearMciPlotFolder.mkdir(parents=True, exist_ok=True)

                            timeStart = time_coords[transition_indices["MCI_linear_saturation"]]
                            timeEnd = time_coords[transition_indices["MCI_nonlinear_saturation"]]

                            run_spectral_analysis(field_da, nonlinearMciStats, field, field_unit, maxK, maxW,
                                nonlinearMciPlotFolder, f"{sim_name}_combined" + "_linearMCI", inputDeck, backgroundIonSpecies, fastIonSpecies,
                                displayPlots, bispectra, growthRates, gammaWindowTciMin, gammaWindowTciMax,
                                saveGrowthRatePlots, numGrowthRatesToPlot, noTitle, timeStart=timeStart, timeEnd=timeEnd)
                            
                            # Process NL saturation
                            saturatedMciStats = fieldStats.createGroup(nlSaturation_name)
                            saturatedMciPlotFolder = plotFieldFolder / nlSaturation_name
                            saturatedMciPlotFolder.mkdir(parents=True, exist_ok=True)

                            timeStart = time_coords[transition_indices["MCI_nonlinear_saturation"]]

                            run_spectral_analysis(field_da, saturatedMciStats, field, field_unit, maxK, maxW,
                                saturatedMciPlotFolder, f"{sim_name}_combined" + "_linearMCI", inputDeck, backgroundIonSpecies, fastIonSpecies,
                                displayPlots, bispectra, growthRates, gammaWindowTciMin, gammaWindowTciMax,
                                saveGrowthRatePlots, numGrowthRatesToPlot, noTitle, timeStart=timeStart)

        # Explicit cleanup of Xarray objects
        for orig_ds in angle_datasets.values():
            orig_ds.close()
        ds_combined.close()
        print(f"Completed processing batch for '{sim_name}'.")

        if debug:
            end = time.perf_counter_ns()
            times.append(end - start)

    if debug:
        print(f"Average simulation analysis time (s): {np.mean(times) / 1e9}.")
          
if __name__ == "__main__":

    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--dir",
        action="store",
        help="Directory containing either one simulation run, or multiple simulation directories for evaluation.",
        required = True,
        type=Path
    )
    parser.add_argument(
        "--fields",
        action="store",
        help="EPOCH fields to use for analysis.",
        required = False,
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--outputDir",
        action="store",
        help="Directory to write output stats and plots.",
        required = False,
        type=Path
    )
    parser.add_argument(
        "--initFolders",
        action="store_true",
        help="Initialise folders for concurrent processing.",
        required = False
    )
    parser.add_argument(
        "--growthRates",
        action="store_true",
        help="Calculate growth rates.",
        required = False
    )
    parser.add_argument(
        "--bispectra",
        action="store_true",
        help="Plot bispectra and bicoherence across the entire simulation.",
        required = False
    )
    parser.add_argument(
        "--maxK",
        action="store",
        help="Max wavenumber for analysis.",
        required = False,
        type=float
    )
    parser.add_argument(
        "--maxW",
        action="store",
        help="Max wavenumber for analysis.",
        required = False,
        type=float
    )
    parser.add_argument(
        "--numGrowthRatesToPlot",
        action="store",
        help="Number of wavenumber max growth rates to plot.",
        required = False,
        type=int
    )
    parser.add_argument(
        "--minGammaFitWindow",
        action="store",
        help="Minimum gamma fit window, in percentage of the total trace.",
        required = False,
        type=float
    )
    parser.add_argument(
        "--maxGammaFitWindow",
        action="store",
        help="Maximum gamma fit window, in percentage of the total trace.",
        required = False,
        type=float
    )
    parser.add_argument(
        "--mciStartThresholdPct",
        action="store",
        help="Percentage of energy variation from baseline required to trigger recording of the MCI.",
        required = False,
        type=float
    )
    parser.add_argument(
        "--saturationVariationThresholdPct",
        action="store",
        help="Maximum variation in energy percentage from a time t to the end of the simulation to record the MCI as saturated at time t.",
        required = False,
        type=float
    )
    parser.add_argument(
        "--runNumber",
        action="store",
        help="Run number to analyse (folder must be in directory and named \'run_##\' where ## is runNumber).",
        required = False,
        type=int
    )
    parser.add_argument(
        "--displayPlots",
        action="store_true",
        help="Display plots in addition to saving to file.",
        required = False
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
        "--noLegend",
        action="store_true",
        help="No legend on plots for posters, papers etc. which will include captions or a centralised legend instead.",
        required = False
    )
    parser.add_argument(
        "--saveGammaPlots",
        action="store_true",
        help="Save max growth rate plots to file.",
        required = False
    )
    parser.add_argument(
        "--fastSpeciesName",
        action="store",
        help="Name of fast ion species.",
        required = False,
        type=str
    )
    parser.add_argument(
        "--bkgdSpeciesName",
        action="store",
        help="Name of background ion species.",
        required = False,
        type=str
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debugging statements.",
        required = False
    )
    
    args = parser.parse_args()

    debug = args.debug

    fields = args.fields if args.fields is not None else []

    if args.initFolders:
        init_fields = args.fields if args.fields is not None else []
        init_fields.append("energy") # Energy is no longer optional
        init_fields = list(set(init_fields)) # Remove any duplicates
        initialise_folder_structure(outputDirectory = args.outputDir, fields=init_fields)
        print("Output folder structure initialised, exiting....")
        exit(0)

    process_simulation_batch(
        simulationDataFolder=args.dir, 
        analysisOutputFolder=args.outputDir,
        runNumber=args.runNumber,
        fields=fields,
        maxK=args.maxK,
        maxW=args.maxW,
        growthRates=args.growthRates,
        bispectra = args.bispectra,
        fastIonSpecies=args.fastSpeciesName if args.fastSpeciesName is not None else 'He-4 2+',
        backgroundIonSpecies=args.bkgdSpeciesName if args.bkgdSpeciesName is not None else 'D+',
        numGrowthRatesToPlot=args.numGrowthRatesToPlot, 
        displayPlots=args.displayPlots,
        bigLabels=args.bigLabels,
        noTitle=args.noTitle,
        noLegend=args.noLegend,
        saveGrowthRatePlots=args.saveGammaPlots,
        mci_thresholds = {
            "mci_threshold_pct" : args.mciStartThresholdPct if args.mciStartThresholdPct is not None else 0.05, 
            "saturation_variation_threshold_pct" : args.saturationVariationThresholdPct if args.saturationVariationThresholdPct is not None else 0.01
        },
        debug = args.debug
    )
