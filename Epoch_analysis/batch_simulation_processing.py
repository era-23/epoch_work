import argparse
import os
from pathlib import Path
from matplotlib import pyplot as plt
from sdf_xarray import SDFPreprocess
from scipy import constants
from scipy.interpolate import make_smoothing_spline
from scipy.signal import find_peaks
from plasmapy.formulary import frequencies as ppf
from plasmapy.formulary import speeds as pps
from plasmapy.formulary import lengths as ppl
import astropy.units as u
import epoch_utils as e_utils
import netCDF4 as nc
import xarray as xr
import glob
import epydeck
import numpy as np
import shutil as sh
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import xrft  # noqa: E402

global debug

def initialise_folder_structure(
        outputDirectory : Path,
        fields : list,
        create : bool = False,
        energy : bool = True,
        plotGrowthRates : bool = False
) -> tuple[Path, Path]:
    
    dataFolder = outputDirectory / "data"
    plotsFolder = outputDirectory / "plots"
                
    if create:

        if debug:
            print(f"Creating folder structure in {outputDirectory}....")

        if os.path.exists(outputDirectory):
            sh.rmtree(outputDirectory)
        os.mkdir(outputDirectory)
        os.mkdir(dataFolder)
        os.mkdir(plotsFolder)

        if energy:
            energyPlotFolder = plotsFolder / "energy"
            os.mkdir(energyPlotFolder)

        for field in fields:
            plotFieldFolder = plotsFolder / field
            os.mkdir(plotFieldFolder)

            if plotGrowthRates:
                gammaPlotFolder = plotFieldFolder / "growth_rates"
                os.mkdir(gammaPlotFolder)

    
    return dataFolder, plotsFolder

def calculate_simulation_metadata(
        inputDeck : dict,
        dataset,
        outputNcRoot : nc.Dataset,
        fastSpecies : str = 'p+',
        bkgdSpecies : str = 'p+') -> tuple[float, float]:
    
    beam = False
    if "frac_beam" in inputDeck["constant"].keys():
        beam =  inputDeck["constant"]["frac_beam"] > 0.0

    # Check parameters in SI
    num_t = int(inputDeck['constant']['num_time_samples'])
    outputNcRoot.numTimePoints = num_t

    sim_time = float(dataset['Magnetic_Field_Bz'].coords["time"][-1]) * u.s
    outputNcRoot.simTime_s = sim_time
    outputNcRoot.timeSamplingFreq_Hz = num_t/sim_time
    outputNcRoot.timeNyquistFreq_Hz = num_t/(2.0 *sim_time)

    num_cells = int(inputDeck['constant']['num_cells'])
    outputNcRoot.numCells = num_cells

    sim_L = float(dataset['Magnetic_Field_Bz'].coords["X_Grid_mid"][-1]) * u.m
    outputNcRoot.simLength_m = sim_L
    outputNcRoot.spaceSamplingFreq_Pm = num_cells/sim_L
    outputNcRoot.spaceNyquistFreq_Pm = num_cells/(2.0 *sim_L)

    B0 = inputDeck['constant']['b0_strength']
    outputNcRoot.B0strength = B0
    
    B0_angle = inputDeck['constant']["b0_angle"]
    outputNcRoot.B0angle = B0_angle
    background_density = inputDeck['constant']['background_density']
    outputNcRoot.backgroundDensity = background_density
    beam_frac = inputDeck['constant']['frac_beam']
    outputNcRoot.beamFraction = beam_frac

    debye_length = ppl.Debye_length(inputDeck['constant']['background_temp'] * u.K, background_density / u.m**3)
    outputNcRoot.debyeLength_m = debye_length.value
    sim_L_dl = sim_L / debye_length
    outputNcRoot.simLength_dL = sim_L_dl
    outputNcRoot.cellWidth_dL = sim_L_dl / num_cells

    number_density_bkgd = background_density * (1.0 - beam_frac)
    if beam:
        ion_gyrofrequency = ppf.gyrofrequency(B0 * u.T, fastSpecies)
    else:
        ion_gyrofrequency = ppf.gyrofrequency(B0 * u.T, bkgdSpecies)

    outputNcRoot.ionGyrofrequency_radPs = ion_gyrofrequency.value
    ion_gyroperiod = 1.0 / ion_gyrofrequency
    outputNcRoot.ionGyroperiod_sPrad = ion_gyroperiod.value
    ion_gyroperiod_s = ion_gyroperiod * 2.0 * np.pi * u.rad
    
    outputNcRoot.ionGyroperiod_s = ion_gyroperiod_s.value
    plasma_freq = ppf.plasma_frequency(number_density_bkgd / u.m**3, bkgdSpecies)
    outputNcRoot.plasmaFrequency_radPs = plasma_freq.value

    alfven_velocity = pps.Alfven_speed(B0 * u.T, number_density_bkgd / u.m**3, bkgdSpecies)
    outputNcRoot.alfvenSpeed = alfven_velocity.value

    wLH_si = ppf.lower_hybrid_frequency(B0 * u.T, number_density_bkgd / u.m**3, bkgdSpecies)
    outputNcRoot.lhFrequency_radPs = wLH_si.value
    wLH_cyclo = wLH_si / ion_gyrofrequency
    outputNcRoot.lhFrequency_ionGyroF = wLH_cyclo.value
    
    # Lengths
    if beam:
        irb_v_perp = inputDeck['constant']['v_perp_ratio'] * alfven_velocity
        irb_gyroradius = proton_gyroradius = ppl.gyroradius(B = B0 * u.T, particle = "p", Vperp=irb_v_perp)
        outputNcRoot.fastIonGyroradius = irb_gyroradius.value
        outputNcRoot.simLength_rLfi = sim_L.value / irb_gyroradius.value
        outputNcRoot.cellWidth_rLfi = (sim_L.value / irb_gyroradius.value) / num_cells
    proton_gyroradius = ppl.gyroradius(B = B0 * u.T, particle = "p", T=inputDeck['constant']['background_temp'] * u.K)
    outputNcRoot.protonGyroradius = proton_gyroradius.value
    outputNcRoot.simLength_rLp = sim_L.value / proton_gyroradius.value
    outputNcRoot.cellWidth_rLp = (sim_L.value / proton_gyroradius.value) / num_cells
    electron_gyroradius = ppl.gyroradius(B = B0 * u.T, particle = "e", T=inputDeck['constant']['background_temp'] * u.K)
    outputNcRoot.electronGyroradius = electron_gyroradius.value
    outputNcRoot.simLength_rLe = sim_L.value / electron_gyroradius.value
    outputNcRoot.cellWidth_rLe = (sim_L.value / electron_gyroradius.value) / num_cells
    
    sim_L_vA_Tci = sim_L / (ion_gyroperiod_s * alfven_velocity)

    # Normalised units
    simtime_Tci = sim_time / ion_gyroperiod_s
    outputNcRoot.simTime_Tci = simtime_Tci
    outputNcRoot.timeSamplingFreq_Wci = num_t/simtime_Tci
    outputNcRoot.timeNyquistFreq_Wci = num_t/(2.0 *simtime_Tci)
    outputNcRoot.simLength_VaTci = sim_L_vA_Tci
    outputNcRoot.spaceSamplingFreq_WciOverVa = num_cells/sim_L_vA_Tci
    outputNcRoot.spaceNyquistFreq_WciOverVa = num_cells/(2.0 * sim_L_vA_Tci)

    if debug:
        print(f"Num time points: {num_t}")
        print(f"Sim time in SI: {sim_time}")
        print(f"Sampling frequency: {num_t/sim_time}")
        print(f"Nyquist frequency: {num_t/(2.0 *sim_time)}")
        print(f"Num cells: {num_cells}")
        print(f"Sim length: {sim_L}")
        print(f"Sampling frequency: {num_cells/sim_L}")
        print(f"Nyquist frequency: {num_cells/(2.0 *sim_L)}")
        
        print(f"B0 = {B0}")
        print(f"B0 angle = {B0_angle}")
        print(f"B0z = {B0 * np.sin(B0_angle * (np.pi / 180.0))}")

        print(f"Debye length: {debye_length}")
        print(f"Background ion gyroradius: {proton_gyroradius}")
        print(f"Background electron gyroradius: {electron_gyroradius}")
        if beam: 
            print(f"Fast ion gyroradius: {irb_gyroradius}")
        print(f"Sim length in Debye lengths: {sim_L_dl}")
        cell_width_dl = sim_L_dl /num_cells
        cell_width = cell_width_dl * debye_length
        if beam:
            print(f"Cell width: {cell_width_dl:.3f} debye lengths ({cell_width/proton_gyroradius:.3f} proton gyroradii, {cell_width/electron_gyroradius:.3f} electron gyroradii, {cell_width/irb_gyroradius:.3f} fast ion gyroradii)")
        else:
            print(f"Cell width: {cell_width_dl/num_cells:.3f} debye lengths ({cell_width/proton_gyroradius:.3f} proton gyroradii, {cell_width/electron_gyroradius:.3f} electron gyroradii)")

        print(f"Ion gyrofrequency: {ion_gyrofrequency}")
        print(f"Ion gyroperiod: {ion_gyroperiod}")
        print(f"Ion gyroperiod: {ion_gyroperiod_s}")
        print(f"Plasma frequency: {plasma_freq}")
        print(f"Alfven speed: {alfven_velocity}")

        print(f"NORMALISED: Sim time in Tci: {simtime_Tci}")
        print(f"NORMALISED: Temporal sampling frequency in Wci: {num_t/simtime_Tci}")
        print(f"NORMALISED: Temporal Nyquist frequency in Wci: {num_t/(2.0 *simtime_Tci)}")
        print(f"NORMALISED: Sim L in vA*Tci: {sim_L_vA_Tci}")
        print(f"NORMALISED: Spatial sampling frequency in Wci/vA: {num_cells/sim_L_vA_Tci}")
        print(f"NORMALISED: Nyquist frequency in Wci/vA: {num_cells/(2.0 * sim_L_vA_Tci)}")

    return ion_gyroperiod_s, alfven_velocity

def normalise_data(dataset : xr.Dataset, ion_gyroperiod : float, alfven_velocity : float) -> xr.Dataset:
    
    evenly_spaced_time = np.linspace(dataset.coords["time"][0].data, dataset.coords["time"][-1].data, len(dataset.coords["time"].data))
    dataset = dataset.interp(time=evenly_spaced_time)
    Tci = evenly_spaced_time / ion_gyroperiod
    vA_Tci = dataset.coords["X_Grid_mid"] / (ion_gyroperiod * alfven_velocity)
    dataset = dataset.drop_vars("X_Grid")
    dataset = dataset.assign_coords({"time" : Tci, "X_Grid_mid" : vA_Tci})
    dataset = dataset.rename(X_Grid_mid="x_space")

    return dataset

def run_energy_analysis(
    dataset : xr.Dataset,
    inputDeck : dict,
    simName : str,
    savePlotsFolder : Path,
    statsFile : nc.Dataset,
    displayPlots : bool = False,
    noTitle : bool = False
):
    beam = False
    if "frac_beam" in inputDeck["constant"].keys():
        beam =  inputDeck["constant"]["frac_beam"] > 0.0

    # Create stats group if not already existing
    if "Energy" not in statsFile.groups.keys():
        energyStats = statsFile.createGroup("Energy")
        energyStats.createDimension("time", dataset.coords["time"].size)
        time_var = energyStats.createVariable("time", "f8", ("time",))
        ped = energyStats.createVariable("protonMeanEnergyDensity", "f8", ("time",))
        eed = energyStats.createVariable("electronMeanEnergyDensity", "f8", ("time",))
        efd = energyStats.createVariable("electricFieldMeanEnergyDensity", "f8", ("time",))
        mfd = energyStats.createVariable("magneticFieldMeanEnergyDensity", "f8", ("time",))
    else:
        energyStats = statsFile.groups["Energy"]
        time_var = energyStats.variables["time"]
        ped = energyStats.variables["protonMeanEnergyDensity"]
        eed = energyStats.variables["electronMeanEnergyDensity"]
        efd = energyStats.variables["electricFieldMeanEnergyDensity"]
        mfd = energyStats.variables["magneticFieldMeanEnergyDensity"]
    
    time_var[:] = dataset.coords["time"].data
    energyStats.long_name = "Particle and field energy data"

    bkgd_density = inputDeck['constant']['background_density']
    frac_beam = inputDeck['constant']['frac_beam']

    proton_density = bkgd_density * (1.0 - frac_beam) # m^-3
    electron_density = bkgd_density # m^-3

    # Mean over all cells for mean particle/field energy
    proton_KE : xr.DataArray = dataset['Derived_Average_Particle_Energy_proton'].load()
    protonKE_mean = proton_KE.mean(dim = "x_space").data # J
    protonKEdensity_mean = protonKE_mean * proton_density # J / m^3
    del(proton_KE)
    del(protonKE_mean)
    
    ped[:] = protonKEdensity_mean

    electron_KE : xr.DataArray = dataset['Derived_Average_Particle_Energy_electron'].load()
    electronKE_mean = electron_KE.mean(dim = "x_space").data # J
    electronKEdensity_mean = electronKE_mean * electron_density # J / m^3
    del(electron_KE)
    del(electronKE_mean)
    
    eed[:] = electronKEdensity_mean

    Ex : xr.DataArray = dataset['Electric_Field_Ex'].load()
    Ey : xr.DataArray = dataset['Electric_Field_Ey'].load()
    Ez : xr.DataArray = dataset['Electric_Field_Ez'].load()
    electricFieldStrength = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    del(Ex, Ey, Ez)
    electricFieldEnergyDensity : xr.DataArray = (constants.epsilon_0 * electricFieldStrength**2) / 2.0 # J / m^3
    electricFieldDensity_mean = electricFieldEnergyDensity.mean(dim="x_space").data # J / m^3
    del(electricFieldEnergyDensity)
    
    efd[:] = electricFieldDensity_mean

    Bx : xr.DataArray = dataset['Magnetic_Field_Bx'].load()
    By : xr.DataArray = dataset['Magnetic_Field_By'].load()
    Bz : xr.DataArray = dataset['Magnetic_Field_Bz'].load()
    magneticFieldStrength = np.sqrt(Bx**2 + By**2 + Bz**2)
    del(Bx, By, Bz)
    magneticFieldEnergyDensity : xr.DataArray = (magneticFieldStrength**2 / (2.0 * constants.mu_0)) # J / m^3
    magneticFieldEnergyDensity_mean = magneticFieldEnergyDensity.mean(dim = "x_space").data # J / m^3
    del(magneticFieldEnergyDensity)
    
    mfd[:] = magneticFieldEnergyDensity_mean

    # Calculate B and E energy and convert others to to J/m3
    deltaMeanMagneticEnergyDensity = magneticFieldEnergyDensity_mean - magneticFieldEnergyDensity_mean[0] # J / m^3
    deltaMeanElectricEnergyDensity = electricFieldDensity_mean - electricFieldDensity_mean[0] # J / m^3
    deltaProtonKE_density = protonKEdensity_mean - protonKEdensity_mean[0] # J / m^3
    deltaElectronKE_density = electronKEdensity_mean - electronKEdensity_mean[0] # J / m^3

    totalAbsoluteMeanEnergyDensity = protonKEdensity_mean + electronKEdensity_mean + magneticFieldEnergyDensity_mean + electricFieldDensity_mean
    totalDeltaMeanEnergyDensity = deltaProtonKE_density + deltaElectronKE_density + deltaMeanMagneticEnergyDensity + deltaMeanElectricEnergyDensity
    timeCoords = dataset.coords['time']

    # Write stats
    # Start, end and peak energy densities, and times of maximum and minimum
    if beam:

        fastIonDensity = bkgd_density * frac_beam # m^-3
        fastIonKE : xr.DataArray = dataset['Derived_Average_Particle_Energy_ion_ring_beam'].load()
        fastIonKE_mean = fastIonKE.mean(dim = "x_space").data # J
        fastIonKEdensity_mean = fastIonKE_mean * fastIonDensity # J / m^3
        del(fastIonKE)
        del(fastIonKE_mean)
        if "fastIonMeanEnergyDensity" not in energyStats.variables.keys():
            fed = energyStats.createVariable("fastIonMeanEnergyDensity", "f8", ("time",))
        else:
            fed = energyStats.variables["fastIonMeanEnergyDensity"]
        fed[:] = fastIonKEdensity_mean
        deltaFastIonKE_density = fastIonKEdensity_mean - fastIonKEdensity_mean[0] # J / m^3
        totalAbsoluteMeanEnergyDensity += fastIonKEdensity_mean
        totalDeltaMeanEnergyDensity += deltaFastIonKE_density

        energyStats.fastIonEnergyDensity_start = fastIonKEdensity_mean[0]
        energyStats.fastIonEnergyDensity_end = fastIonKEdensity_mean[-1]
        index = np.nanargmax(fastIonKEdensity_mean)
        energyStats.fastIonEnergyDensity_max = fastIonKEdensity_mean[index]
        energyStats.fastIonEnergyDensity_timeMax = timeCoords[index]
        index = np.nanargmin(fastIonKEdensity_mean)
        energyStats.fastIonEnergyDensity_min = fastIonKEdensity_mean[index]
        energyStats.fastIonEnergyDensity_timeMin = timeCoords[index]
        energyStats.fastIonEnergyDensity_delta = deltaFastIonKE_density[-1]

    energyStats.backgroundIonEnergyDensity_start = protonKEdensity_mean[0]
    energyStats.backgroundIonEnergyDensity_end = protonKEdensity_mean[-1]
    index = np.nanargmax(protonKEdensity_mean)
    energyStats.backgroundIonEnergyDensity_max = protonKEdensity_mean[index]
    energyStats.backgroundIonEnergyDensity_timeMax = timeCoords[index]
    index = np.nanargmin(protonKEdensity_mean)
    energyStats.backgroundIonEnergyDensity_min = protonKEdensity_mean[index]
    energyStats.backgroundIonEnergyDensity_timeMin = timeCoords[index]
    energyStats.backgroundIonEnergyDensity_delta = deltaProtonKE_density[-1]

    energyStats.electronEnergyDensity_start = electronKEdensity_mean[0]
    energyStats.electronEnergyDensity_end = electronKEdensity_mean[-1]
    index = np.nanargmax(electronKEdensity_mean)
    energyStats.electronEnergyDensity_max = electronKEdensity_mean[index]
    energyStats.electronEnergyDensity_timeMax = timeCoords[index]
    index = np.nanargmin(electronKEdensity_mean)
    energyStats.electronEnergyDensity_min = electronKEdensity_mean[index]
    energyStats.electronEnergyDensity_timeMin = timeCoords[index]
    energyStats.electronEnergyDensity_delta = deltaElectronKE_density[-1]

    energyStats.electricFieldEnergyDensity_start = electricFieldDensity_mean[0]
    energyStats.electricFieldEnergyDensity_end = electricFieldDensity_mean[-1]
    index = np.nanargmax(electricFieldDensity_mean)
    energyStats.electricFieldEnergyDensity_max = electricFieldDensity_mean[index]
    energyStats.electricFieldEnergyDensity_timeMax = timeCoords[index]
    index = np.nanargmin(electricFieldDensity_mean)
    energyStats.electricFieldEnergyDensity_min = electricFieldDensity_mean[index]
    energyStats.electricFieldEnergyDensity_timeMin = timeCoords[index]
    energyStats.electricFieldEnergyDensity_delta = deltaMeanElectricEnergyDensity[-1]

    energyStats.magneticFieldEnergyDensity_start = magneticFieldEnergyDensity_mean[0]
    energyStats.magneticFieldEnergyDensity_end = magneticFieldEnergyDensity_mean[-1]
    index = np.nanargmax(magneticFieldEnergyDensity_mean)
    energyStats.magneticFieldEnergyDensity_max = magneticFieldEnergyDensity_mean[index]
    energyStats.magneticFieldEnergyDensity_timeMax = timeCoords[index]
    index = np.nanargmin(magneticFieldEnergyDensity_mean)
    energyStats.magneticFieldEnergyDensity_min = magneticFieldEnergyDensity_mean[index]
    energyStats.magneticFieldEnergyDensity_timeMin = timeCoords[index]
    energyStats.magneticFieldEnergyDensity_delta = deltaMeanMagneticEnergyDensity[-1]

    # Totals
    energyStats.totalEnergyDensity_start = float(totalAbsoluteMeanEnergyDensity[0])
    energyStats.totalEnergyDensity_end = float(totalAbsoluteMeanEnergyDensity[-1])
    pctConservation = float(100.0 * ((totalAbsoluteMeanEnergyDensity[-1]-totalAbsoluteMeanEnergyDensity[0])/totalAbsoluteMeanEnergyDensity[0]))
    energyStats.totalEnergyDensityConservation_pct = pctConservation

    if debug:
        print(f"---------------------- ENERGY: {simName} ---------------------")
        print(f"Total energy start: {float(totalAbsoluteMeanEnergyDensity[0])}")
        print(f"Total energy end: {float(totalAbsoluteMeanEnergyDensity[-1])}")
        print(f"Total energy conservation: {pctConservation}%")

    deltaEnergies = {
        "protonMeanEnergyDensity" : deltaProtonKE_density,
        "electronMeanEnergyDensity" : deltaElectronKE_density,
        "magneticFieldMeanEnergyDensity" : deltaMeanMagneticEnergyDensity,
        "electricFieldMeanEnergyDensity" : deltaMeanElectricEnergyDensity
    }
    percentageBaseline = float(totalAbsoluteMeanEnergyDensity[0])
    if beam:
        deltaEnergies["fastIonMeanEnergyDensity"] = deltaFastIonKE_density
        percentageBaseline = float(fastIonKEdensity_mean[0])

    maxPeakIndices = {}
    minTroughIndices = {}
    pctEnergies = {}

    maxExtent = np.max([np.array(v) for v in deltaEnergies.values()])
    minExtent = np.min([np.array(v) for v in deltaEnergies.values()])
    dataRange = maxExtent - minExtent
    prominence = 0.01 * dataRange # Peak prominence must be at least 1% of the data range

    # Initialise plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    filename = Path(f"{simName}_percentage_energy_change.png")

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

        # Record
        smoothPctData = 100.0 * (smoothDeltaData / percentageBaseline)
        hasPeaks = bool(ed_peaks.size > 0)
        hasTroughs = bool(ed_troughs.size > 0)

        maxAtSimEnd = bool((len(smoothDeltaData)-1)-np.argmax(smoothDeltaData) < 5)
        minAtSimEnd = bool((len(smoothDeltaData)-1)-np.argmin(smoothDeltaData) < 5)
        energyStats[variable].maxAtSimEnd = int(maxAtSimEnd)
        energyStats[variable].minAtSimEnd = int(minAtSimEnd)     

        if debug:
            print(f"Variable: {variable}")
            print(f"Has peaks: {hasPeaks}")
            print(f"Has troughs: {hasTroughs}")
            print(f"Time series length: {len(smoothDeltaData)}, index max: {np.argmax(smoothDeltaData)}, maxAtSimEnd: {maxAtSimEnd}")
            print(f"Time series length: {len(smoothDeltaData)}, index min: {np.argmin(smoothDeltaData)}, minAtSimEnd: {minAtSimEnd}")
        
        colour = next((e_utils.E_TRACE_SPECIES_COLOUR_MAP[c] for c in e_utils.E_TRACE_SPECIES_COLOUR_MAP.keys() if c in variable), False)
        if colour:
            ax.plot(timeCoords, percentageED,  label=e_utils.SPECIES_NAME_MAP[variable], alpha=0.4, color = colour)
            ax.plot(timeCoords, smoothPctData, label=f"Smoothed {e_utils.SPECIES_NAME_MAP[variable]}", linestyle="--", color = colour)
        else:
            ax.plot(timeCoords, percentageED,  label=e_utils.SPECIES_NAME_MAP[variable], alpha=0.4)
            ax.plot(timeCoords, smoothPctData, label=f"Smoothed {e_utils.SPECIES_NAME_MAP[variable]}", linestyle="--")
            
        energyStats[variable].hasPeaks = int(hasPeaks)
        energyStats[variable].hasTroughs = int(hasTroughs)
        
        if hasPeaks:
            energyStats[variable].peakIndices = ed_peaks
            energyStats[variable].peakValues_delta = smoothDeltaData[ed_peaks]
            energyStats[variable].peakValues_pct = smoothPctData[ed_peaks]
            energyStats[variable].peakTimes = [float(t) for t in timeCoords[ed_peaks]]
            if debug:
                print(f"Peaks: {smoothDeltaData[ed_peaks]} ({smoothPctData[ed_peaks]}%) at {ed_peaks}")
                print(f"Time of peaks: {[float(t) for t in timeCoords[ed_peaks]]}")
            
            ax.scatter(timeCoords[ed_peaks], smoothPctData[ed_peaks], marker="x", color="black")
            maxPeakIndices[variable] = ed_peaks[np.argmax([smoothPctData[p] for p in ed_peaks])]
        
        if hasTroughs:
            energyStats[variable].troughIndices = ed_troughs
            energyStats[variable].troughValues_delta = smoothDeltaData[ed_troughs]
            energyStats[variable].troughValues_pct = smoothPctData[ed_troughs]
            energyStats[variable].troughTimes = [float(t) for t in timeCoords[ed_troughs]]
            if debug:
                print(f"Troughs: {smoothDeltaData[ed_troughs]} ({smoothPctData[ed_troughs]}%) at {ed_troughs}")
                print(f"Time of troughs: {[float(t) for t in timeCoords[ed_troughs]]}")
                print("...................................................................................")
            ax.scatter(timeCoords[ed_troughs], smoothPctData[ed_troughs], marker="+", color="black")
            minTroughIndices[variable] = ed_troughs[np.argmin([smoothPctData[p] for p in ed_troughs])]
        
    ax.legend()
    ax.set_xlabel(r"Time [$\tau_{ci}$]")
    ax.set_ylabel("Change in energy density [%]")
    ax.grid()
    if not noTitle:
        if beam:
            ax.set_title(f"{simName}: Percentage change in ED relative to fast ion energy")
        else:
            ax.set_title(f"{simName}: Percentage change in ED relative to total starting energy")
    fig.tight_layout()
    fig.savefig(savePlotsFolder / filename)
    if displayPlots:
        plt.show()
    plt.close("all")
    
    # Specific to IRB transfer
    if beam:
        fastVar = "fastIonMeanEnergyDensity"
        protVar = "protonMeanEnergyDensity"
        elecVar = "electronMeanEnergyDensity"
        fast = deltaEnergies[fastVar]
        fast_pct = pctEnergies[fastVar]
        proton = deltaEnergies[protVar]
        proton_pct = pctEnergies[protVar]
        electron = deltaEnergies[elecVar]
        electron_pct = pctEnergies[elecVar]

        hasFastIonGain = bool(fast[-1] > 0.0) or energyStats[fastVar].hasPeaks
        hasBkgdIonGain = bool(proton[-1] > 0.0)
        hasBkgdElectronGain = bool(electron[-1] > 0.0)

        energyStats.hasOverallFastIonGain = int(hasFastIonGain)
        energyStats.hasOverallBkgdIonGain = int(hasBkgdIonGain)
        energyStats.hasOverallBkgdElectronGain = int(hasBkgdElectronGain)

        if debug:
            print(f"Overall fast ion gain: {hasFastIonGain}") # bool
            print(f"Overall background proton gain: {hasBkgdIonGain}") # bool
            print(f"Overall background electron gain: {hasBkgdElectronGain}") # bool

        if fastVar in minTroughIndices.keys():

            pGainAtFiTrough = float(proton[minTroughIndices[fastVar]])
            pGainAtFiTrough_pct = float(proton_pct[minTroughIndices[fastVar]])
            eGainAtFiTrough = float(electron[minTroughIndices[fastVar]])
            eGainAtFiTrough_pct = float(electron_pct[minTroughIndices[fastVar]])

            energyStats.bkgdIonChangeAtFastIonTrough = pGainAtFiTrough
            energyStats.bkgdIonChangeAtFastIonTrough_pct = pGainAtFiTrough_pct
            energyStats.bkgdElectronChangeAtFastIonTrough = eGainAtFiTrough
            energyStats.bkgdElectronChangeAtFastIonTrough_pct = eGainAtFiTrough_pct

            if debug:
                print(f"Fast ion trough: {minTroughIndices[fastVar]}") # numeric and already recorded above
                print(f"Proton gain at fast ion trough: {pGainAtFiTrough} ({pGainAtFiTrough_pct}%)") # numeric
                print(f"Electron gain at fast ion trough: {eGainAtFiTrough} ({eGainAtFiTrough_pct}%)")# numeric

        if protVar in maxPeakIndices.keys():
            fiLossAtProtonPeak = float(fast[maxPeakIndices[protVar]])
            fiLossAtProtonPeak_pct = float(fast_pct[maxPeakIndices[protVar]])
            energyStats.fastIonChangeAtBkgdIonPeak = fiLossAtProtonPeak
            energyStats.fastIonChangeAtBkgdIonPeak_pct = fiLossAtProtonPeak_pct

            if debug:
                print(f"Fast ion loss at proton peak: {fiLossAtProtonPeak} ({fiLossAtProtonPeak_pct}%)") # numeric
        
        if elecVar in maxPeakIndices.keys():
            fiLossAtElectronPeak = float(fast[maxPeakIndices[elecVar]])
            fiLossAtElectronPeak_pct = float(fast_pct[maxPeakIndices[elecVar]])
            energyStats.fastIonChangeAtBkgdElectronPeak = fiLossAtElectronPeak
            energyStats.fastIonChangeAtBkgdElectronPeak_pct = fiLossAtElectronPeak_pct

            if debug:
                print(f"Fast ion loss at electron peak: {fiLossAtElectronPeak} ({fiLossAtElectronPeak_pct}%)") # numeric

        if debug:
            print("------------------------------------------------------------------------------------")
 
    # Initialise absolute energy plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    filename = Path(f"{simName}_absolute_energy_change.png")
    if beam:
        ax.plot(timeCoords, deltaFastIonKE_density, label = r"Fast ion", color="red")
    ax.plot(timeCoords, deltaProtonKE_density, label = r"Bkgd proton", color="orange")
    ax.plot(timeCoords, deltaElectronKE_density, label = r"Bkgd electron", color="blue")
    ax.plot(timeCoords, deltaMeanMagneticEnergyDensity, label = r"B-field", color="purple", linestyle="--")
    ax.plot(timeCoords, deltaMeanElectricEnergyDensity, label = r"E-field", color="green", linestyle="--")
    ax.plot(timeCoords, totalDeltaMeanEnergyDensity, label = r"Total", color="black")
    ax.set_xlabel(r'Time [$\tau_{ci}$]')
    ax.set_ylabel(r"Change in energy density [$J/m^3$]")
    if not noTitle: 
        ax.set_title(f"{simName}: Absolute energy in particles and EM fields", wrap=True)
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(savePlotsFolder / filename)
    if displayPlots:
        plt.show()
    plt.close("all")
            
def process_simulation_batch(
        directory : Path,
        dataFolder : Path,
        plotsFolder : Path,
        fields : list,
        maxK : float = None,
        maxW : float = None,
        growthRates : bool = True,
        bispectra : bool = True,
        gammaWindowPctMin : float = 5.0,
        gammaWindowPctMax : float = 15.0,
        fastSpecies : str = 'p+',
        bkgdSpecies : str = 'p+',
        bigLabels : bool = False,
        noTitle : bool = False,
        createPlots = False,
        displayPlots = False,
        saveGrowthRatePlots = False,
        numGrowthRatesToPlot : int = 0,
        energy = True):
    """
    Processes a batch of simulations:
    - Calculates plasma characteristics
    - Normalises to ion cyclotron frequency (fast species if present) and Alfven velocity
    - Creates plots (optionally displays) of omega-k and t-k of 'field' (or optionally delta field) up to maxK
    - Saves plots to outFileDirectory (or an output folder in 'directory' if not specified)
    - Calculates growth rates by k
    - Saves growth rates and plasma characteristics to netCDF

    ### Parameters:
        directory: Path -- Directory of simulation outputs, containing "run_n" folders each with sdf output files and an input.deck
        field: list = 'all' -- List of string fields to use for analysis, either EPOCH output fields such as ['Magnetic_Field_Bz', 'Electric_Field_Ex']
            or ['all'] to process all magnetic and electric field components. Defaults to Bz
        outFileDirectory : Path = None -- directory to use for output, defaults to the input directory
        numGrowthRates : int = 10 -- number of wavenumbers for which to calculate growth rates
        maxK : float = 100.0 -- Maximum wavenumber for plots and analysis
        maxW : float = None -- Maximum frequency for plots and analysis
        maxResPct : float = 0.1 -- Percentage of growth rates to use based on ranked residual values, e.g. 0.1 = top 10% of gammas ranked by (lowest) residuals (default).
        maxResidual : float = 0.1 -- Maximum residual value to consider a well-conditioned fit. Must be baselined from null cases 
        gammaWindowPct : float = 5.0 -- Percentage of trace (in time) to use as a growth rate fitting window. Defaults to 5%
        minSignalPower : float = 0.08 -- Minimum peak signal power to observe in frequency-wavenumber space to discern MCI activity. Defaults to 0.08T (FIELD DEPENDENT and must be baselined from null cases.) 
        takeLog = False -- Take logarithm of data 
        deltaField = False -- Consider the change in field strength from t = 1 (t = 0 is discarded as initial conditions) rather than absolute values
        beam = True -- Take ion ring beam (fast) as significant ion species 
        fastSpecies : str = 'proton' -- Fast (ion ring beam) species, if present
        bkgdSpecies : str = 'proton' -- Background ion species
        createPlots = True -- Generate plots
        displayPlots = False -- Display plots as they are generated
    """

    run_folders = []
    if directory.name.startswith("run_"): # Single simulation
        run_folders.append(directory)
    else: # Multiple simulations
        run_folders = glob.glob(str(directory / "run_*") + os.path.sep) 

    if bigLabels:
        plt.rcParams.update({'axes.titlesize': 26.0})
        plt.rcParams.update({'axes.labelsize': 24.0})
        plt.rcParams.update({'xtick.labelsize': 20.0})
        plt.rcParams.update({'ytick.labelsize': 20.0})
        plt.rcParams.update({'legend.fontsize': 18.0})
    else:
        plt.rcParams.update({'axes.titlesize': 18.0})
        plt.rcParams.update({'axes.labelsize': 16.0})
        plt.rcParams.update({'xtick.labelsize': 14.0})
        plt.rcParams.update({'ytick.labelsize': 14.0})
        plt.rcParams.update({'legend.fontsize': 14.0})
    
    for simFolder in run_folders:

        simFolder = Path(simFolder)
        print(f"Analyzing simulation '{simFolder.name}'")

        # Read dataset
        ds = xr.open_mfdataset(
            str(simFolder / "*.sdf"),
            data_vars='minimal', 
            coords='minimal', 
            compat='override', 
            preprocess=SDFPreprocess()
        )

        # Drop initial conditions because they may not represent a solution
        ds = ds.sel(time=ds.coords["time"]>ds.coords["time"][0])

        # Read input deck
        inputDeck = {}
        with open(str(simFolder / "input.deck")) as id:
            inputDeck = epydeck.loads(id.read())
        
        statsFilename = simFolder.name + "_stats.nc"
        statsFilepath = os.path.join(dataFolder, statsFilename)
        statsRoot = nc.Dataset(statsFilepath, "a", format="NETCDF4")

        ion_gyroperiod, alfven_velocity = calculate_simulation_metadata(inputDeck, ds, statsRoot, fastSpecies, bkgdSpecies)

        ds = normalise_data(ds, ion_gyroperiod, alfven_velocity)

        # Energy analysis
        if energy:
            energyPlotFolder = plotsFolder / "energy"
            run_energy_analysis(ds, inputDeck, simFolder.name, energyPlotFolder, statsRoot, displayPlots = displayPlots, noTitle=noTitle)

        if "all" in fields:
            fields = [str(f) for f in ds.data_vars.keys() if str(f).startswith("Electric_Field") or str(f).startswith("Magnetic_Field")]
        
        for field in fields:

            print(f"Analyzing field '{field}'...")
            plotFieldFolder = Path(os.path.join(plotsFolder, field))

            # For field-specific stats
            fieldStats = statsRoot.createGroup(field)
            field_unit = ds[field].units
            fieldStats.baseUnit = field_unit
            field_mag = float(np.abs(ds[field].sum()))
            fieldStats.totalMagnitude = field_mag
            dx = float(ds.coords['x_space'][2] - ds.coords['x_space'][1])
            dy = float(ds.coords['time'][2] - ds.coords['time'][1])
            parseval_field = float((np.abs(ds[field])**2).sum()) * dx * dy
            fieldStats.parsevalField = parseval_field
            if debug:
                print(f"Sum of squared {field} magnitude * dx * dy: {parseval_field}")
            field_mean = float(ds[field].mean())
            fieldStats.meanMagnitude = field_mean
            delta = np.abs((ds[field] - field_mean))
            fieldStats.totalDelta = float(delta.sum())
            squared_delta = float((np.abs(ds[field] - field_mean)**2).sum())
            parseval_fieldDelta = squared_delta * dx * dy
            fieldStats.parsevalFieldDelta = parseval_fieldDelta
            if debug:
                print(f"Sum of squared {field} delta * dx * dy: {parseval_fieldDelta}")
            field_dMean = float(delta.mean())
            fieldStats.meanDelta = field_dMean
            if debug:
                print(f"Mean(field - mean): {field_dMean}")
            del(delta)
            del(squared_delta)

            if createPlots or growthRates or bispectra:
                # Take FFT
                original_spec : xr.DataArray = xrft.xrft.fft(ds[field], true_amplitude=True, true_phase=True, window=None)
                original_spec = original_spec.rename(freq_time="frequency", freq_x_space="wavenumber")
                # Remove zero-frequency component
                original_spec = original_spec.where(original_spec.wavenumber!=0.0, None)

                tk_spec = e_utils.create_t_k_spectrum(original_spec, fieldStats, maxK, load=True, debug=debug)

                # Dispersion relations
                if createPlots:
                    e_utils.create_omega_k_plots(original_spec, fieldStats, field, field_unit, plotFieldFolder, simFolder.name, inputDeck, bkgdSpecies, fastSpecies, maxK=maxK, maxW=maxW, display=displayPlots, debug=debug)
                    e_utils.create_t_k_plot(tk_spec, field, field_unit, plotFieldFolder, simFolder.name, maxK, displayPlots)

                if bispectra:
                    e_utils.bispectral_analysis(tk_spec, simFolder.name, field, displayPlots, plotFieldFolder, maxK = maxK)

                # Linear growth rates
                if growthRates:
                    e_utils.process_growth_rates(tk_spec, fieldStats, plotFieldFolder, simFolder, field, gammaWindowPctMin, gammaWindowPctMax, saveGrowthRatePlots, numGrowthRatesToPlot, displayPlots, noTitle, debug)
    
        statsRoot.close()
        ds.close()
            
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
        "--runNumber",
        action="store",
        help="Run number to analyse (folder must be in directory and named \'run_##\' where ## is runNumber).",
        required = False,
        type=int
    )
    parser.add_argument(
        "--createPlots",
        action="store_true",
        help="Create dispersion plots and save to file.",
        required = False
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
        "--energy",
        action="store_true",
        help="Run energy analysis.",
        required = False
    )
    parser.add_argument(
        "--saveGammaPlots",
        action="store_true",
        help="Save max growth rate plots to file.",
        required = False
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debugging statements.",
        required = False
    )
    
    args = parser.parse_args()

    debug = args.debug

    # Check outputDir
    createFolders = True
    if args.outputDir is not None:
        if Path.is_dir(args.outputDir):
            if os.path.exists(os.path.join(args.outputDir, 'data')):
                print(f"Existing analysis folder found at '{args.outputDir}'")
                createFolders = False
            else:
                createFolders = True
        else:
            createFolders = True

        outputDirectory = args.outputDir
    else:
        outputDirectory = args.dir / Path("analysis")
        if not os.path.exists(outputDirectory):
            os.mkdir(outputDirectory)

    print(f"Using analysis folder at '{outputDirectory}'")

    dataFolder, plotsFolder = initialise_folder_structure(outputDirectory, args.fields, createFolders, args.energy, args.saveGammaPlots)

    if args.runNumber is not None:
        args.dir = Path(os.path.join(args.dir, f"run_{args.runNumber}"))

    process_simulation_batch(
        directory=args.dir, 
        dataFolder=dataFolder,
        plotsFolder=plotsFolder,
        fields=args.fields if args.fields is not None else [],
        maxK=args.maxK,
        maxW=args.maxW,
        growthRates=args.growthRates,
        bispectra = args.bispectra,
        gammaWindowPctMin=args.minGammaFitWindow,
        gammaWindowPctMax=args.maxGammaFitWindow,
        numGrowthRatesToPlot=args.numGrowthRatesToPlot, 
        createPlots=args.createPlots,
        displayPlots=args.displayPlots,
        bigLabels=args.bigLabels,
        noTitle=args.noTitle,
        saveGrowthRatePlots=args.saveGammaPlots,
        energy=args.energy)