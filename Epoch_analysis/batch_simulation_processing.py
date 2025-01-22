import argparse
import csv
import os
from pathlib import Path
from matplotlib import pyplot as plt
from sdf_xarray import SDFPreprocess
from typing import List
from plasmapy.formulary import frequencies as ppf
from plasmapy.formulary import speeds as pps
from plasmapy.formulary import lengths as ppl
import astropy.units as u
import epoch_utils as utils
import netCDF4 as nc
import xarray as xr
import glob
import epydeck
import numpy as np
import numpy.polynomial.polynomial as poly
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import xrft  # noqa: E402

global debug

def initialise_folder_structure(
        dataDirectory : Path,
        outFileDirectory : Path = None
) -> tuple[Path, Path, Path, Path]:
    
    if outFileDirectory is None:
        outFileDirectory = dataDirectory / "analysis"
        if not os.path.exists(outFileDirectory):
            os.mkdir(outFileDirectory)
    metadataFolder = outFileDirectory / "metadata"
    if not os.path.exists(metadataFolder):
        os.mkdir(metadataFolder)
    growthDataFolder = outFileDirectory / "growth_rates"
    if not os.path.exists(growthDataFolder):
        os.mkdir(growthDataFolder)
    plotsFolder = outFileDirectory / "plots"
    if not os.path.exists(plotsFolder):
        os.mkdir(plotsFolder)

    return metadataFolder, growthDataFolder, plotsFolder

def create_netCDF_data_structure(
        root : nc.Dataset,
        numGrowthRates : int
) -> nc.Dataset:
    growth_rate_group = root.createGroup("growthRates")
    growth_rate_group.createDimension("rank", numGrowthRates)
    growth_rate_group.createDimension("wavenumber")
    growth_rate_group.createDimension("time")
    growth_rate_group.createDimension("yIntercept")
    growth_rate_group.createDimension("residual")
    k_var = growth_rate_group.createVariable("wavenumber", "f4", ("wavenumber",))
    k_var.units = "wCI/vA"
    t_var = growth_rate_group.createVariable("time", "f4", ("time",))
    t_var.units = "tauCI"
    growth_rate_group.createVariable("yIntercept", "f4", ("yIntercept",))
    growth_rate_group.createVariable("residual", "f4", ("yIntercept",))
    gamma_var = growth_rate_group.createVariable("growthRate", "f8", ("rank", "wavenumber", "time", "yIntercept", "residual",))
    gamma_var.units = "wCI"
    gamma_var.standard_name = "linear_growth_rate"

    peak_power_group = growth_rate_group.createGroup("maxPeakPowerWavenumbers")
    peak_power_group.selectionCriteria = "maximumPeakPowerK"

    total_power_group = growth_rate_group.createGroup("maxTotalPowerWavenumbers")
    total_power_group.selectionCriteria = "maximumTotalPowerK"

    return root
    
def find_max_growth_rates(
        tkSpectrum : xr.DataArray,
        wavenumberIndicesToCalculate : np.ndarray,
        gammaWindow : int,
        savePlots : bool = False,
        displayPlots : bool = False
) -> List[utils.LinearGrowthRate] :

    max_growth_rates = []

    for index in wavenumberIndicesToCalculate:

        # Must load data here for quick iteration over windows
        signal = tkSpectrum.isel(wavenumber=index).load()

        signalK=tkSpectrum.coords['wavenumber'][index]
        signalPeak=float(signal.max())
        signalTotal=float(signal.sum())

        signal_growth_rates = []

        for w in range(len(signal) - (gammaWindow + 1)): # For each window

            t_k_window = signal[w:(w + gammaWindow)]
            coefs, stats = poly.polyfit(x = signal.coords["time"][w:(w + gammaWindow)], y = np.log(t_k_window), deg = 1, full = True)
            w_gamma = coefs[1]
            y_int = coefs[0]
            res = stats[0][0]
            signal_growth_rates.append(
                utils.LinearGrowthRate(timeStartIndex=w,
                                       timeEndIndex=(w + gammaWindow),
                                       timeMidpointIndex=int(gammaWindow/2),
                                       gamma=w_gamma,
                                       yIntercept=y_int,
                                       residual=res))
        del(signal)
            
        max_signal_growth_rate_index = np.nanargmax([lgr.gamma for lgr in signal_growth_rates])
        gamma : utils.LinearGrowthRate = signal_growth_rates[max_signal_growth_rate_index]
        gamma.wavenumber = signalK
        gamma.timeMidpoint = float(tkSpectrum.coords['time'][gamma.timeMidpointIndex])
        gamma.peakPower = signalPeak
        gamma.totalPower = signalTotal
        max_growth_rates.append(gamma)
        if debug: 
            print(f"Max gamma found: {gamma.to_string()}")

    return max_growth_rates

def find_max_growth_rates_of_top_n_k_with_max_total_power(
        tkSpectrum : xr.DataArray,
        field : str,
        runName : str,
        gammaWindow : int = 100,
        n : int = 10,
        savePlots : bool = False,
        displayPlots : bool = False,
        numPlotsToSaveDisplay : int = 0,
        saveFolder : Path = None
):
    # Short-circuit
    if n == 0:
        return []
    
    if debug:
        print(f"Finding max growth rates in {field} of {n} highest total power wavenumbers....")

    # Apply max which returns highest values along axis
    max_powers = tkSpectrum.sum(dim='time')
    max_powers = np.nan_to_num(max_powers)

    # Find indices of highest peark powers
    max_power_k_indices = np.argpartition(max_powers, -n)[-n:][::-1]

    gammas = find_max_growth_rates(tkSpectrum, max_power_k_indices, gammaWindow, displayPlots)

    n = 0
    # Debugging
    if displayPlots or savePlots:
        for i in range(numPlotsToSaveDisplay):
            signal = tkSpectrum[:,max_power_k_indices[i]]
            max_g = gammas[i]
            timeVals = signal.coords['time'][max_g.timeStartIndex:max_g.timeEndIndex]
            plt.close("all")
            fig, ax = plt.subplots(figsize=(12, 8))
            np.log(signal).plot(ax=ax)
            ax.plot(timeVals, max_g.gamma * timeVals + max_g.yIntercept, label = r"$\gamma = $" + f"{max_g.gamma:.3f}" + r"$\omega_{ci}$")
            ax.set_xlabel(r"Time [$\tau_{ci}$]")
            ax.set_ylabel(f"Log of {field} signal power")
            ax.grid()
            ax.legend()
            plt.title(f'k = {float(max_g.wavenumber):.3f}')
            if savePlots:
                plt.savefig(saveFolder / Path(f"{i}_{runName}_{field}_max_total_power_k_{max_g.wavenumber:.3f}.png"))
            if displayPlots:
                plt.show()
            plt.clf()

    return gammas

def find_max_growth_rates_of_top_n_k_with_max_peak_power(
        tkSpectrum : xr.DataArray,
        field : str,
        runName : str,
        gammaWindow : int = 100,
        n : int = 10,
        savePlots : bool = False,
        displayPlots : bool = False,
        numPlotsToSaveDisplay : int = 0,
        saveFolder : Path = None
) -> List[utils.LinearGrowthRate]:
    
    # Short-circuit
    if n < 1:
        return []
    
    if debug:
        print(f"Finding max growth rates in {field} of {n} highest peak power wavenumbers....")

    # Apply max which returns highest values along axis
    peak_powers = tkSpectrum.max(dim='time')
    peak_powers = np.nan_to_num(peak_powers)

    # Find indices of highest peark powers
    peak_power_k_indices = np.argpartition(peak_powers, -n)[-n:][::-1]

    gammas = find_max_growth_rates(tkSpectrum, peak_power_k_indices, gammaWindow, displayPlots)

    # Debugging
    if displayPlots or savePlots:
        numPlotsToSaveDisplay = np.min([n, numPlotsToSaveDisplay])
        for i in range(numPlotsToSaveDisplay):
            signal = tkSpectrum[:,peak_power_k_indices[i]]
            max_g = gammas[i]
            timeVals = signal.coords['time'][max_g.timeStartIndex:max_g.timeEndIndex]
            plt.close("all")
            fig, ax = plt.subplots(figsize=(12, 8))
            np.log(signal).plot(ax=ax)
            ax.plot(timeVals, max_g.gamma * timeVals + max_g.yIntercept, label = r"$\gamma = $" + f"{max_g.gamma:.3f}" + r"$\omega_{ci}$")
            ax.set_xlabel(r"Time [$\tau_{ci}$]")
            ax.set_ylabel(f"Log of {field} signal power")
            ax.grid()
            ax.legend()
            plt.title(f'k = {float(max_g.wavenumber):.3f}')
            if savePlots:
                plt.savefig(saveFolder / Path(f"{i}_{runName}_{field}_max_peak_power_k_{max_g.wavenumber:.3f}.png"))
            if displayPlots:
                plt.show()
            plt.clf()

    return gammas

def create_t_k_plots(
        tkSpectrum : xr.DataArray,
        field : str,
        field_unit : str,
        saveDirectory : Path,
        runName : str,
        maxK : float = None,
        log : bool = False,
        display : bool = False):
    
    print("Generating t-k plot....")

    if log:
        tkSpectrum = np.log(tkSpectrum)
    if maxK is not None:
        tkSpectrum = tkSpectrum.sel(wavenumber=tkSpectrum.wavenumber<=maxK)
        tkSpectrum = tkSpectrum.sel(wavenumber=tkSpectrum.wavenumber>=-maxK)

    # Time-wavenumber
    fig, axs = plt.subplots(figsize=(15, 10))
    tkSpectrum.plot(ax=axs, x = "wavenumber", y = "time", cbar_kwargs={'label': f'Spectral power in {field} [{field_unit}]' if not log else f'Log of spectral power in {field}'}, cmap='plasma')
    axs.grid()
    axs.set_xlabel(r"Wavenumber [$\omega_{ci}/V_A$]")
    axs.set_ylabel(r"Time [$\tau_{ci}$]")
    filename = Path(f'{runName}_{field.replace("_", "")}_tk_log-{log}_maxK-{maxK if maxK is not None else "all"}.png')
    fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
    plt.clf()

def create_t_k_spectrum(
        originalFftSpectrum : xr.DataArray, statsFile : nc.Dataset) -> xr.DataArray :
    
    tk_spec = originalFftSpectrum.where(originalFftSpectrum.frequency>0.0, 0.0)
    original_zero_freq_amplitude = tk_spec.sel(wavenumber=0.0)
    # Double spectrum to conserve E
    tk_spec = np.sqrt(2.0) * tk_spec # <---- Should this be sqrt(2)?
    tk_spec.loc[dict(wavenumber=0.0)] = original_zero_freq_amplitude # Restore original 0-freq amplitude
    tk_spec = xrft.xrft.ifft(tk_spec, dim="frequency")
    tk_spec = tk_spec.rename(freq_frequency="time")
    tk_spec = abs(tk_spec)

    # Log stats on spectrum
    tk_sum = float(tk_spec.sum())
    statsFile.totalTkSpectralPower = tk_sum

    tk_squared = float((np.abs(tk_spec)**2).sum())
    parseval_tk = tk_squared  * tk_spec.coords['wavenumber'].spacing * tk_spec.coords['time'].spacing
    statsFile.parsevalTk = parseval_tk
    
    tk_peak = float(np.nanmax(tk_spec))
    statsFile.peakTkSpectralPower = tk_peak
    
    tk_mean = float(tk_spec.mean())
    statsFile.meanTkSpectralPower = tk_mean
    
    if debug:
        print(f"Sum of t-k squared * dk * dt: {parseval_tk}")
        print(f"Max peak in t-k: {tk_peak}")
        print(f"Mean of t-k: {tk_mean}")

    return tk_spec

def create_omega_k_plots(
        fftSpectrum : xr.DataArray,
        statsFile : nc.Dataset,
        field : str,
        field_unit : str,
        saveDirectory : Path,
        runName : str,
        inputDeck : dict,
        bkgdSpecies : str = 'p+',
        fastSpecies : str = 'p+',
        maxK : float = None,
        maxW : float = None,
        log : bool = False,
        display : bool = False):

    print("Generating w-k plots....")

    spec = abs(fftSpectrum)

    # Select positive temporal frequencies
    spec = spec.sel(frequency=spec.frequency>=0.0)

    # Trim to max wavenumber and frequency, if specified
    if maxK is not None:
        spec = spec.sel(wavenumber=spec.wavenumber<=maxK)
        spec = spec.sel(wavenumber=spec.wavenumber>=-maxK)
    if maxW is not None:
        spec = spec.sel(frequency=spec.frequency<=maxW)

    # Log stats on spectrum
    spec_sum = float(spec.sum())
    statsFile.totalWkSpectralPower = spec_sum 
    squared_sum = float((np.abs(spec)**2).sum())
    parseval_wk = squared_sum * spec.coords['frequency'].spacing * spec.coords['wavenumber'].spacing * 2.0 # Double because half the energy is outside the range we care about 
    statsFile.parsevalWk = parseval_wk
    spec_peak = float(np.nanmax(spec))
    statsFile.peakWkSpectralPower = spec_peak
    spec_mean = float(spec.mean())
    statsFile.meanWkSpectralPower = spec_mean
    
    if debug:
        print(f"Sum of omega-k sqared * dw * dk: {parseval_wk}")
        print(f"Max peak in omega-k: {spec_peak}")
        print(f"Mean of omega-k: {spec_mean}")

    # Power in omega over all k
    fig, axs = plt.subplots(figsize=(15, 10))
    power_trace = spec.sum(dim = "wavenumber")
    power_trace.plot(ax=axs)
    axs.set_xticks(ticks=np.arange(np.floor(power_trace.coords['frequency'][0]), np.ceil(power_trace.coords['frequency'][-1])+1.0, 1.0), minor=True)
    axs.grid(which='both', axis='x')
    axs.set_xlabel(r"Frequency [$\omega_{ci}$]")
    axs.set_ylabel(f"Sum of power in {field} over all k [{field_unit}]")
    filename = Path(f'{runName}_{field.replace("_", "")}_powerByOmega_log-{log}_maxK-{maxK if maxK is not None else "all"}_maxW-{maxW if maxW is not None else "all"}.png')
    fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
        plt.clf()
        plt.close("all")

    # Power in k over all omega
    fig, axs = plt.subplots(figsize=(15, 10))
    power_trace = spec.sum(dim = "frequency")
    power_trace.plot(ax=axs)
    axs.set_xticks(ticks=np.arange(np.floor(power_trace.coords['wavenumber'][0]), np.ceil(power_trace.coords['wavenumber'][-1])+1.0, 1.0), minor=True)
    axs.grid(which='both', axis='x')
    axs.set_xlabel(r"Wavenumber [$\omega_{ci}/V_A$]")
    omega = r'$\omega_{ci}$'
    axs.set_ylabel(f"Sum of power in {field} over all {omega} [{field_unit}]")
    filename = Path(f'{runName}_{field.replace("_", "")}_powerByK_log-{log}_maxK-{maxK if maxK is not None else "all"}_maxW-{maxW if maxW is not None else "all"}.png')
    fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
        plt.clf()
        plt.close("all")

    if log:
        spec = np.log(spec)

    # Full dispersion relation for positive omega
    fig, axs = plt.subplots(figsize=(15, 10))
    spec.plot(ax=axs, cbar_kwargs={'label': f'Spectral power in {field} [{field_unit}]' if not log else f'Log of spectral power in {field}'}, cmap='plasma')
    axs.set_ylabel(r"Frequency [$\omega_{ci}$]")
    axs.set_xlabel(r"Wavenumber [$\omega_{ci}/V_A$]")
    filename = Path(f'{runName}_{field.replace("_", "")}_wk_log-{log}_maxK-{maxK if maxK is not None else "all"}_maxW-{maxW if maxW is not None else "all"}.png')
    fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
        plt.clf()
        plt.close("all")

    # Positive omega/positive k with vA and lower hybrid frequency
    fig, axs = plt.subplots(figsize=(15, 10))
    spec = spec.sel(wavenumber=spec.wavenumber>0.0)
    spec.plot(ax=axs, cbar_kwargs={'label': f'Spectral power in {field} [{field_unit}]' if not log else f'Log of spectral power in {field}'}, cmap='plasma')
    axs.plot(spec.coords['wavenumber'].data, spec.coords['wavenumber'].data, 'k--', label=r'$V_A$ branch')
    B0 = inputDeck['constant']['b0_strength']
    bkgd_number_density = float(inputDeck['constant']['background_density'])
    wLH_cyclo = ppf.lower_hybrid_frequency(B0 * u.T, bkgd_number_density * u.m**-3, bkgdSpecies) / ppf.gyrofrequency(B0 * u.T, fastSpecies)
    axs.axhline(y = wLH_cyclo, color='black', linestyle=':', label=r'Lower hybrid frequency')
    axs.legend(loc='upper right')
    axs.set_ylabel(r"Frequency [$\omega_{ci}$]")
    axs.set_xlabel(r"Wavenumber [$\omega_{ci}/V_A$]")
    filename = Path(f'{runName}_{field.replace("_", "")}_wk_positiveK_log-{log}_maxK-{maxK if maxK is not None else "all"}_maxW-{maxW if maxW is not None else "all"}.png')
    fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
        plt.clf()
        plt.close("all")

def calculate_simulation_metadata(
        inputDeck : dict,
        dataset,
        outputNcRoot : nc.Dataset,
        beam = True,
        fastSpecies : str = 'p+',
        bkgdSpecies : str = 'p+') -> tuple[float, float]:
    
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
        print(f"Sim length in Debye lengths: {sim_L_dl}")

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

def process_simulation_batch(
        directory : Path,
        fields : list = ['Magnetic_Field_Bz'],
        outFileDirectory : Path = None,
        maxK : float = None,
        maxW : float = None,
        numGrowthRates : int = 10,
        maxResPct : float = 0.1,
        maxResidual : float = 0.1, # IMPLEMENT THIS ONCE BASELINED
        gammaWindowPct : float = 10.0,
        minSignalPower : float = 0.08, # IMPLEMENT THIS ONCE BASELINED
        takeLog = False, 
        deltaField = False, 
        beam = True,
        fastSpecies : str = 'p+',
        bkgdSpecies : str = 'p+',
        plotTitleSize : float = 18.0,
        plotLabelSize : float = 16.0,
        plotTickSize : float = 14.0,
        createPlots = False,
        displayPlots = False,
        saveGrowthRatePlots = False,
        outputType : str = "csv"):
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

    metadataFolder, growthDataFolder, plotsFolder = initialise_folder_structure(directory, outFileDirectory)

    run_folders = []
    if directory.name.startswith("run_"): # Single simulation
        run_folders.append(directory)
    else: # Multiple simulations
        run_folders = glob.glob(str(directory / "run_*") + os.path.sep) 

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
        statsFilepath = os.path.join(metadataFolder, statsFilename)
        statsRoot = nc.Dataset(statsFilepath, "w", format="NETCDF4")

        ion_gyroperiod, alfven_velocity = calculate_simulation_metadata(inputDeck, ds, statsRoot, beam, fastSpecies, bkgdSpecies)

        ds = normalise_data(ds, ion_gyroperiod, alfven_velocity)

        if "all" in fields:
            fields = [str(f) for f in ds.data_vars.keys() if str(f).startswith("Electric_Field") or str(f).startswith("Magnetic_Field")]
        
        for field in fields:

            print(f"Analyzing field '{field}'...")
            plotFieldFolder = Path(os.path.join(plotsFolder, field))
            if not os.path.exists(plotFieldFolder):
                os.mkdir(plotFieldFolder)
            dataFieldFolder = Path(os.path.join(growthDataFolder, field))
            if not os.path.exists(dataFieldFolder):
                os.mkdir(dataFieldFolder)

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

            # Take FFT
            original_spec : xr.DataArray = xrft.xrft.fft(ds[field], true_amplitude=True, true_phase=True, window=None)
            original_spec = original_spec.rename(freq_time="frequency", freq_x_space="wavenumber")
            # Remove zero-frequency component
            original_spec = original_spec.where(original_spec.wavenumber!=0.0, None)

            tk_spec = create_t_k_spectrum(original_spec, fieldStats)

            # Dispersion relations
            if createPlots:

                plt.rcParams.update({'axes.labelsize': plotLabelSize})
                plt.rcParams.update({'axes.titlesize': plotTitleSize})
                plt.rcParams.update({'xtick.labelsize': plotTickSize})
                plt.rcParams.update({'ytick.labelsize': plotTickSize})

                create_omega_k_plots(original_spec, fieldStats, field, field_unit, plotFieldFolder, simFolder.name, inputDeck, maxK=maxK, maxW=maxW, log=takeLog, display=displayPlots)

                create_t_k_plots(tk_spec, field, field_unit, plotFieldFolder, simFolder.name, maxK, takeLog, displayPlots)

            # Linear growth rates
            gammaWindowIndices = int((gammaWindowPct / 100.0) * tk_spec.coords['time'].size)
            peakPowerGammaFolder = dataFieldFolder / "peak_power_Ks"
            if not os.path.exists(peakPowerGammaFolder):
                os.mkdir(peakPowerGammaFolder)
            totalPowerGammaFolder = dataFieldFolder / "max_total_power_Ks"
            if not os.path.exists(totalPowerGammaFolder):
                os.mkdir(totalPowerGammaFolder)
            max_peak_gammas = find_max_growth_rates_of_top_n_k_with_max_peak_power(tk_spec, field, simFolder.name, gammaWindowIndices, numGrowthRates, savePlots=saveGrowthRatePlots, displayPlots=displayPlots, numPlotsToSaveDisplay=numGrowthRates, saveFolder=peakPowerGammaFolder)
            max_total_gammas = find_max_growth_rates_of_top_n_k_with_max_total_power(tk_spec, field, simFolder.name, gammaWindowIndices, numGrowthRates, savePlots=saveGrowthRatePlots, displayPlots=displayPlots, numPlotsToSaveDisplay=numGrowthRates, saveFolder=totalPowerGammaFolder)

            if outputType == "netcdf":
                statsRoot = create_netCDF_data_structure(statsRoot)
                # Write data here once netCDF output implemented
            elif outputType == "csv":
                dataFilename = simFolder.name + "_data.csv"
                dataFilepath = Path(os.path.join(peakPowerGammaFolder, dataFilename))
                with open(str(dataFilepath.absolute()), mode="w") as csvOut:
                    writer = csv.writer(csvOut)
                    writer.writerow(["rank", "wavenumber", "peakPower", "totalPower", "time", "maxGamma", "residual", "fitYintercept"])

                    for i in range(0, len(max_peak_gammas)):
                        gamma = max_peak_gammas[i]
                        writer.writerow([i, gamma.wavenumber, gamma.peakPower, gamma.totalPower, gamma.timeMidpoint, gamma.gamma, gamma.residual, gamma.yIntercept])

                dataFilepath = Path(os.path.join(totalPowerGammaFolder, dataFilename))
                with open(str(dataFilepath.absolute()), mode="w") as csvOut:
                    writer = csv.writer(csvOut)
                    writer.writerow(["rank", "wavenumber", "peakPower", "totalPower", "time", "maxGamma", "residual", "fitYintercept"])

                    for i in range(0, len(max_total_gammas)):
                        gamma = max_total_gammas[i]
                        writer.writerow([i, float(gamma.wavenumber), gamma.peakPower, gamma.totalPower, float(gamma.timeMidpoint), gamma.gamma, gamma.residual, gamma.yIntercept])

        statsRoot.close()
            
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
        required = True,
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
        "--numGrowthRates",
        action="store",
        help="Number of wavenumbers for which to calculate growth rates.",
        required = True,
        type=int
    )
    parser.add_argument(
        "--takeLog",
        action="store_true",
        help="Take logarithm of field data for plotting.",
        required = False
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
        "--saveGammaPlots",
        action="store_true",
        help="Save max growth rate plots to file.",
        required = False
    )
    parser.add_argument(
        "--outputType",
        action="store",
        help="Output data type for growth rates (csv or netcdf)",
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

    process_simulation_batch(
        directory=args.dir, 
        outFileDirectory=args.outputDir,
        fields=args.fields,
        maxK=args.maxK,
        maxW=args.maxW,
        numGrowthRates=args.numGrowthRates, 
        takeLog=args.takeLog, 
        createPlots=args.createPlots, 
        displayPlots=args.displayPlots,
        saveGrowthRatePlots=args.saveGammaPlots,
        outputType=args.outputType)