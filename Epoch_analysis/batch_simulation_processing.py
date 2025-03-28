import argparse
import os
from pathlib import Path
from matplotlib import pyplot as plt
from sdf_xarray import SDFPreprocess
from scipy import constants, stats
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
        dataDirectory : Path,
        fields : list,
        create : bool = False,
        energy : bool = True,
        plotGrowthRates : bool = False,
        outFileDirectory : Path = None
) -> tuple[Path, Path, Path]:
    
    if outFileDirectory is None:
        outFileDirectory = dataDirectory / "analysis"

    if debug:
        print(f"Creating folder structure in {outFileDirectory}....")
            
    dataFolder = outFileDirectory / "data"
    plotsFolder = outFileDirectory / "plots"
                
    if create:
        if os.path.exists(outFileDirectory):
            sh.rmtree(outFileDirectory)
        os.mkdir(outFileDirectory)
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

def create_netCDF_fieldVariable_structure(
        fieldRoot : nc.Dataset,
        numGrowthRates : int
) -> nc.Dataset:
    growth_rate_group = fieldRoot.createGroup("growthRates")
    posGrowthRateGrp = growth_rate_group.createGroup("positive")
    negGrowthRateGrp = growth_rate_group.createGroup("negative")
    groups = [posGrowthRateGrp, negGrowthRateGrp]
    
    for group in groups:
        group.createDimension("wavenumber", numGrowthRates)

        k_var = group.createVariable("wavenumber", datatype="f4", dimensions=("wavenumber",))
        k_var.units = "wCI/vA"
        group.createVariable("peakPower", datatype="f4", dimensions=("wavenumber",))
        group.createVariable("totalPower", datatype="f4", dimensions=("wavenumber",))
        t_var = group.createVariable("time", datatype="f4", dimensions=("wavenumber",))
        t_var.units = "tCI"
        gamma_var = group.createVariable("growthRate", datatype="f8", dimensions=("wavenumber",))
        gamma_var.units = "wCI"
        gamma_var.standard_name = "linear_growth_rate"
        group.createVariable("rSquared", datatype="f4", dimensions=("wavenumber",))
        group.createVariable("yIntercept", datatype="f4", dimensions=("wavenumber",))

    return growth_rate_group
    
def plot_growth_rates(
        tkSpectrum : xr.DataArray,
        field : str,
        growthRateData : list[e_utils.LinearGrowthRate],
        numToPlot : int,
        selectionMetric : str,
        save : bool = False,
        display : bool = False,
        noTitle : bool = False,
        saveFolder : Path = None,
        runName : str = None
):
    # Short-circuit
    if numToPlot == 0:
        return
    
    signString = "positive" if growthRateData[0].gamma > 0.0 else "negative"
    
    if debug:
        print(f"Plotting best {signString} growth rates in {field} of {numToPlot} {selectionMetric} power wavenumbers....")
    
    if selectionMetric == "peak":
        # Find n highest peak or total powers
        growth_rates_to_plot = sorted(growthRateData, key=lambda gamma: gamma.peakPower, reverse=True)[:numToPlot]
    elif selectionMetric == "total":
        growth_rates_to_plot = sorted(growthRateData, key=lambda gamma: gamma.totalPower, reverse=True)[:numToPlot]
    else:
        raise NotImplementedError("Only \'peak\' or \'total\' power selection criteria implemented.")
    
    rank = 0
    for g in growth_rates_to_plot:
        signal = tkSpectrum.sel(wavenumber=g.wavenumber)
        timeVals = signal.coords['time'][g.timeStartIndex:g.timeEndIndex]
        plt.close("all")
        fig, ax = plt.subplots(figsize=(12, 8))
        np.log(signal).plot(ax=ax)
        ax.plot(timeVals, g.gamma * timeVals + g.yIntercept, label = r"$\gamma = $" + f"{g.gamma:.3f}" + r"$\omega_{ci}$")
        ax.set_xlabel(r"Time [$\tau_{ci}$]")
        ax.set_ylabel(f"Log of {field} signal power")
        ax.grid()
        ax.legend()
        if not noTitle:
            plt.title(f'{runName}_{field}_{signString}_growth_k_{g.wavenumber:.3f}_{selectionMetric}_power_rank_{rank}')
        if save:
            plt.savefig(saveFolder / Path(f"{runName}_{field}_{signString}_growth_k_{g.wavenumber:.3f}_{selectionMetric}Power_rank_{rank}.png"))
        if display:
            plt.show()
        rank += 1
        plt.close('all')

def find_max_growth_rates(
        tkSpectrum : xr.DataArray,
        gammaWindowMin : int,
        gammaWindowMax : int,
        skipIndices : int = 1
):

    print("Finding max growth rates....")

    best_pos_growth_rates = []
    best_neg_growth_rates = []

    num_wavenumbers = tkSpectrum.sizes["wavenumber"]

    for index in range(0, num_wavenumbers):

        # Must load data here for quick iteration over windows
        # Changed: this is pre-loaded in create_t_k_spectrum()
        signal = tkSpectrum.isel(wavenumber=index)

        signalK=float(tkSpectrum.coords['wavenumber'][index])
        signalPeak=float(signal.max())
        signalTotal=float(signal.sum())

        windowWidths = range(gammaWindowMin, gammaWindowMax + 1, skipIndices)
        len_widths = len(windowWidths)

        best_pos_params = None
        best_neg_params = None
        best_pos_r_squared = float('-inf')
        best_neg_r_squared = float('-inf')

        width_count = 0
        for width in windowWidths: # For each possible window width
            
            windowStarts = range(0, len(signal) - (width + 1), skipIndices)

            if debug:
                width_count += 1
                len_windows = len(windowStarts)
                window_count = 0

            for window in windowStarts: # For each window
                
                if debug:
                    window_count += 1
                    print(f"Processing width {width} starting at {window} in k={signalK}. Width {width_count}/{len_widths} window {window_count}/{len_windows}....")

                t_k_window = signal[window:(width + window)]

                slope, intercept, r_value, _, _ = stats.linregress(signal.coords["time"][window:(width + window)], np.log(t_k_window))
                r_squared = r_value ** 2
                
                if not np.isnan(slope):
                    if slope > 0.0:
                        if r_squared > best_pos_r_squared:
                            best_pos_r_squared = r_squared
                            best_pos_params = (slope, intercept, width, window, r_squared)
                    else:
                        if r_squared > best_neg_r_squared:
                            best_neg_r_squared = r_squared
                            best_neg_params = (slope, intercept, width, window, r_squared)

        if best_pos_params is not None:
            gamma, y_int, window_width, windowStart, r_sqrd = best_pos_params
            best_pos_growth_rates.append(
                e_utils.LinearGrowthRate(timeStartIndex=windowStart,
                                    timeEndIndex=(windowStart + window_width),
                                    timeMidpointIndex=windowStart+(int(window_width/2)),
                                    gamma=gamma,
                                    yIntercept=y_int,
                                    rSquared=r_sqrd,
                                    wavenumber=signalK,
                                    timeMidpoint=float(tkSpectrum.coords['time'][windowStart+(int(window_width/2))]),
                                    peakPower = signalPeak,
                                    totalPower = signalTotal))
            
        if best_neg_params is not None:
            gamma, y_int, window_width, windowStart, r_sqrd = best_neg_params
            best_neg_growth_rates.append(
                e_utils.LinearGrowthRate(timeStartIndex=windowStart,
                                    timeEndIndex=(windowStart + window_width),
                                    timeMidpointIndex=windowStart+(int(window_width/2)),
                                    gamma=gamma,
                                    yIntercept=y_int,
                                    rSquared=r_sqrd,
                                    wavenumber=signalK,
                                    timeMidpoint=float(tkSpectrum.coords['time'][windowStart+(int(window_width/2))]),
                                    peakPower = signalPeak,
                                    totalPower = signalTotal))
        del(signal)
        
        if debug: 
            if best_pos_params is not None:
                print(f"Max positive gamma found: {best_pos_growth_rates[-1].to_string()}")
            if best_neg_params is not None:
                print(f"Max negative gamma found: {best_neg_growth_rates[-1].to_string()}")

    return best_pos_growth_rates, best_neg_growth_rates

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
    plt.close('all')

def create_t_k_spectrum(
        originalFftSpectrum : xr.DataArray, 
        statsFile : nc.Dataset,
        maxK : float = None,
        load : bool = True
) -> xr.DataArray :
    
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

    statsFile.peakTkSpectralPowerRatio = tk_peak/tk_mean
    
    if debug:
        print(f"Sum of t-k squared * dk * dt: {parseval_tk}")
        print(f"Max peak in t-k: {tk_peak}")
        print(f"Mean of t-k: {tk_mean}")
        print(f"Ratio of peak to mean in t-k: {tk_peak/tk_mean}")

    tk_spec = tk_spec.sel(wavenumber=tk_spec.wavenumber<=maxK)
    tk_spec = tk_spec.sel(wavenumber=tk_spec.wavenumber>=-maxK)

    if load:
        tk_spec = tk_spec.load()

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

    spec = abs(fftSpectrum.load())

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
        plt.close("all")

    # Positive omega/positive k with vA and lower hybrid frequency
    fig, axs = plt.subplots(figsize=(15, 10))
    spec = spec.sel(wavenumber=spec.wavenumber>0.0)
    spec.plot(ax=axs, cbar_kwargs={'label': f'Spectral power in {field} [{field_unit}]' if not log else f'Log of spectral power in {field}'}, cmap='plasma')
    axs.plot(spec.coords['wavenumber'].data, spec.coords['wavenumber'].data, 'w--', label=r'$V_A$ branch')
    B0 = inputDeck['constant']['b0_strength']
    bkgd_number_density = float(inputDeck['constant']['background_density'])
    wLH_cyclo = ppf.lower_hybrid_frequency(B0 * u.T, bkgd_number_density * u.m**-3, bkgdSpecies) / ppf.gyrofrequency(B0 * u.T, fastSpecies)
    axs.axhline(y = wLH_cyclo, color='gray', linestyle=':', label=r'Lower hybrid frequency')
    axs.legend(loc='upper left')
    axs.set_ylabel(r"Frequency [$\omega_{ci}$]")
    axs.set_xlabel(r"Wavenumber [$\omega_{ci}/V_A$]")
    filename = Path(f'{runName}_{field.replace("_", "")}_wk_positiveK_log-{log}_maxK-{maxK if maxK is not None else "all"}_maxW-{maxW if maxW is not None else "all"}.png')
    fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
        plt.close("all")
    
    del(spec)

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

def run_energy_analysis(
    dataset : xr.Dataset,
    inputDeck : dict,
    simName : str,
    savePlotsFolder : Path,
    statsFile : nc.Dataset,
    displayPlots : bool = False,
    noTitle : bool = False,
    beam : bool = True,
    percentage : bool = True
):
    # Create stats group
    energyStats = statsFile.createGroup("Energy")
    energyStats.createDimension("time", dataset.coords["time"].size)
    time_var = energyStats.createVariable("time", "f8", ("time",))
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
    ped = energyStats.createVariable("protonMeanEnergyDensity", "f8", ("time",))
    ped[:] = protonKEdensity_mean

    electron_KE : xr.DataArray = dataset['Derived_Average_Particle_Energy_electron'].load()
    electronKE_mean = electron_KE.mean(dim = "x_space").data # J
    electronKEdensity_mean = electronKE_mean * electron_density # J / m^3
    del(electron_KE)
    del(electronKE_mean)
    eed = energyStats.createVariable("electronMeanEnergyDensity", "f8", ("time",))
    eed[:] = electronKEdensity_mean

    Ex : xr.DataArray = dataset['Electric_Field_Ex'].load()
    Ey : xr.DataArray = dataset['Electric_Field_Ey'].load()
    Ez : xr.DataArray = dataset['Electric_Field_Ez'].load()
    electricFieldStrength = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    del(Ex, Ey, Ez)
    electricFieldEnergyDensity : xr.DataArray = (constants.epsilon_0 * electricFieldStrength**2) / 2.0 # J / m^3
    electricFieldDensity_mean = electricFieldEnergyDensity.mean(dim="x_space").data # J / m^3
    del(electricFieldEnergyDensity)
    efd = energyStats.createVariable("electricFieldMeanEnergyDensity", "f8", ("time",))
    efd[:] = electricFieldDensity_mean

    Bx : xr.DataArray = dataset['Magnetic_Field_Bx'].load()
    By : xr.DataArray = dataset['Magnetic_Field_By'].load()
    Bz : xr.DataArray = dataset['Magnetic_Field_Bz'].load()
    magneticFieldStrength = np.sqrt(Bx**2 + By**2 + Bz**2)
    del(Bx, By, Bz)
    magneticFieldEnergyDensity : xr.DataArray = (magneticFieldStrength**2 / (2.0 * constants.mu_0)) # J / m^3
    magneticFieldEnergyDensity_mean = magneticFieldEnergyDensity.mean(dim = "x_space").data # J / m^3
    del(magneticFieldEnergyDensity)
    mfd = energyStats.createVariable("magneticFieldMeanEnergyDensity", "f8", ("time",))
    mfd[:] = magneticFieldEnergyDensity_mean

    # Calculate B and E energy and convert others to to J/m3
    deltaMeanMagneticEnergyDensity = magneticFieldEnergyDensity_mean - magneticFieldEnergyDensity_mean[0] # J / m^3
    deltaMeanElectricEnergyDensity = electricFieldDensity_mean - electricFieldDensity_mean[0] # J / m^3
    deltaProtonKE_density = protonKEdensity_mean - protonKEdensity_mean[0] # J / m^3
    deltaElectronKE_density = electronKEdensity_mean - electronKEdensity_mean[0] # J / m^3

    totalDeltaMeanEnergyDensity = deltaProtonKE_density + deltaElectronKE_density + deltaMeanMagneticEnergyDensity + deltaMeanElectricEnergyDensity
    timeCoords = dataset.coords['time']

    # Initialise plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    filename = Path(f"{simName}_absolute_energy_change.png")

    # Write stats
    # Start, end and peak energy densities, and times of maximum and minimum
    if beam:

        fastIonDensity = bkgd_density * frac_beam # m^-3
        fastIonKE : xr.DataArray = dataset['Derived_Average_Particle_Energy_ion_ring_beam'].load()
        fastIonKE_mean = fastIonKE.mean(dim = "x_space").data # J
        fastIonKEdensity_mean = fastIonKE_mean * fastIonDensity # J / m^3
        del(fastIonKE)
        del(fastIonKE_mean)
        fed = energyStats.createVariable("fastIonMeanEnergyDensity", "f8", ("time",))
        fed[:] = fastIonKEdensity_mean
        deltaFastIonKE_density = fastIonKEdensity_mean - fastIonKEdensity_mean[0] # J / m^3
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

        ax.plot(timeCoords, deltaFastIonKE_density, label = r"ion ring beam KE", color="red")

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

    # Main plotting
    ax.plot(timeCoords, deltaProtonKE_density, label = r"Bkgd Proton KE", color="orange")
    ax.plot(timeCoords, deltaElectronKE_density, label = r"Bkgd Electron KE", color="blue")
    ax.plot(timeCoords, deltaMeanMagneticEnergyDensity, label = r"B-field E", color="purple", linestyle="--")
    ax.plot(timeCoords, deltaMeanElectricEnergyDensity, label = r"E-field E", color="green", linestyle="--")
    ax.plot(timeCoords, totalDeltaMeanEnergyDensity, label = r"Total E", color="black")
    ax.set_xlabel(r'Time [$\tau_{ci}$]')
    ax.set_ylabel(r"Change in energy density [$J/m^3$]")
    if not noTitle: 
        ax.set_title(f"{simName}: Evolution of absolute energy in particles and EM fields", wrap=True)
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(savePlotsFolder / filename)
    if displayPlots:
        plt.show()
    plt.close("all")

    if percentage:

        totalMeanEnergyDensity_0 = (
            magneticFieldEnergyDensity_mean[0] 
            + electricFieldDensity_mean[0] 
            + protonKEdensity_mean[0]
            + electronKEdensity_mean[0]
        )

        fig, ax = plt.subplots(figsize=(12, 8))
        filename = Path(f"{simName}_percentage_energy_change.png")

        if beam:
            fastIon_baseline = fastIonKEdensity_mean[0]
            deltaFastIonKEdensity_pct = 100.0 * (deltaFastIonKE_density / fastIon_baseline) # %
            totalMeanEnergyDensity_0 += fastIon_baseline
            ax.plot(timeCoords, deltaFastIonKEdensity_pct, label = "Fast Ion KE", color='red')

            baseline_E = fastIon_baseline
        else:
            baseline_E = totalMeanEnergyDensity_0

        deltaMeanMagneticEnergyDensity_pct = 100.0 * (deltaMeanMagneticEnergyDensity / baseline_E) # %
        deltaMeanElectricEnergyDensity_pct = 100.0 * (deltaMeanElectricEnergyDensity / baseline_E) # %
        deltaProtonKEdensity_pct = 100.0 * (deltaProtonKE_density / baseline_E) # %
        deltaElectronKEdensity_pct = 100.0 * (deltaElectronKE_density / baseline_E) # %

        totalDeltaMeanEnergyDensity_pct = deltaMeanMagneticEnergyDensity_pct + deltaMeanElectricEnergyDensity_pct + deltaProtonKEdensity_pct + deltaElectronKEdensity_pct
        
        if beam:
            totalDeltaMeanEnergyDensity_pct += deltaFastIonKEdensity_pct

        ax.plot(timeCoords, deltaProtonKEdensity_pct, label = "Bkgd Proton KE", color="orange")
        ax.plot(timeCoords, deltaElectronKEdensity_pct, label = "Bkgd Electron KE", color="blue")
        ax.plot(timeCoords, deltaMeanMagneticEnergyDensity_pct, label = "B-field E", color="purple", linestyle="--")
        ax.plot(timeCoords, deltaMeanElectricEnergyDensity_pct, label = "E-field E", color="green", linestyle="--")
        ax.plot(timeCoords, totalDeltaMeanEnergyDensity_pct, label = "Total E", color="black")
        #ax.set_yscale('symlog')
        ax.set_xlabel(r'Time [$\tau_{ci}$]')
        ax.set_ylabel("Change in energy density [%]")
        if not noTitle:
            ax.set_title(f"{simName}: Evolution of energy change in particles and EM fields, "
                        + f"as a percentage of original {'fast ion' if beam else 'total'} energy", wrap=True)
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
        fields : list = ['Magnetic_Field_Bz'],
        maxK : float = None,
        maxW : float = None,
        growthRates : bool = True,
        maxResPct : float = 0.1,
        maxResidual : float = 0.1, # IMPLEMENT THIS ONCE BASELINED
        gammaWindowPctMin : float = 5.0,
        gammaWindowPctMax : float = 15.0,
        gammaWindowSkipIndices : float = 5,
        minSignalPower : float = 0.08, # IMPLEMENT THIS ONCE BASELINED
        takeLog = False, 
        beam = True,
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
        statsRoot = nc.Dataset(statsFilepath, "w", format="NETCDF4")

        ion_gyroperiod, alfven_velocity = calculate_simulation_metadata(inputDeck, ds, statsRoot, beam, fastSpecies, bkgdSpecies)

        ds = normalise_data(ds, ion_gyroperiod, alfven_velocity)

        # Energy analysis
        if energy:
            energyPlotFolder = plotsFolder / "energy"
            run_energy_analysis(ds, inputDeck, simFolder.name, energyPlotFolder, statsRoot, displayPlots = displayPlots, noTitle=noTitle, beam = beam, percentage = True)

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

            # Take FFT
            original_spec : xr.DataArray = xrft.xrft.fft(ds[field], true_amplitude=True, true_phase=True, window=None)
            original_spec = original_spec.rename(freq_time="frequency", freq_x_space="wavenumber")
            # Remove zero-frequency component
            original_spec = original_spec.where(original_spec.wavenumber!=0.0, None)

            tk_spec = create_t_k_spectrum(original_spec, fieldStats, maxK, load=True)

            # Dispersion relations
            if createPlots:
                create_omega_k_plots(original_spec, fieldStats, field, field_unit, plotFieldFolder, simFolder.name, inputDeck, maxK=maxK, maxW=maxW, log=takeLog, display=displayPlots)
                create_t_k_plots(tk_spec, field, field_unit, plotFieldFolder, simFolder.name, maxK, takeLog, displayPlots)

            # Linear growth rates
            if growthRates:
                gammaWindowIndicesMin = int((gammaWindowPctMin / 100.0) * tk_spec.coords['time'].size)
                gammaWindowIndicesMax = int((gammaWindowPctMax / 100.0) * tk_spec.coords['time'].size)
                max_pos_gammas, max_neg_gammas = find_max_growth_rates(tk_spec, gammaWindowIndicesMin, gammaWindowIndicesMax)
                
                if saveGrowthRatePlots:
                    gammaPlotFolder = plotFieldFolder / "growth_rates"
                    plot_growth_rates(tk_spec, field, max_pos_gammas, numGrowthRatesToPlot, "peak", saveGrowthRatePlots, displayPlots, noTitle, gammaPlotFolder, simFolder.name)
                    gammaPlotFolder = plotFieldFolder / "growth_rates"
                    plot_growth_rates(tk_spec, field, max_neg_gammas, numGrowthRatesToPlot, "peak", saveGrowthRatePlots, displayPlots, noTitle, gammaPlotFolder, simFolder.name)

                maxNumGammas = np.max([len(max_pos_gammas), len(max_neg_gammas)])
                gammaNc : nc.Dataset = create_netCDF_fieldVariable_structure(fieldStats, maxNumGammas)
                for i in range(len(max_pos_gammas)):
                    posGammaNc = gammaNc.groups["positive"]
                    gamma = max_pos_gammas[i]
                    posGammaNc.variables["wavenumber"][i] = gamma.wavenumber
                    posGammaNc.variables["peakPower"][i] = gamma.peakPower
                    posGammaNc.variables["totalPower"][i] = gamma.totalPower
                    posGammaNc.variables["time"][i] = gamma.timeMidpoint
                    posGammaNc.variables["growthRate"][i] = gamma.gamma
                    posGammaNc.variables["rSquared"][i] = gamma.r_squared
                    posGammaNc.variables["yIntercept"][i] = gamma.yIntercept
                for i in range(len(max_neg_gammas)):
                    negGammaNc = gammaNc.groups["negative"]
                    gamma = max_neg_gammas[i]
                    negGammaNc.variables["wavenumber"][i] = gamma.wavenumber
                    negGammaNc.variables["peakPower"][i] = gamma.peakPower
                    negGammaNc.variables["totalPower"][i] = gamma.totalPower
                    negGammaNc.variables["time"][i] = gamma.timeMidpoint
                    negGammaNc.variables["growthRate"][i] = gamma.gamma
                    negGammaNc.variables["rSquared"][i] = gamma.r_squared
                    negGammaNc.variables["yIntercept"][i] = gamma.yIntercept

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
        "--growthRates",
        action="store_true",
        help="Calculate growth rates.",
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
        "--skipIndices",
        action="store",
        help="Number of indices to skip at a time when fitting gamma windows, e.g. every n window lengths will be fit starting at every n indices between max and min. Required to make analysis tractable.",
        required = False,
        type=int
    )
    parser.add_argument(
        "--runNumber",
        action="store",
        help="Run number to analyse (folder must be in directory and named \'run_##\' where ## is runNumber).",
        required = False,
        type=int
    )
    parser.add_argument(
        "--createFolders",
        action="store_true",
        help="Initialise folder structure.",
        required = False
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help="Process simulation(s).",
        required = False
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
        "--beam",
        action="store_true",
        help="Account for a ring beam of fast ions.",
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

    dataFolder, plotsFolder = initialise_folder_structure(args.dir, args.fields, args.createFolders, args.energy, args.saveGammaPlots, args.outputDir)

    if args.process:
        
        if args.runNumber is not None:
            args.dir = Path(os.path.join(args.dir, f"run_{args.runNumber}"))

        process_simulation_batch(
            directory=args.dir, 
            dataFolder=dataFolder,
            plotsFolder=plotsFolder,
            fields=args.fields,
            maxK=args.maxK,
            maxW=args.maxW,
            growthRates=args.growthRates,
            gammaWindowPctMin=args.minGammaFitWindow,
            gammaWindowPctMax=args.maxGammaFitWindow,
            
            numGrowthRatesToPlot=args.numGrowthRatesToPlot, 
            takeLog=args.takeLog, 
            beam = args.beam,
            createPlots=args.createPlots, 
            displayPlots=args.displayPlots,
            bigLabels=args.bigLabels,
            noTitle=args.noTitle,
            saveGrowthRatePlots=args.saveGammaPlots,
            energy=args.energy)