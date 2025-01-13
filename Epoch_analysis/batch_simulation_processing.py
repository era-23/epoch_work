import argparse
import os
from pathlib import Path
from matplotlib import pyplot as plt
from scipy import constants
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
import numpy.typing as npt
import numpy.polynomial.polynomial as poly
import xrft

def find_max_growth_rates(
        tkSpectrum : xr.DataArray,
        wavenumberIndicesToCalculate : np.ndarray,
        gammaWindow : int
) -> List[utils.LinearGrowthRate] :

    max_growth_rates = []

    for index in wavenumberIndicesToCalculate:

        signal = tkSpectrum.isel(wavenumber=index)

        signal_growth_rates = []

        for w in range(len(signal) - (gammaWindow + 1)): # For each window

            t_k_window = signal[w:(w + gammaWindow)]
            coefs, stats = poly.polyfit(x = signal.coords["time"][w:(w + gammaWindow)], y = np.log(t_k_window), deg = 1, full = True)
            signal_growth_rates.append(
                utils.LinearGrowthRate(wavenumber=tkSpectrum.coords['wavenumber'][index],
                                       timeValues=signal.coords["time"][w:(w + gammaWindow)],
                                       timeMidpoint=signal.coords["time"][w:(w + gammaWindow)][int(gammaWindow/2)],
                                       gamma=coefs[1],
                                       yIntercept=coefs[0],
                                       residual=stats[0]))
            
        max_signal_growth_rate_index = np.nanargmax([lgr.gamma for lgr in signal_growth_rates])
        max_growth_rates.append(signal_growth_rates[max_signal_growth_rate_index])

        # Debugging
        max = signal_growth_rates[max_signal_growth_rate_index]
        np.log(signal).plot()
        plt.plot(max.timeValues, max.gamma * max.timeValues + max.yIntercept)
        plt.title(f'k = {float(max.wavenumber):.3f}')
        plt.show()

    return max_growth_rates

def find_max_growth_rates_of_top_n_k_with_max_total_power(
        tkSpectrum : xr.DataArray,
        gammaWindow : int,
        n : int = 10,
):
    # Apply max which returns highest values along axis
    max_powers = tkSpectrum.sum(dim='time')
    max_powers = np.nan_to_num(max_powers)

    # Find indices of highest peark powers
    max_power_k_indices = np.argpartition(max_powers, -n)[-n:][::-1]

    return find_max_growth_rates(tkSpectrum, max_power_k_indices, gammaWindow)

def find_max_growth_rates_of_top_n_k_with_max_peak_power(
        tkSpectrum : xr.DataArray,
        gammaWindow : int = 100,
        n : int = 10
) -> List[utils.LinearGrowthRate]:
    
    # Apply max which returns highest values along axis
    peak_powers = tkSpectrum.max(dim='time')
    peak_powers = np.nan_to_num(peak_powers)

    # Find indices of highest peark powers
    peak_power_k_indices = np.argpartition(peak_powers, -n)[-n:][::-1]

    return find_max_growth_rates(tkSpectrum, peak_power_k_indices, gammaWindow)
    

def create_t_k_plots(
        tkSpectrum : xr.DataArray,
        field : str,
        field_unit : str,
        saveDirectory : Path,
        runName : str,
        maxK : float = None,
        log : bool = False,
        display : bool = False):
    
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

def create_t_k_spectrum(
        originalFftSpectrum : xr.DataArray) -> xr.DataArray :
    
    tk_spec = originalFftSpectrum.where(originalFftSpectrum.frequency>0.0, 0.0)
    original_zero_freq_amplitude = tk_spec.sel(wavenumber=0.0)
    tk_spec = 2.0 * tk_spec # Double spectrum to conserve E
    tk_spec.loc[dict(wavenumber=0.0)] = original_zero_freq_amplitude # Restore original 0-freq amplitude
    tk_spec = xrft.xrft.ifft(tk_spec, dim="frequency")
    tk_spec = tk_spec.rename(freq_frequency="time")
    tk_spec = abs(tk_spec)

    return tk_spec

def create_omega_k_plots(
        fftSpectrum : xr.DataArray,
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

    #print(fftSpectrum.sum())
    spec = abs(fftSpectrum)
    #print(spec.sum())
    # Select positive temporal frequencies
    spec = spec.sel(frequency=spec.frequency>=0.0)
    #print(spec.sum())
    # Trim to max wavenumber and frequency, if specified
    if maxK is not None:
        spec = spec.sel(wavenumber=spec.wavenumber<=maxK)
        spec = spec.sel(wavenumber=spec.wavenumber>=-maxK)
    if maxW is not None:
        spec = spec.sel(frequency=spec.frequency<=maxW)
    #print(spec.sum())
    # Power in omega over all k
    fig, axs = plt.subplots(figsize=(15, 10))
    power_trace = spec.sum(dim = "wavenumber")
    #print(power_trace.sum())
    power_trace.plot(ax=axs)
    axs.set_xticks(ticks=np.arange(np.floor(power_trace.coords['frequency'][0]), np.ceil(power_trace.coords['frequency'][-1])+1.0, 1.0), minor=True)
    axs.grid(which='both', axis='x')
    axs.set_xlabel(r"Frequency [$\omega_{ci}$]")
    axs.set_ylabel(f"Sum of power in {field} over all k [{field_unit}]")
    filename = Path(f'{runName}_{field.replace("_", "")}_powerByOmega_log-{log}_maxK-{maxK if maxK is not None else "all"}_maxW-{maxW if maxW is not None else "all"}.png')
    fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()

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

    if log:
        spec = np.log(spec)

    # Full dispersion relation for positive omega
    fig, axs = plt.subplots(figsize=(15, 10))
    spec.plot(ax=axs, cbar_kwargs={'label': f'Spectral power in {field} [{field_unit}]' if not log else f'Log of spectral power in {field}'}, cmap='plasma')
    axs.set_ylabel(r"Frequency [$\omega_{ci}$]")
    axs.set_xlabel(r"Wavenumber [$\omega_{ci}/V_A$]")
    print(saveDirectory)
    filename = Path(f'{runName}_{field.replace("_", "")}_wk_log-{log}_maxK-{maxK if maxK is not None else "all"}_maxW-{maxW if maxW is not None else "all"}.png')
    print(str(saveDirectory / filename))
    fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()

    # Positive omega/positive k with vA and lower hybrid frequency
    fig, axs = plt.subplots(figsize=(15, 10))
    spec = spec.sel(wavenumber=spec.wavenumber>0.0)
    spec.plot(ax=axs, cbar_kwargs={'label': f'Spectral power in {field} [{field_unit}]' if not log else f'Log of spectral power in {field}'}, cmap='plasma')
    axs.plot(spec.coords['wavenumber'].data, spec.coords['wavenumber'].data, 'k--', label=r'$V_A$ branch')
    B0 = inputDeck['constant']['b0_strength']
    bkgd_number_density = float(inputDeck['constant']['background_density'])
    wLH_cyclo = ppf.lower_hybrid_frequency(B0 * u.T, bkgd_number_density * u.m**-3, bkgdSpecies) / ppf.gyrofrequency(B0 * u.T, fastSpecies)
    manual_ion_gyro = (constants.elementary_charge * B0) / constants.proton_mass
    manual_electron_gyro = (constants.elementary_charge * B0) / constants.electron_mass
    james_wLH = (1.0 / (np.sqrt(manual_ion_gyro * manual_electron_gyro))) / manual_ion_gyro
    my_w_LH = 1.0 / (manual_ion_gyro*u.rad*u.s**-1 * np.sqrt((1.0/ppf.plasma_frequency(bkgd_number_density * u.m**-3, 'p+')**2) + (1.0/(manual_ion_gyro*u.rad*u.s**-1*manual_electron_gyro*u.rad*u.s**-1))))
    print(f"PPF Lower hybrid: {wLH_cyclo}")
    print(f"James' Lower hybrid: {james_wLH}")
    print(f"My Lower hybrid: {my_w_LH}")
    wUH_cyclo = ppf.upper_hybrid_frequency(B0 * u.T, bkgd_number_density * u.m**-3) / ppf.gyrofrequency(B0 * u.T, fastSpecies)
    print(f"Upper hybrid: {wUH_cyclo}")
    axs.axhline(y = wLH_cyclo, color='black', linestyle=':', label=r'Lower hybrid frequency')
    axs.legend()
    axs.set_ylabel(r"Frequency [$\omega_{ci}$]")
    axs.set_xlabel(r"Wavenumber [$\omega_{ci}/V_A$]")
    filename = Path(f'{runName}_{field.replace("_", "")}_wk_positiveK_log-{log}_maxK-{maxK if maxK is not None else "all"}_maxW-{maxW if maxW is not None else "all"}.png')
    fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()

def calculate_simulation_parameters(
        inputDeck : dict,
        dataset,
        outputNcRoot : nc.Dataset,
        beam = True,
        fastSpecies : str = 'p+',
        bkgdSpecies : str = 'p+') -> tuple[float, float]:
    
    # Check parameters in SI
    num_t = int(inputDeck['constant']['num_time_samples'])
    print(f"Num time points: {num_t}")
    outputNcRoot.numTimePoints = num_t
    sim_time = float(dataset['Magnetic_Field_Bz'].coords["time"][-1]) * u.s
    print(f"Sim time in SI: {sim_time}")
    outputNcRoot.simTime_s = sim_time
    print(f"Sampling frequency: {num_t/sim_time}")
    outputNcRoot.timeSamplingFreq_Hz = num_t/sim_time
    print(f"Nyquist frequency: {num_t/(2.0 *sim_time)}")
    outputNcRoot.timeNyquistFreq_Hz = num_t/(2.0 *sim_time)
    num_cells = int(inputDeck['constant']['num_cells'])
    print(f"Num cells: {num_cells}")
    outputNcRoot.numCells = num_cells
    sim_L = float(dataset['Magnetic_Field_Bz'].coords["X_Grid_mid"][-1]) * u.m
    print(f"Sim length: {sim_L}")
    outputNcRoot.simLength_m = sim_L
    print(f"Sampling frequency: {num_cells/sim_L}")
    outputNcRoot.spaceSamplingFreq_Pm = num_cells/sim_L
    print(f"Nyquist frequency: {num_cells/(2.0 *sim_L)}")
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
    print(f"Debye length: {debye_length}")
    outputNcRoot.debyeLength_m = debye_length.value
    sim_L_dl = sim_L / debye_length
    print(f"Sim length in Debye lengths: {sim_L_dl}")
    outputNcRoot.simLength_dL = sim_L_dl

    number_density_bkgd = background_density * (1.0 - beam_frac)
    if beam:
        ion_gyrofrequency = ppf.gyrofrequency(B0 * u.T, fastSpecies)
    else:
        ion_gyrofrequency = ppf.gyrofrequency(B0 * u.T, bkgdSpecies)

    print(f"Ion gyrofrequency: {ion_gyrofrequency}")
    outputNcRoot.ionGyrofrequency_radPs = ion_gyrofrequency.value
    ion_gyroperiod = 1.0 / ion_gyrofrequency
    print(f"Ion gyroperiod: {ion_gyroperiod}")
    outputNcRoot.ionGyroperiod_sPrad = ion_gyroperiod.value
    ion_gyroperiod_s = ion_gyroperiod * 2.0 * np.pi * u.rad
    print(f"Ion gyroperiod: {ion_gyroperiod_s}")
    outputNcRoot.ionGyroperiod_s = ion_gyroperiod_s.value
    plasma_freq = ppf.plasma_frequency(number_density_bkgd / u.m**3, bkgdSpecies)
    print(f"Plasma frequency: {plasma_freq}")
    outputNcRoot.plasmaFrequency_radPs = plasma_freq.value

    alfven_velocity = pps.Alfven_speed(B0 * u.T, number_density_bkgd / u.m**3, bkgdSpecies)
    print(f"Alfven speed: {alfven_velocity}")
    outputNcRoot.alfvenSpeed = alfven_velocity.value
    sim_L_vA_Tci = sim_L / (ion_gyroperiod_s * alfven_velocity)

    # Normalised units
    simtime_Tci = sim_time / ion_gyroperiod_s
    print(f"NORMALISED: Sim time in Tci: {simtime_Tci}")
    outputNcRoot.simTime_Tci = simtime_Tci
    print(f"NORMALISED: Temporal sampling frequency in Wci: {num_t/simtime_Tci}")
    outputNcRoot.timeSamplingFreq_Wci = num_t/simtime_Tci
    print(f"NORMALISED: Temporal Nyquist frequency in Wci: {num_t/(2.0 *simtime_Tci)}")
    outputNcRoot.timeNyquistFreq_Wci = num_t/(2.0 *simtime_Tci)
    print(f"NORMALISED: Sim L in vA*Tci: {sim_L_vA_Tci}")
    outputNcRoot.simLength_VaTci = sim_L_vA_Tci
    print(f"NORMALISED: Spatial sampling frequency in Wci/vA: {num_cells/sim_L_vA_Tci}")
    outputNcRoot.spaceSamplingFreq_WciOverVa = num_cells/sim_L_vA_Tci
    print(f"NORMALISED: Nyquist frequency in Wci/vA: {num_cells/(2.0 * sim_L_vA_Tci)}")
    outputNcRoot.spaceNyquistFreq_WciOverVa = num_cells/(2.0 * sim_L_vA_Tci)

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
        maxK : float = 100.0,
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
        displayPlots = False):
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

    for simFolder in glob.glob(str(directory / "run_*") + os.path.sep):

        simFolder = Path(simFolder)

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

        if outFileDirectory is None:
            outFileDirectory = directory / "analysis"
            if not os.path.exists(outFileDirectory):
                os.mkdir(outFileDirectory)
        outFilename = simFolder.name + "_analysis.nc"
        outFilepath = os.path.join(outFileDirectory, outFilename)
        outputRoot = nc.Dataset(outFilepath, "w", format="NETCDF4")

        ion_gyroperiod, alfven_velocity = calculate_simulation_parameters(inputDeck, ds, outputRoot, beam, fastSpecies, bkgdSpecies)

        ds = normalise_data(ds, ion_gyroperiod, alfven_velocity)

        if "all" in fields:
            fields = [str(f) for f in ds.data_vars.keys() if str(f).startswith("Electric_Field") or str(f).startswith("Magnetic_Field")]
        
        for field in fields:

            # Take FFT
            field_unit = ds[field].units
            #print(float(ds[field].load().sum()))
            original_spec : xr.DataArray = xrft.xrft.fft(ds[field].load(), true_amplitude=True, true_phase=True, window=None)
            original_spec = original_spec.rename(freq_time="frequency", freq_x_space="wavenumber")
            # Remove zero-frequency component
            original_spec = original_spec.where(original_spec.wavenumber!=0.0, None)

            tk_spec = create_t_k_spectrum(original_spec)

            # Dispersion relations
            if createPlots:

                plt.rcParams.update({'axes.labelsize': plotLabelSize})
                plt.rcParams.update({'axes.titlesize': plotTitleSize})
                plt.rcParams.update({'xtick.labelsize': plotTickSize})
                plt.rcParams.update({'ytick.labelsize': plotTickSize})

                create_omega_k_plots(original_spec, field, field_unit, outFileDirectory, simFolder.name, inputDeck, maxK=maxK, maxW=maxW, log=takeLog, display=displayPlots)

                create_t_k_plots(tk_spec, field, field_unit, outFileDirectory, simFolder.name, maxK, takeLog, displayPlots)

            growth_rate_group = outputRoot.createGroup("growthRates")
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

            gammaWindowIndices = int((gammaWindowPct / 100.0) * tk_spec.coords['time'].size)
            max_gammas = find_max_growth_rates_of_top_n_k_with_max_peak_power(tk_spec, gammaWindowIndices, 3)
            # print("Max growth rates found in wavenumbers with highest peak power: ")
            # for mg in max_gammas:
            #     print(mg.to_string())
            for i in range(0, len(max_gammas)):
                
                # How to append growth rate dataclasses?
                gamma_var[i,]

            max_gammas = find_max_growth_rates_of_top_n_k_with_max_total_power(tk_spec, gammaWindowIndices, 3)
            # print("Max growth rates found in wavenumbers with highest total power: ")
            # for mg in max_gammas:
            #     print(mg.to_string())
            total_power_group = growth_rate_group.createGroup("maxTotalPowerWavenumbers")

        outputRoot.close()
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--dir",
        action="store",
        help="Directory containing multiple simulation directories for evaluation.",
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
    
    args = parser.parse_args()

    process_simulation_batch(directory=args.dir, fields=args.fields, numGrowthRates=args.numGrowthRates, takeLog=args.takeLog, createPlots=args.createPlots, displayPlots=args.displayPlots)