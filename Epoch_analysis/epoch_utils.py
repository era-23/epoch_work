from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import netCDF4 as nc
import astropy.units as u
import xrft  # noqa: E402
from scipy import stats
from plasmapy.formulary import frequencies as ppf
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from dataclasses import dataclass

@dataclass
class LinearGrowthRate:
    gamma : float
    timeStartIndex : float = None
    timeEndIndex : float = None
    timeMidpointIndex : float = None
    timeMidpoint: float = None
    yIntercept : float = None
    rSquared : float = None
    wavenumber : float = None
    peakPower : float = None
    totalPower : float = None

    def to_string(self) -> str:
        return f"Wavenumber: {float(self.wavenumber)}, Peak power: {float(self.peakPower)}, Total power: {float(self.totalPower)}, Time (midpoint): {float(self.timeMidpoint)}, Growth rate: {float(self.gamma)}, SoS residual: {float(self.rSquared)}"

@dataclass
class LinearGrowthRateByT:
    time: float
    gamma: float
    yIntercept: float
    residual: float

@dataclass
class MaxGrowthRate:
    wavenumber: float
    time: float
    gamma: float
    yIntercept: float

@dataclass
class GPModel:
    kernelName : str
    inputNames : list
    normalisedInputs : np.ndarray
    outputName : str
    output : np.ndarray
    regressionModel : GaussianProcessRegressor = None
    classificationModel : GaussianProcessClassifier = None
    modelParams : dict = None

E_TRACE_SPECIES_COLOUR_MAP = {
    "proton" : "orange",
    "electron" : "blue",
    "magneticField" : "purple",
    "electricField" : "green",
    "fastIon" : "red"
}

SPECIES_NAME_MAP = {
    "protonMeanEnergyDensity" : "Bkgd proton",
    "electronMeanEnergyDensity" : "Bkgd electron",
    "magneticFieldMeanEnergyDensity" : "B-field",
    "electricFieldMeanEnergyDensity" : "E-field",
    "fastIonMeanEnergyDensity" : "Fast ion"
}

# Formerly everything below maxRes percentile, i.e. maxRes == 0.2 --> all values within bottom (best) 20th percentile
# Now everything at maxRes proportion and below, i.e. maxRes == 0.2 --> bottom (best) 20% of values
def filter_by_residuals(x, residuals, maxRes):
    x = np.array(x)

    min_res = np.nanmin(residuals)
    max_res = np.nanmax(residuals)
    range_res = max_res - min_res
    absoluteMaxResidual = min_res + (maxRes * range_res)
    
    # Filter growth rates
    x_low_residuals = np.array([i for i in x if i.residual <= absoluteMaxResidual])

    return x_low_residuals

def create_omega_k_plots(
        fftSpectrum : xr.DataArray,
        statsFile : nc.Dataset,
        field : str,
        field_unit : str,
        saveDirectory : Path,
        runName : str,
        inputDeck : dict,
        bkgdSpecies : str,
        fastSpecies : str,
        maxK : float,
        maxW : float,
        display : bool,
        debug):

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
    filename = Path(f'{runName}_{field.replace("_", "")}_powerByOmega_maxK-{maxK if maxK is not None else "all"}_maxW-{maxW if maxW is not None else "all"}.png')
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
    filename = Path(f'{runName}_{field.replace("_", "")}_powerByK_maxK-{maxK if maxK is not None else "all"}_maxW-{maxW if maxW is not None else "all"}.png')
    fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
        plt.close("all")

    # Full dispersion relation for positive omega
    fig, axs = plt.subplots(figsize=(15, 10))
    spec.plot(ax=axs, cbar_kwargs={'label': f'Spectral power in {field} [{field_unit}]'}, cmap='plasma')
    axs.set_ylabel(r"Frequency [$\omega_{ci}$]")
    axs.set_xlabel(r"Wavenumber [$\omega_{ci}/V_A$]")
    filename = Path(f'{runName}_{field.replace("_", "")}_wk_maxK-{maxK if maxK is not None else "all"}_maxW-{maxW if maxW is not None else "all"}.png')
    fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
        plt.close("all")

    log_spec = np.log(spec)

    # Full dispersion relation for positive omega (log)
    fig, axs = plt.subplots(figsize=(15, 10))
    log_spec.plot(ax=axs, cbar_kwargs={'label': f'Log of spectral power in {field}'}, cmap='plasma')
    axs.set_ylabel(r"Frequency [$\omega_{ci}$]")
    axs.set_xlabel(r"Wavenumber [$\omega_{ci}/V_A$]")
    filename = Path(f'{runName}_{field.replace("_", "")}_wk_log_maxK-{maxK if maxK is not None else "all"}_maxW-{maxW if maxW is not None else "all"}.png')
    fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
        plt.close("all")

    # Positive omega/positive k with vA and lower hybrid frequency
    fig, axs = plt.subplots(figsize=(15, 10))
    spec = spec.sel(wavenumber=spec.wavenumber>0.0)
    spec.plot(ax=axs, cbar_kwargs={'label': f'Spectral power in {field} [{field_unit}]'}, cmap='plasma')
    axs.plot(spec.coords['wavenumber'].data, spec.coords['wavenumber'].data, 'w--', label=r'$V_A$ branch')
    B0 = inputDeck['constant']['b0_strength']
    bkgd_number_density = float(inputDeck['constant']['background_density'])
    wLH_cyclo = ppf.lower_hybrid_frequency(B0 * u.T, bkgd_number_density * u.m**-3, bkgdSpecies) / ppf.gyrofrequency(B0 * u.T, fastSpecies)
    axs.axhline(y = wLH_cyclo, color='white', linestyle=':', label=r'Lower hybrid frequency')
    axs.legend(loc='upper left')
    axs.set_ylabel(r"Frequency [$\omega_{ci}$]")
    axs.set_xlabel(r"Wavenumber [$\omega_{ci}/V_A$]")
    filename = Path(f'{runName}_{field.replace("_", "")}_wk_positiveK_maxK-{maxK if maxK is not None else "all"}_maxW-{maxW if maxW is not None else "all"}.png')
    fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
        plt.close("all")

    # Positive omega/positive k with vA and lower hybrid frequency (log)
    fig, axs = plt.subplots(figsize=(15, 10))
    log_spec = log_spec.sel(wavenumber=log_spec.wavenumber>0.0)
    log_spec.plot(ax=axs, cbar_kwargs={'label': f'Log of spectral power in {field}'}, cmap='plasma')
    axs.plot(log_spec.coords['wavenumber'].data, log_spec.coords['wavenumber'].data, 'w--', label=r'$V_A$ branch')
    axs.axhline(y = wLH_cyclo, color='white', linestyle=':', label=r'Lower hybrid frequency')
    axs.legend(loc='upper left')
    axs.set_ylabel(r"Frequency [$\omega_{ci}$]")
    axs.set_xlabel(r"Wavenumber [$\omega_{ci}/V_A$]")
    filename = Path(f'{runName}_{field.replace("_", "")}_wk_positiveK_log_maxK-{maxK if maxK is not None else "all"}_maxW-{maxW if maxW is not None else "all"}.png')
    fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
        plt.close("all")
    
    del(spec)
    del(log_spec)

def create_t_k_spectrum(
        originalFftSpectrum : xr.DataArray, 
        statsFile : nc.Dataset = None,
        maxK : float = 100.0,
        load : bool = True,
        debug : bool = False
) -> xr.DataArray :
    
    tk_spec = originalFftSpectrum.where(originalFftSpectrum.frequency>0.0, 0.0)
    original_zero_freq_amplitude = tk_spec.sel(wavenumber=0.0)
    # Double spectrum to conserve E
    tk_spec = np.sqrt(2.0) * tk_spec # <---- Should this be sqrt(2)?
    tk_spec.loc[dict(wavenumber=0.0)] = original_zero_freq_amplitude # Restore original 0-freq amplitude
    tk_spec = xrft.xrft.ifft(tk_spec, dim="frequency")
    tk_spec = tk_spec.rename(freq_frequency="time")
    abs_spec = np.abs(tk_spec)

    if statsFile is not None:
        # Log stats on spectrum
        tk_sum = float(abs_spec.sum())
        statsFile.totalTkSpectralPower = tk_sum

        tk_squared = float((np.abs(abs_spec)**2).sum())
        parseval_tk = tk_squared  * abs_spec.coords['wavenumber'].spacing * abs_spec.coords['time'].spacing
        statsFile.parsevalTk = parseval_tk
        
        tk_peak = float(np.nanmax(abs_spec))
        statsFile.peakTkSpectralPower = tk_peak
        
        tk_mean = float(abs_spec.mean())
        statsFile.meanTkSpectralPower = tk_mean

        statsFile.peakTkSpectralPowerRatio = tk_peak/tk_mean
    
    if debug:
        print(f"Sum of t-k squared * dk * dt: {parseval_tk}")
        print(f"Max peak in t-k: {tk_peak}")
        print(f"Mean of t-k: {tk_mean}")
        print(f"Ratio of peak to mean in t-k: {tk_peak/tk_mean}")

    if maxK is not None:
        tk_spec = tk_spec.sel(wavenumber=tk_spec.wavenumber<=maxK)
        tk_spec = tk_spec.sel(wavenumber=tk_spec.wavenumber>=-maxK)

    if load:
        tk_spec = tk_spec.load()

    return tk_spec

def create_t_k_plot(
        tkSpectrum : xr.DataArray,
        field : str,
        field_unit : str,
        saveDirectory : Path = None,
        runName : str = None,
        maxK : float = 100.0,
        display : bool = False):
    
    print("Generating t-k plot....")

    if maxK is not None:
        tkSpec_plot = tkSpectrum.sel(wavenumber=tkSpectrum.wavenumber<=maxK)
        tkSpec_plot = tkSpec_plot.sel(wavenumber=tkSpec_plot.wavenumber>=-maxK)
    tkSpec_plot = np.abs(tkSpec_plot)
    tkSpec_plot_log = np.log(tkSpec_plot)
    
    # Time-wavenumber
    fig, axs = plt.subplots(figsize=(15, 10))
    tkSpec_plot.plot(ax=axs, x = "wavenumber", y = "time", cbar_kwargs={'label': f'Spectral power in {field} [{field_unit}]'}, cmap='plasma')
    axs.grid()
    axs.set_xlabel(r"Wavenumber [$\omega_{ci}/V_A$]")
    axs.set_ylabel(r"Time [$\tau_{ci}$]")
    if saveDirectory is not None:
        filename = Path(f'{runName}_{field.replace("_", "")}_tk_maxK-{maxK if maxK is not None else "all"}.png')
        fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
    plt.close('all')

    # Time-wavenumber (log)
    fig, axs = plt.subplots(figsize=(15, 10))
    tkSpec_plot_log.plot(ax=axs, x = "wavenumber", y = "time", cbar_kwargs={'label': f'Log of spectral power in {field}'}, cmap='plasma')
    axs.grid()
    axs.set_xlabel(r"Wavenumber [$\omega_{ci}/V_A$]")
    axs.set_ylabel(r"Time [$\tau_{ci}$]")
    if saveDirectory is not None:
        filename = Path(f'{runName}_{field.replace("_", "")}_tk_log_maxK-{maxK if maxK is not None else "all"}.png')
        fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
    plt.close('all')

def create_autobispectrum(
        signalWindows : list, 
        maxK = None, 
        mask : bool = True
):
    initBispec = True
    count = 0
    for window in signalWindows:
        
        # Build bispectrum by averaging FFT data around the time point
        if initBispec:
            bispec = np.zeros((window.shape[1], window.shape[1]), dtype=complex)
            initBispec = False
        
        # y = spec.mean(dim="time").to_numpy()
        y = window.to_numpy()
        nfft = window.shape[1]
        # Create all combinations of k1 and k2
        k = np.fft.fftshift(np.fft.fftfreq(nfft, 1/nfft))
        K1, K2 = np.meshgrid(k, k)
        K3 = K1 + K2
        i1 = (K1 - k.min()).astype(int)
        i2 = (K2 - k.min()).astype(int)
        i3 = (K3 - k.min()).astype(int)

        # Mask
        k_mask = np.abs(i3) < (nfft/2)
        valid_I3 = np.where(k_mask, i3, 0)
        Y3_conj = np.conj(y[:,valid_I3])

        # Use broadcasting to access X[k1], X[k2], X[k1 + k2]
        # b = y[:,K1] * y[:,K2] * Y3_conj
        bispec += np.mean(y[:,i1] * y[:,i2] * Y3_conj, axis = 0)
        bispec[np.logical_not(k_mask)] = 0.0

        count += 1

    # bispec = np.fft.fftshift(bispec) / count
    bispec = bispec / count
    waxis = np.linspace(-maxK, maxK, bispec.shape[0])

    if mask:
        # Lower triangle mask
        lower_mask = np.tri(bispec.shape[0], bispec.shape[1], k=-1)
        bispec = np.ma.array(bispec, mask = lower_mask)

    return bispec, waxis
    
def create_autobicoherence(
        signalWindows : list, 
        maxK = None, 
        mask : bool = True
):
    initBicoh = True
    count = 0
    for window in signalWindows:
        
        # Build bispectrum by averaging FFT data around the time point
        if initBicoh:
            bicoh = np.zeros((window.shape[1], window.shape[1]), dtype=complex)
            initBicoh = False
        
        # y = spec.mean(dim="time").to_numpy()
        y = window.to_numpy()
        nfft = window.shape[1]
        # Create all combinations of k1 and k2
        k = np.fft.fftshift(np.fft.fftfreq(nfft, 1/nfft))
        K1, K2 = np.meshgrid(k, k)
        K3 = K1 + K2
        i1 = (K1 - k.min()).astype(int)
        i2 = (K2 - k.min()).astype(int)
        i3 = (K3 - k.min()).astype(int)

        # Mask
        k_mask = np.abs(i3) < (nfft/2)
        valid_I3 = np.where(k_mask, i3, 0)
        Y3_conj = np.conj(y[:,valid_I3])

        # Use broadcasting to access X[k1], X[k2], X[k1 + k2]
        bicoh_top = np.abs(np.mean(y[:,i1] * y[:,i2] * Y3_conj, axis = 0))**2
        bicoh_bottom = np.mean(np.abs(y[:,i1] * y[:,i2])**2, axis=0) * np.mean(np.abs(Y3_conj)**2, axis=0)
        bicoh += bicoh_top / bicoh_bottom

        count += 1

    # bispec = np.fft.fftshift(bispec) / count
    bicoh = bicoh / count
    waxis = np.linspace(-maxK, maxK, bicoh.shape[0])

    if mask:
        # Lower triangle mask
        lower_mask = np.tri(bicoh.shape[0], bicoh.shape[1], k=-1)
        bicoh = np.ma.array(bicoh, mask = lower_mask)

    return bicoh, waxis

def bispectral_analysis(
        tkSpectrum : xr.DataArray, 
        runName : str,
        field : str,
        display : bool = False,
        saveDirectory : Path = None,
        timePoint_tci : float = None, 
        totalWindow_tci = 2.0, 
        fftWindowSize_tci = 0.25, 
        overlap = 0.5, 
        maxK = None, 
        mask : bool = True
):
    print("Doing bispectral analysis....")

    times = np.array(tkSpectrum.coords["time"])

    timeCentre = timePoint_tci
    if timePoint_tci is None:
        timeCentre = times[len(times)//2]
        totalWindow_tci = np.floor(times[-1])
        fftWindowSize_tci = 0.25
        overlap = 0.25

    # Work out how many indices in each window and window spectrum
    windowSize_indices = int((times.size / times[-1]) * fftWindowSize_tci)
    overlap_indices = int(np.round(overlap * windowSize_indices))
    windowStart_tci = timeCentre - (0.5 * totalWindow_tci)
    windowStart_idx = np.where(abs(times-windowStart_tci)==abs(times-windowStart_tci).min())[0][0]
    windowEnd_tci = timeCentre + (0.5 * totalWindow_tci)
    windowEnd_idx = np.where(abs(times-windowEnd_tci)==abs(times-windowEnd_tci).min())[0][0]
    signalWindows = []
    startIdx = windowStart_idx
    endIdx = startIdx + windowSize_indices
    while endIdx < windowEnd_idx:
        w = tkSpectrum.isel(time=slice(startIdx, endIdx))
        signalWindows.append(w)
        startIdx = (endIdx + 1) - overlap_indices
        endIdx = startIdx + windowSize_indices

    # Autobispectrum
    bispec, waxis = create_autobispectrum(signalWindows, maxK, mask)
    maxis_waxis = waxis[-1]
    minis_waxis = waxis[0]
    fig, axs = plt.subplots(figsize=(12, 12))
    img = axs.imshow(np.abs(bispec), extent=[minis_waxis, maxis_waxis, minis_waxis, maxis_waxis], origin="lower", cmap="plasma")
    axs.set_title(f"{field} autobispectrum t = {timeCentre}" if timePoint_tci is not None else f"{field} autobispectrum (full time window)", pad=20.0)
    axs.set_xlabel('Wavenumber $k_1$')
    axs.set_ylabel('Wavenumber $k_2$')
    axs.grid(True)
    fig.colorbar(img, ax=axs, label='Magnitude', shrink = 0.7)
    fig.tight_layout()
    if saveDirectory is not None:
        filename = Path(f'{runName}_{field.replace("_", "")}_autobispectrum_t{timeCentre:.3f}_maxK-{maxK if maxK is not None else "all"}.png')
        fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
    plt.close('all')

    # Log autobispectrum
    fig, axs = plt.subplots(figsize=(12, 12))
    img = axs.imshow(np.log(np.abs(bispec)), extent=[minis_waxis, maxis_waxis, minis_waxis, maxis_waxis], origin="lower", cmap="plasma")
    axs.set_title(f"Log {field} autobispectrum t = {timeCentre}" if timePoint_tci is not None else f"Log {field} autobispectrum (full time window)", pad=20.0)
    axs.set_xlabel('Wavenumber $k_1$')
    axs.set_ylabel('Wavenumber $k_2$')
    axs.grid(True)
    fig.colorbar(img, ax=axs, label='Log magnitude', shrink = 0.7)
    fig.tight_layout()
    if saveDirectory is not None:
        filename = Path(f'{runName}_log_{field.replace("_", "")}_autobispectrum_t{timeCentre:.3f}_maxK-{maxK if maxK is not None else "all"}.png')
        fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
    plt.close('all')

    # Autobicoherence
    bicoh, waxis = create_autobicoherence(signalWindows, maxK, mask)
    maxis_waxis = waxis[-1]
    minis_waxis = waxis[0]
    fig, axs = plt.subplots(figsize=(12, 12))
    img = axs.imshow(np.abs(bicoh), extent=[minis_waxis, maxis_waxis, minis_waxis, maxis_waxis], origin="lower", cmap="plasma")
    axs.set_title(f"{field} auto-bicoherence{r'$^2$'} t = {timeCentre}" if timePoint_tci is not None else f"{field} auto-bicoherence{r'$^2$'} (full time window)", pad=20.0)
    axs.set_xlabel('Wavenumber $k_1$')
    axs.set_ylabel('Wavenumber $k_2$')
    axs.grid(True)
    fig.colorbar(img, ax=axs, label='Magnitude', shrink = 0.7)
    fig.tight_layout()
    if saveDirectory is not None:
        filename = Path(f'{runName}_{field.replace("_", "")}_autobicoherence_t{timeCentre:.3f}_maxK-{maxK if maxK is not None else "all"}.png')
        fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
    plt.close('all')
    
def create_netCDF_fieldVariable_structure(
        fieldRoot : nc.Dataset,
        numGrowthRates : int
) -> nc.Dataset:
    growth_rate_group = fieldRoot.createGroup("growthRates")
    posGrowthRateGrp = growth_rate_group.createGroup("positive")
    negGrowthRateGrp = growth_rate_group.createGroup("negative")
    groups = [posGrowthRateGrp, negGrowthRateGrp]
    
    # NOTE: This will skip adding fields if the "wavenumber" dimension is already present, so this assumes no changes to the variable creation below is required.
    #       A more sophisticated diff would be better.
    for group in groups:
        if "wavenumber" in group.dimensions.keys():
            continue
        else:
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
        growthRateData : list[LinearGrowthRate],
        numToPlot : int,
        selectionMetric : str,
        save : bool,
        display : bool,
        noTitle : bool,
        saveFolder : Path,
        runName : str,
        debug : bool):

    # Short-circuit
    if numToPlot == 0:
        return
    
    spectrum = np.abs(tkSpectrum)

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
        signal = spectrum.sel(wavenumber=g.wavenumber)
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

def find_best_growth_rates(
        tkSpectrum : xr.DataArray,
        gammaWindowPctMin : int,
        gammaWindowPctMax : int,
        debug : bool):

    spectrum = np.abs(tkSpectrum)
    # Change: This now only evaluates at integer values of percentage, i.e. if 5% - 15% range is given, this evaluates at 5%, 6%, 7% etc.
    gammaWindowIndicesMin = int(np.rint((gammaWindowPctMin / 100.0) * spectrum.coords['time'].size))
    gammaWindowIndicesMax = int(np.rint((gammaWindowPctMax / 100.0) * spectrum.coords['time'].size))
    onePercentIndices = int(np.rint(0.01 * spectrum.coords['time'].size))

    best_pos_growth_rates = []
    best_neg_growth_rates = []

    num_wavenumbers = spectrum.sizes["wavenumber"]

    for index in range(0, num_wavenumbers):

        # Must load data here for quick iteration over windows
        # Changed: this is pre-loaded in create_t_k_spectrum()
        signal = spectrum.isel(wavenumber=index)

        signalK=float(spectrum.coords['wavenumber'][index])
        signalPeak=float(signal.max())
        signalTotal=float(signal.sum())

        windowWidths = range(gammaWindowIndicesMin, gammaWindowIndicesMax + 1, onePercentIndices)
        len_widths = len(windowWidths)

        best_pos_params = None
        best_neg_params = None
        best_pos_r_squared = float('-inf')
        best_neg_r_squared = float('-inf')

        width_count = 0
        for width in windowWidths: # For each possible window width
            
            windowStarts = range(0, len(signal) - (width + 1), onePercentIndices)

            if debug:
                width_count += 1
                len_windows = len(windowStarts)
                window_count = 0

            for window in windowStarts: # For each window
                
                if debug:
                    window_count += 1
                    print(f"Processing width {width} ({(width/spectrum.coords['time'].size)*100.0}% of signal) starting at {window} in k = {signalK}. Width {width_count}/{len_widths} window {window_count}/{len_windows}....")

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
                LinearGrowthRate(timeStartIndex=windowStart,
                                    timeEndIndex=(windowStart + window_width),
                                    timeMidpointIndex=windowStart+(int(window_width/2)),
                                    gamma=gamma,
                                    yIntercept=y_int,
                                    rSquared=r_sqrd,
                                    wavenumber=signalK,
                                    timeMidpoint=float(spectrum.coords['time'][windowStart+(int(window_width/2))]),
                                    peakPower = signalPeak,
                                    totalPower = signalTotal))
            
        if best_neg_params is not None:
            gamma, y_int, window_width, windowStart, r_sqrd = best_neg_params
            best_neg_growth_rates.append(
                LinearGrowthRate(timeStartIndex=windowStart,
                                    timeEndIndex=(windowStart + window_width),
                                    timeMidpointIndex=windowStart+(int(window_width/2)),
                                    gamma=gamma,
                                    yIntercept=y_int,
                                    rSquared=r_sqrd,
                                    wavenumber=signalK,
                                    timeMidpoint=float(spectrum.coords['time'][windowStart+(int(window_width/2))]),
                                    peakPower = signalPeak,
                                    totalPower = signalTotal))
        del(signal)
        
        if debug: 
            if best_pos_params is not None:
                print(f"Max positive gamma found: {best_pos_growth_rates[-1].to_string()}")
            if best_neg_params is not None:
                print(f"Max negative gamma found: {best_neg_growth_rates[-1].to_string()}")

    return best_pos_growth_rates, best_neg_growth_rates

def process_growth_rates(
        tkSpectrum : xr.DataArray,
        fieldRoot : nc.Dataset,
        plotFieldFolder : Path,
        simFolder : Path,
        field : str,
        gammaWindowPctMin : float,
        gammaWindowPctMax : float,
        saveGrowthRatePlots : bool,
        numGrowthRatesToPlot : int,
        displayPlots : bool,
        noTitle : bool,
        debug : bool):

    print("Processing growth rates....")

    best_pos_gammas, best_neg_gammas = find_best_growth_rates(tkSpectrum, gammaWindowPctMin, gammaWindowPctMax, debug)
    maxNumGammas = np.max([len(best_pos_gammas), len(best_neg_gammas)])
    growthRateStatsRoot = create_netCDF_fieldVariable_structure(fieldRoot, maxNumGammas)

    if saveGrowthRatePlots:
        gammaPlotFolder = plotFieldFolder / "growth_rates"
        plot_growth_rates(tkSpectrum, field, best_pos_gammas, numGrowthRatesToPlot, "peak", saveGrowthRatePlots, displayPlots, noTitle, gammaPlotFolder, simFolder.name, debug)
        plot_growth_rates(tkSpectrum, field, best_neg_gammas, numGrowthRatesToPlot, "total", saveGrowthRatePlots, displayPlots, noTitle, gammaPlotFolder, simFolder.name, debug)

    # Positive growth
    posGammaNc = growthRateStatsRoot.groups["positive"]
    for i in range(len(best_pos_gammas)):
        gamma = best_pos_gammas[i]
        posGammaNc.variables["wavenumber"][i] = gamma.wavenumber
        posGammaNc.variables["peakPower"][i] = gamma.peakPower
        posGammaNc.variables["totalPower"][i] = gamma.totalPower
        posGammaNc.variables["time"][i] = gamma.timeMidpoint
        posGammaNc.variables["growthRate"][i] = gamma.gamma
        posGammaNc.variables["rSquared"][i] = gamma.rSquared
        posGammaNc.variables["yIntercept"][i] = gamma.yIntercept

    keyMetricsIndices = {
        "maxFoundInSimulation" : np.argmax([g.gamma for g in best_pos_gammas]),
        "bestInHighestPeakPowerK" : np.argmax([g.peakPower for g in best_pos_gammas]),
        "bestInHighestTotalPowerK" : np.argmax([g.totalPower for g in best_pos_gammas])
    }
    for metric, index in keyMetricsIndices.items():
        group = posGammaNc.createGroup(metric)
        gamma = best_pos_gammas[index]
        group.growthRate = float(gamma.gamma)
        group.time=float(gamma.timeMidpoint)
        group.yIntercept=float(gamma.yIntercept)
        group.rSquared=float(gamma.rSquared)
        group.wavenumber=float(gamma.wavenumber)
        group.peakPower=float(gamma.peakPower)
        group.totalPower=float(gamma.totalPower)

    # Negative growth
    negGammaNc = growthRateStatsRoot.groups["negative"]
    for i in range(len(best_neg_gammas)):
        gamma = best_neg_gammas[i]
        negGammaNc.variables["wavenumber"][i] = gamma.wavenumber
        negGammaNc.variables["peakPower"][i] = gamma.peakPower
        negGammaNc.variables["totalPower"][i] = gamma.totalPower
        negGammaNc.variables["time"][i] = gamma.timeMidpoint
        negGammaNc.variables["growthRate"][i] = gamma.gamma
        negGammaNc.variables["rSquared"][i] = gamma.rSquared
        negGammaNc.variables["yIntercept"][i] = gamma.yIntercept

    keyMetricsIndices["maxFoundInSimulation"] = np.argmin([g.gamma for g in best_neg_gammas])
    keyMetricsIndices["bestInHighestPeakPowerK"] = np.argmax([g.peakPower for g in best_neg_gammas])
    keyMetricsIndices["bestInHighestTotalPowerK"] = np.argmax([g.totalPower for g in best_neg_gammas])
    for metric, index in keyMetricsIndices.items():
        group = negGammaNc.createGroup(metric)
        gamma = best_neg_gammas[index]
        group.growthRate = float(gamma.gamma)
        group.time=float(gamma.timeMidpoint)
        group.yIntercept=float(gamma.yIntercept)
        group.rSquared=float(gamma.rSquared)
        group.wavenumber=float(gamma.wavenumber)
        group.peakPower=float(gamma.peakPower)
        group.totalPower=float(gamma.totalPower)