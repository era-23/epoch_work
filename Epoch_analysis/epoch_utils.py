from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import netCDF4 as nc
import astropy.units as u
import xrft  # noqa: E402
from scipy import stats
from scipy.interpolate import make_smoothing_spline, BSpline
from plasmapy.formulary import frequencies as ppf
from dataclasses import dataclass

from collections.abc import Sequence
from warnings import warn
from inference.pdf.hdi import sample_hdi
from inference.pdf.kde import GaussianKDE, KDE2D

from matplotlib import colormaps

@dataclass
class LinearGrowthRate:
    gamma : float
    timeStartIndex : float = None
    timeEndIndex : float = None
    timeMidpointIndex : float = None
    timeMidpoint: float = None
    yIntercept : float = None
    rSquared : float = None
    stdErr : float = None
    rawWindowVariance : float = None
    wavenumber : float = None
    peakPower : float = None
    totalPower : float = None
    smoothingFunction : BSpline = None

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
            group.createVariable("stdErr", datatype="f4", dimensions=("wavenumber",))
            group.createVariable("rawWindowVariance", datatype="f4", dimensions=("wavenumber",))

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
        logSignal = np.log(signal)
        logSignal.plot(ax=ax, alpha = 0.5, color = "blue")
        if g.smoothingFunction is not None:
            ax.plot(logSignal.coords['time'], np.log(g.smoothingFunction(logSignal.coords['time'])), linestyle = "dashed", color="purple", label = "Smoothed signal")
        ax.plot(timeVals, g.gamma * timeVals + g.yIntercept, color = "orange", label = r"$\gamma = $" + f"{g.gamma:.3f}" + r"$\pm$" + f"{g.stdErr:.3f}" + r"$\omega_{ci}$")
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
        useSmoothing : bool = True,
        debug : bool = False):

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
        rawSignal = spectrum.isel(wavenumber=index)

        signalK=float(spectrum.coords['wavenumber'][index])
        signalPeak=float(rawSignal.max())
        signalTotal=float(rawSignal.sum())

        if useSmoothing:
            # Smooth signal for finding growth rates
            smoothingFunction = make_smoothing_spline(rawSignal.coords["time"], np.nan_to_num(rawSignal), lam = 0.01)
            signal = smoothingFunction(rawSignal.coords["time"])
            # rawSignal.plot(alpha = 0.7)
            # plt.plot(rawSignal.coords["time"], signal, linestyle = "dashed", color = "purple")
            # plt.show()    
        else:
            smoothingFunction = None
            signal = rawSignal        

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

                result = stats.linregress(rawSignal.coords["time"][window:(width + window)], np.log(t_k_window))
                r_squared = result.rvalue ** 2
                
                if not np.isnan(result.slope):
                    if result.slope > 0.0:
                        if r_squared > best_pos_r_squared:
                            best_pos_r_squared = r_squared
                            best_pos_params = (result.slope, result.intercept, width, window, r_squared, result.stderr)
                    else:
                        if r_squared > best_neg_r_squared:
                            best_neg_r_squared = r_squared
                            best_neg_params = (result.slope, result.intercept, width, window, r_squared, result.stderr)

        if best_pos_params is not None:
            gamma, y_int, window_width, window_start, r_sqrd, std_err = best_pos_params
            best_pos_growth_rates.append(
                LinearGrowthRate(timeStartIndex=window_start,
                                timeEndIndex=(window_start + window_width),
                                timeMidpointIndex=window_start+(int(window_width/2)),
                                gamma=gamma,
                                yIntercept=y_int,
                                rSquared=r_sqrd,
                                stdErr=std_err,
                                rawWindowVariance=np.var(rawSignal[window_start:window_start+window_width]),
                                wavenumber=signalK,
                                timeMidpoint=float(spectrum.coords['time'][window_start+(int(window_width/2))]),
                                peakPower = signalPeak,
                                totalPower = signalTotal,
                                smoothingFunction=smoothingFunction))
            
        if best_neg_params is not None:
            gamma, y_int, window_width, window_start, r_sqrd, std_err = best_neg_params
            best_neg_growth_rates.append(
                LinearGrowthRate(timeStartIndex=window_start,
                                timeEndIndex=(window_start + window_width),
                                timeMidpointIndex=window_start+(int(window_width/2)),
                                gamma=gamma,
                                yIntercept=y_int,
                                rSquared=r_sqrd,
                                stdErr=std_err,
                                rawWindowVariance=np.var(rawSignal[window_start:window_start+window_width]),
                                wavenumber=signalK,
                                timeMidpoint=float(spectrum.coords['time'][window_start+(int(window_width/2))]),
                                peakPower = signalPeak,
                                totalPower = signalTotal,
                                smoothingFunction=smoothingFunction))
        del(rawSignal)
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

    best_pos_gammas, best_neg_gammas = find_best_growth_rates(tkSpectrum, gammaWindowPctMin, gammaWindowPctMax, useSmoothing = True, debug = debug)
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
        posGammaNc.variables["stdErr"][i] = gamma.stdErr
        posGammaNc.variables["rawWindowVariance"][i] = gamma.rawWindowVariance
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
        group.stdErr = float(gamma.stdErr)
        group.rawWindowVariance = float(gamma.rawWindowVariance)
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
        negGammaNc.variables["stdErr"][i] = gamma.stdErr
        negGammaNc.variables["rawWindowVariance"][i] = gamma.rawWindowVariance
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
        group.stdErr = float(gamma.stdErr)
        group.rawWindowVariance = float(gamma.rawWindowVariance)
        group.wavenumber=float(gamma.wavenumber)
        group.peakPower=float(gamma.peakPower)
        group.totalPower=float(gamma.totalPower)

def my_matrix_plot(
    data_series,
    series_labels: list[str] = None,
    parameter_labels: list[str] = None,
    show: bool = True,
    normalisePDF: bool = True,
    reference: Sequence[float] = None,
    filename: str = None,
    plot_style: str = "contour",
    colormap_list: list = ["Blues", "Greens"],
    show_ticks: bool = None,
    point_colors: Sequence[float] = None,
    hdi_fractions=(0.35, 0.65, 0.95),
    point_size: int = 1,
    label_size: int = 10,
):
    """
    Construct a 'matrix plot' for a set of variables which shows all possible
    1D and 2D marginal distributions.

    :param data_series: \
        A list of data series', each of which is a list of array-like objects containing the samples for each variable.
    
    :param series_labels: \
        A list of strings to be used as labels for each data series being plotted, in the same order as those in data_series.

    :param parameter_labels: \
        A list of strings to be used as axis labels for each parameter being plotted, in the same order as those in data_series.

    :param bool show: \
        Sets whether the plot is displayed.

    :param reference: \
        A list of reference values for each parameter which will be over-plotted.

    :param str filename: \
        File path to which the matrix plot will be saved (if specified).

    :param str plot_style: \
        Specifies the type of plot used to display the 2D marginal distributions.
        Available styles are 'contour' for filled contour plotting, 'hdi' for
        highest-density interval contouring, 'histogram' for hexagonal-bin histogram,
        and 'scatter' for scatterplot.

    :param str colormap_list: \
        A list of colormaps to be used for plotting. Must be at least as long as len(data_series)
        and contain names of valid colormaps present in ``matplotlib.colormaps``.

    :param bool show_ticks: \
        By default, axis ticks are only shown when plotting less than 6 variables.
        This behaviour can be overridden for any number of parameters by setting
        show_ticks to either True or False.

    :param point_colors: \
        An array containing data which will be used to set the colors of the points
        if the plot_style argument is set to 'scatter'.

    :param point_size: \
        An array containing data which will be used to set the size of the points
        if the plot_style argument is set to 'scatter'.

    :param hdi_fractions: \
        The highest-density intervals used for contouring, specified in terms of
        the fraction of the total probability contained in each interval. Should
        be given as an iterable of floats, each in the range [0, 1].

    :param int label_size: \
        The font-size used for axis labels.
    """
    N_series = len(data_series)
    N_par = len(data_series[0])
    if parameter_labels is None:  # set default axis labels if none are given
        if N_par >= 10:
            parameter_labels = [f"p{i}" for i in range(N_par)]
        else:
            parameter_labels = [f"param {i}" for i in range(N_par)]
    else:
        if len(parameter_labels) != N_par:
            raise ValueError(
                """\n
                \r[ matrix_plot error ]
                \r>> The number of labels given does not match
                \r>> the number of plotted parameters.
                """
            )

    if reference is not None:
        if len(reference) != N_par:
            raise ValueError(
                """\n
                \r[ matrix_plot error ]
                \r>> The number of reference values given does not match
                \r>> the number of plotted parameters.
                """
            )
    # check that given plot style is valid, else default to a histogram
    if plot_style not in ["contour", "hdi", "histogram", "scatter"]:
        plot_style = "contour"
        warn(
            "'plot_style' must be set as either 'contour', 'hdi', 'histogram' or 'scatter'"
        )

    iterable = hasattr(hdi_fractions, "__iter__")
    if not iterable or not all(0 < f < 1 for f in hdi_fractions):
        raise ValueError(
            """\n
            \r[ matrix_plot error ]
            \r>> The 'hdi_fractions' argument must be given as an
            \r>> iterable of floats, each in the range [0, 1].
            """
        )

    # by default, we suppress axis ticks if there are 6 parameters or more to keep things tidy
    if show_ticks is None:
        show_ticks = N_par < 6

    L = 200
    cmaps = []
    marginal_colors = []
    cmap_count = 0
    for c in colormap_list:
        if c in colormaps:
            cmaps.append(colormaps[c])
        else:
            cmaps.append(colormaps[cmap_count])
            cmap_count += 1
            warn(f"'{c}' is not a valid colormap from matplotlib.colormaps. Using default.")
        # find the darker of the two ends of the colormap, and use it for the marginal plots
        marginal_colors.append(sorted([cmaps[-1](10), cmaps[-1](245)], key=lambda x: sum(x[:-1]))[0])

    # build axis arrays and determine limits for all variables
    axis_limits = []
    axis_arrays = []
    if type(data_series[0][0]) is list:
        parameters = data_series[0]
    else:
        parameters = [s.tolist() for s in data_series[0]] 
    for n_series in range(1, N_series):
        for n_sample in range(N_par):
            if type(data_series[n_series][n_sample]) is list:
                parameters[n_sample] += data_series[n_series][n_sample]
            else:
                parameters[n_sample] += data_series[n_series][n_sample].tolist()
    for sample in parameters:
        # get the 98% HDI to calculate plot limits
        lwr, upr = sample_hdi(sample, fraction=0.98)
        # store the limits and axis array
        axis_limits.append([lwr - (upr - lwr) * 0.3, upr + (upr - lwr) * 0.3])
        axis_arrays.append(
            np.linspace(lwr - (upr - lwr) * 0.35, upr + (upr - lwr) * 0.35, L)
        )

    fig = plt.figure(figsize=(8, 8))
    # build a lower-triangular indices list in diagonal-striped order
    inds_list = [(N_par - 1, 0)]  # start with bottom-left corner
    for k in range(1, N_par):
        inds_list.extend([(N_par - 1 - i, k - i) for i in range(k + 1)])

    # now create a dictionary of axis objects with correct sharing
    axes = {}
    for tup in inds_list:
        i, j = tup
        x_share = None
        y_share = None

        if i < N_par - 1:
            x_share = axes[(N_par - 1, j)]

        if (j > 0) and (i != j):  # diagonal doesnt share y-axis
            y_share = axes[(i, 0)]

        axes[tup] = plt.subplot2grid(
            (N_par, N_par), (i, j), sharex=x_share, sharey=y_share
        )

    # Pre-compute estimates for correct scaling of data_series
    all_estimates = []
    all_estimate_maxes = []
    for n_series in range(N_series):
        samples = data_series[n_series]
        series_estimates = []
        series_estimate_maxes = []
        for tup in inds_list:
            i, j = tup
            ax = axes[tup]
            # are we on the diagonal?
            if i == j:
                sample = samples[i]
                pdf = GaussianKDE(sample)
                s_estimate = np.array(pdf(axis_arrays[i]))
                series_estimates.append(s_estimate)
                series_estimate_maxes.append(np.max(series_estimates))
        all_estimates.append(series_estimates)
        all_estimate_maxes.append(series_estimate_maxes)
    
    initialiseAxes = True
    for n_series in range(N_series):
        
        samples = data_series[n_series]
        estimates = all_estimates[n_series]
        estimate_maxes = all_estimate_maxes[n_series]
        marginal_color = marginal_colors[n_series]
        cmap = cmaps[n_series]

        # now loop over grid and plot
        for tup in inds_list:
            i, j = tup
            ax = axes[tup]
            # are we on the diagonal?
            if i == j:
                sample = samples[i]
                estimate = estimates[i]
                ax.plot(
                    axis_arrays[i],
                    0.9 * (estimate / estimate.max()) if normalisePDF else 0.9 * (estimate / estimate_maxes[i]),
                    lw=1,
                    color=marginal_color,
                    label = series_labels[n_series]
                )
                ax.fill_between(
                    axis_arrays[i],
                    0.9 * (estimate / estimate.max()) if normalisePDF else 0.9 * (estimate / estimate_maxes[i]),
                    color=marginal_color,
                    alpha=0.1,
                )
                if reference is not None:
                    ax.plot(
                        [reference[i], reference[i]],
                        [0, 1],
                        lw=1.5,
                        ls="dashed",
                        color="red",
                    )
                ax.set_ylim([0, 1])
            else:
                x = samples[j]
                y = samples[i]

                # plot the 2D marginals
                if plot_style == "contour":
                    # Filled contour plotting using 2D gaussian KDE
                    pdf = KDE2D(x=x, y=y)
                    x_ax = axis_arrays[j][::4]
                    y_ax = axis_arrays[i][::4]
                    X, Y = np.meshgrid(x_ax, y_ax)
                    prob = np.array(pdf(X.flatten(), Y.flatten())).reshape([L // 4, L // 4])
                    ax.set_facecolor(cmap(256 // 20))
                    ax.contourf(X, Y, prob, 10, cmap=cmap)

                elif plot_style == "hdi":
                    # Filled contour plotting using 2D gaussian KDE
                    pdf = KDE2D(x=x, y=y)
                    sample_probs = pdf(x, y)
                    pcts = [100 * (1 - f) for f in hdi_fractions]
                    levels = [l for l in np.percentile(sample_probs, pcts)]

                    x_ax = axis_arrays[j][::4]
                    y_ax = axis_arrays[i][::4]
                    X, Y = np.meshgrid(x_ax, y_ax)
                    prob = np.array(pdf(X.flatten(), Y.flatten())).reshape([L // 4, L // 4])
                    levels.append(prob.max())
                    levels = sorted(levels)
                    ax.contourf(X, Y, prob, levels=levels, cmap=cmap, alpha=0.7)
                    ax.contour(X, Y, prob, levels=levels, alpha=0.2)

                elif plot_style == "histogram":
                    # hexagonal-bin histogram
                    ax.set_facecolor(cmap(0))
                    ax.hexbin(x, y, gridsize=35, cmap=cmap)

                else:
                    # scatterplot
                    if point_colors is None:
                        ax.scatter(x, y, color=marginal_color, s=point_size)
                    else:
                        ax.scatter(x, y, c=point_colors, s=point_size, cmap=cmap)

                # plot any reference points if given
                if reference is not None:
                    ax.plot(
                        reference[j],
                        reference[i],
                        marker="o",
                        markersize=7,
                        markerfacecolor="none",
                        markeredgecolor="white",
                        markeredgewidth=3.5,
                    )
                    ax.plot(
                        reference[j],
                        reference[i],
                        marker="o",
                        markersize=7,
                        markerfacecolor="none",
                        markeredgecolor="red",
                        markeredgewidth=2,
                    )

            if initialiseAxes:
                # assign axis labels
                if i == N_par - 1:
                    ax.set_xlabel(parameter_labels[j], fontsize=label_size)
                if j == 0 and i != 0:
                    ax.set_ylabel(parameter_labels[i], fontsize=label_size)
                # impose x-limits on bottom row
                if i == N_par - 1:
                    ax.set_xlim(axis_limits[j])
                # impose y-limits on left column, except the top-left corner
                if j == 0 and i != 0:
                    ax.set_ylim(axis_limits[i])

                if show_ticks:  # set up ticks for the edge plots if they are to be shown
                    # hide x-tick labels for plots not on the bottom row
                    if i < N_par - 1:
                        plt.setp(ax.get_xticklabels(), visible=False)
                    # hide y-tick labels for plots not in the left column
                    if j > 0:
                        plt.setp(ax.get_yticklabels(), visible=False)
                    # remove all y-ticks for 1D marginal plots on the diagonal
                    if i == j:
                        ax.set_yticks([])
                else:  # else remove all ticks from all axes
                    ax.set_xticks([])
                    ax.set_yticks([])
        
        initialiseAxes = False

    # Set legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper right")
    
    # set the plot spacing
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    
    # save/show the figure if required
    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()

    return fig