from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import netCDF4 as nc
import xrft  # noqa: E402
from scipy import constants
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
class SimulationLinearGrowthRate:
    simulation : str
    gamma: float
    time : float
    yIntercept: float
    residual: float
    wavenumber: float
    peakPower : float
    totalPower : float

@dataclass
class LinearGrowthRateByK:
    wavenumber: float
    gamma: float
    yIntercept: float
    residual: float

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

def calculate_lower_hybrid_frequency(
        ion_charge : float,
        ion_mass : float,
        ion_density : float,
        B0 : float,
        unit : str = 'cyc'
    ):
    """Calculate the lower hybrid frequency for purely perpendicular propagation. Assumes plasma frequency >> ion cyclotron frequency.

    Parameters:
        ion_charge (float): Charge of the ion species
        ion_mass (float): Mass of the ion species
        ion_density (float): Background number density of the ion species
        B0 (float): Strength of the background magnetic field
        unit (str): 'si' or 'cyc' -- Unit to be returned, 'si' for rad/s or 'cyc' for ion cyclotron frequency. Default is cyc

    Returns:
        w_LH (float): The lower hybrid frequency in SI or ion cyclotron frequency
    """

    #TWO_PI = 2.0 * np.pi
    plasma_freq_ion = np.sqrt((ion_density * ion_charge**2.0)/(ion_mass * constants.epsilon_0))
    cyclotron_freq_ion = (ion_charge * B0) / (ion_mass)
    cyclotron_freq_electron = (constants.elementary_charge * B0) / (constants.electron_mass)

    w_LH = 1.0 / np.sqrt((1.0 / plasma_freq_ion**2.0) + (1.0 / (cyclotron_freq_electron * cyclotron_freq_ion)))

    if unit == 'cyc':
        w_LH = w_LH / cyclotron_freq_ion

    return w_LH

def create_t_k_spectrum(
        originalFftSpectrum : xr.DataArray, 
        statsFile : nc.Dataset = None,
        maxK : float = None,
        load : bool = True,
        debug = False
) -> xr.DataArray :
    
    tk_spec = originalFftSpectrum.where(originalFftSpectrum.frequency>0.0, 0.0)
    original_zero_freq_amplitude = tk_spec.sel(wavenumber=0.0)
    # Double spectrum to conserve E
    tk_spec = np.sqrt(2.0) * tk_spec # <---- Should this be sqrt(2)?
    tk_spec.loc[dict(wavenumber=0.0)] = original_zero_freq_amplitude # Restore original 0-freq amplitude
    tk_spec = xrft.xrft.ifft(tk_spec, dim="frequency")
    tk_spec = tk_spec.rename(freq_frequency="time")
    tk_spec = abs(tk_spec)

    if statsFile is not None:
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

def create_t_k_plot(
        tkSpectrum : xr.DataArray,
        field : str,
        field_unit : str,
        saveDirectory : Path = None,
        runName : str = "defaultRun",
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
    if saveDirectory is not None:
        filename = Path(f'{runName}_{field.replace("_", "")}_tk_log-{log}_maxK-{maxK if maxK is not None else "all"}.png')
        fig.savefig(str(saveDirectory / filename))
    if display:
        plt.show()
    plt.close('all')