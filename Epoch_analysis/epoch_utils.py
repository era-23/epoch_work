import numpy as np
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


