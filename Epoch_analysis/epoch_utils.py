import numpy as np
from scipy import constants

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


