from sdf_xarray import SDFPreprocess
from pathlib import Path
from scipy import constants
from matplotlib import colors
import matplotlib.pyplot as plt
import epydeck
import numpy as np
import xarray as xr
import argparse
import xrft

def calculate_energy(directory : Path):

    # Read dataset
    ds = xr.open_mfdataset(
        str(directory / "*.sdf"),
        data_vars='minimal', 
        coords='minimal', 
        compat='override', 
        preprocess=SDFPreprocess()
    )

    # Read input deck
    input = {}
    with open(str(directory / "input.deck")) as id:
        input = epydeck.loads(id.read())

    density = input['constant']['background_density']
    frac_beam = input['constant']['frac_beam']
    irb_density = density * frac_beam
    proton_density = density * (1.0 - frac_beam)

    irb_E_array : xr.DataArray = ds['Derived/Average_Particle_Energy/ion_ring_beam']
    proton_E_array : xr.DataArray = ds['Derived/Average_Particle_Energy/proton']
    B_array : xr.DataArray = ds['Magnetic Field/Bz']

    irb_kE_load = irb_E_array.load()
    proton_kE_load = proton_E_array.load()
    B_E_load = B_array.load()

    # Sum over all cells for particle energy, mean for field energy density
    irb_ke_sum : xr.DataArray = irb_kE_load.sum(dim = "X_Grid_mid")
    pro_ke_sum : xr.DataArray = proton_kE_load.sum(dim = "X_Grid_mid")
    

    b_energy : xr.DataArray = (B_E_load**2 / (2.0 * constants.mu_0))
    b_E_mean : xr.DataArray = b_energy.mean(dim = "X_Grid_mid")

    cell_spacing = B_E_load.coords["X_Grid_mid"][1] - B_E_load.coords["X_Grid_mid"][0]
    simLength = B_E_load.coords["X_Grid_mid"][-1] + (0.5 * cell_spacing)

    # Calculate B energy and convert others to to J/m3
    B_dE_density = b_E_mean - b_E_mean[0]
    irb_dkE = irb_ke_sum - irb_ke_sum[0]
    proton_dkE = pro_ke_sum - pro_ke_sum[0]

    irb_ke_density = (irb_dkE / simLength**3) * irb_density
    proton_ke_density = (proton_dkE / simLength*3) * proton_density

    # Time conversion
    TWO_PI = 2.0 * np.pi
    B0 = input['constant']['b0_strength']

    ion_bkgd_mass = input['constant']['ion_mass_e'] * constants.electron_mass
    ion_bkgd_charge = input['species']['proton']['charge'] * constants.elementary_charge
    ion_gyroperiod = (TWO_PI * ion_bkgd_mass) / (ion_bkgd_charge * B0)
    t_in_ci = proton_ke_density.coords["time"] / ion_gyroperiod

    plt.plot(t_in_ci, irb_ke_density.data, label = "ion ring beam KE")
    plt.plot(t_in_ci, proton_ke_density.data, label = "background proton KE")
    plt.plot(t_in_ci, B_energy_density.data, label = "Magnetic field E")
    plt.xlabel("Time [ion_gyroperiods]")
    plt.ylabel("Change in energy density [J/m3]")
    plt.title("Evolution of energy in fast minority ions, background ions and B-field")
    #plt.yscale("symlog")
    plt.legend()
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--dir",
        action="store",
        help="Directory containing all sdf files from simulation.",
        required = True,
        type=Path
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Plot FFT with log of spectral power.",
        required = False
    )
    parser.add_argument(
        "--savePath",
        action="store",
        help="Directory to which figures should be saved.",
        required = False,
        type=Path
    )

    args = parser.parse_args()

    calculate_energy(args.dir)