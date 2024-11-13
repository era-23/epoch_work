from sdf_xarray import SDFPreprocess
from pathlib import Path
from scipy import constants
import matplotlib.pyplot as plt
import epydeck
import numpy as np
import xarray as xr
import argparse

def calculate_energy(directory : Path, irb : bool):

    # Read dataset
    ds = xr.open_mfdataset(
        str(directory / "*.sdf"),
        data_vars='minimal', 
        coords='minimal', 
        compat='override', 
        preprocess=SDFPreprocess()
    )

    # Drop initial conditions because they may not represent a solution
    ds = ds.sel(time=ds.coords["time"]>ds.coords["time"][0])

    # Read input deck
    input = {}
    with open(str(directory / "input.deck")) as id:
        input = epydeck.loads(id.read())

    density = input['constant']['background_density']
    frac_beam = input['constant']['frac_beam']
        
    proton_density = density * (1.0 - frac_beam) # m^-3
    electron_density = density # m^-3

    proton_E_array : xr.DataArray = ds['Derived_Average_Particle_Energy_proton']
    electron_E_array : xr.DataArray = ds['Derived_Average_Particle_Energy_electron']
    Bx_array : xr.DataArray = ds['Magnetic_Field_Bx']
    By_array : xr.DataArray = ds['Magnetic_Field_By']
    Bz_array : xr.DataArray = ds['Magnetic_Field_Bz']
    Ex_array : xr.DataArray = ds['Electric_Field_Ex']
    Ey_array : xr.DataArray = ds['Electric_Field_Ey']
    Ez_array : xr.DataArray = ds['Electric_Field_Ez']

    proton_KE_load = proton_E_array.load() # J
    electron_KE_load = electron_E_array.load() # J
    Bx_load = Bx_array.load() # T
    By_load = By_array.load() # T
    Bz_load = Bz_array.load() # T
    Ex_load = Ex_array.load() # V/m
    Ey_load = Ey_array.load() # V/m
    Ez_load = Ez_array.load() # V/m

    # Time conversion
    TWO_PI = 2.0 * np.pi
    B0 = input['constant']['b0_strength']

    ion_bkgd_mass = input['constant']['ion_mass_e'] * constants.electron_mass
    ion_bkgd_charge = input['species']['proton']['charge'] * constants.elementary_charge
    ion_gyroperiod = (TWO_PI * ion_bkgd_mass) / (ion_bkgd_charge * B0)
    t_in_ci = proton_KE_load.coords["time"] / ion_gyroperiod
    
    # Mean over all cells for mean particle/field energy
    pro_ke_mean : xr.DataArray = proton_KE_load.mean(dim = "X_Grid_mid") # J
    e_ke_mean : xr.DataArray = electron_KE_load.mean(dim = "X_Grid_mid") # J

    E_field = np.sqrt(Ex_load**2 + Ey_load**2 + Ez_load**2)
    E_E_density : xr.DataArray = (constants.epsilon_0 * E_field**2) / 2.0 # J / m^3
    E_mean : xr.DataArray = E_E_density.mean(dim="X_Grid_mid") # J / m^3

    B_field = np.sqrt(Bx_load**2 + By_load**2 + Bz_load**2)
    B_E_density : xr.DataArray = (B_field**2 / (2.0 * constants.mu_0)) # J / m^3
    B_mean : xr.DataArray = B_E_density.mean(dim = "X_Grid_mid") # J / m^3

    # Calculate B energy and convert others to to J/m3
    B_dE_density = B_mean - B_mean[0] # J / m^3
    E_dE_density = E_mean - E_mean[0] # J / m^3
    proton_dKE = pro_ke_mean - pro_ke_mean[0] # J
    electron_dKE = e_ke_mean - e_ke_mean[0] # J
    
    proton_dKE_density = proton_dKE * proton_density # J / m^3
    electron_dKE_density = electron_dKE * electron_density # J / m^3

    #total_dKE_density = proton_dKE_density + B_dE_density + E_dE_density
    total_dKE_density = proton_dKE_density + electron_dKE_density + B_dE_density + E_dE_density

    if irb:
        irb_density = density * frac_beam # m^-3
        irb_E_array : xr.DataArray = ds['Derived_Average_Particle_Energy_ion_ring_beam']
        irb_KE_load = irb_E_array.load() # J
        irb_ke_mean : xr.DataArray = irb_KE_load.mean(dim = "X_Grid_mid") # J
        irb_dKE = irb_ke_mean - irb_ke_mean[0] # J
        irb_dKE_density = irb_dKE * irb_density # J / m^3
        total_dKE_density += irb_dKE_density
        plt.plot(t_in_ci, irb_dKE_density.data, label = "ion ring beam KE")

    plt.plot(t_in_ci, proton_dKE_density.data, label = "background proton KE")
    plt.plot(t_in_ci, electron_dKE_density.data, label = "background electron KE")
    plt.plot(t_in_ci, B_dE_density.data, label = "Magnetic field E")
    plt.plot(t_in_ci, E_dE_density.data, label = "Electric field E")
    plt.plot(t_in_ci, total_dKE_density.data, label = "Total E")
    plt.xlabel("Time [ion_gyroperiods]")
    plt.ylabel("Change in energy density [J/m^3]")
    plt.title(f"{directory.name}: Evolution of energy in fast minority ions, background ions/electrons and EM fields")
    #plt.yscale("symlog")
    plt.legend()
    plt.grid()
    plt.show()

    e_KE_start = float(e_ke_mean[0].data)
    e_KE_end = float(e_ke_mean[-1].data)
    print(f"Change in electron energy density: e- KE t=start: {e_KE_start:.4f}, e- KE t=end: {e_KE_end:.4f} (+{((e_KE_end - e_KE_start)/e_KE_start)*100.0:.4f}%)")
        

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
    parser.add_argument(
        "--irb",
        action="store_true",
        help="Whether or not the ion ring beam should be accounted for or not.",
        required = False
    )

    args = parser.parse_args()

    calculate_energy(args.dir, args.irb)