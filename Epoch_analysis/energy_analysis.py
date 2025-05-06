import glob
from sdf_xarray import SDFPreprocess
from pathlib import Path
from scipy import constants
import matplotlib.pyplot as plt
import epydeck
import numpy as np
import xarray as xr
import argparse

def calculate_energy(directory : Path, irb : bool, pct : bool):

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

    plt.rcParams.update({'axes.labelsize': 16})
    plt.rcParams.update({'axes.titlesize': 18})
    plt.rcParams.update({'xtick.labelsize': 14})
    plt.rcParams.update({'ytick.labelsize': 14})

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
        plt.plot(t_in_ci, irb_dKE_density.data, label = r"ion ring beam KE")

    plt.plot(t_in_ci, proton_dKE_density.data, label = r"background proton KE")
    plt.plot(t_in_ci, electron_dKE_density.data, label = r"background electron KE")
    plt.plot(t_in_ci, B_dE_density.data, label = r"Magnetic field E")
    plt.plot(t_in_ci, E_dE_density.data, label = r"Electric field E")
    plt.plot(t_in_ci, total_dKE_density.data, label = r"Total E")
    plt.xlabel(r'Time [$\tau_{ci}$]')
    plt.ylabel(r"Change in energy density [$J/m^3$]")
    plt.title(f"{directory.name}: Evolution of energy in fast minority ions, background ions/electrons and EM fields")
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()

    if pct:
        # Calculate B energy and convert others to to J/m3
        B_dE_pct = 100.0 * (B_mean - B_mean[0])/B_mean[0] # %
        E_dE_pct = 100.0 * (E_mean - E_mean[0])/E_mean[0] # %
        proton_dKE_density = pro_ke_mean * proton_density # J / m^3
        electron_dKE_density = e_ke_mean * electron_density # J / m^3
        proton_dKE_pct = 100.0 * (proton_dKE_density - proton_dKE_density[0])/proton_dKE_density[0] # %
        electron_dKE_pct = 100.0 * (electron_dKE_density - electron_dKE_density[0])/electron_dKE_density[0] # %
        total_E_density = B_mean + E_mean + proton_dKE_density + electron_dKE_density

        if irb:
            irb_dKE_density = irb_ke_mean * irb_density # J / m^3
            irb_dKE_pct = 100.0 * (irb_dKE_density - irb_dKE_density[0])/irb_dKE_density[0] # J
            plt.plot(t_in_ci, irb_dKE_pct.data, label = "ion ring beam KE")
            total_E_density += irb_dKE_density
            #total_dE_pct  = np.mean([B_dE_pct, E_dE_pct, proton_dKE_pct, electron_dKE_pct, irb_dKE_pct], axis=0)
        #else:
            #total_dE_pct  = np.mean(B_dE_pct + E_dE_pct + proton_dKE_pct + electron_dKE_pct, axis=0)

        total_dE_density_pct = 100.0 * (total_E_density-total_E_density[0])/total_E_density[0]
        
        plt.plot(t_in_ci, proton_dKE_pct.data, label = "background proton KE")
        plt.plot(t_in_ci, electron_dKE_pct.data, label = "background electron KE")
        plt.plot(t_in_ci, B_dE_pct.data, label = "Magnetic field E")
        plt.plot(t_in_ci, E_dE_pct.data, label = "Electric field E")
        plt.plot(t_in_ci, total_dE_density_pct, label = "Total E")
        plt.yscale('symlog')
        plt.xlabel(r'Time [$\tau_{ci}$]')
        plt.ylabel("Percentage change in energy density [%]")
        plt.title(f"{directory.name}: Evolution of energy in fast minority ions, background ions/electrons and EM fields")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
            
    e_KE_start = float(e_ke_mean[0].data)
    e_KE_end = float(e_ke_mean[-1].data)
    print(f"Change in electron energy density: e- KE t=start: {e_KE_start:.4f}, e- KE t=end: {e_KE_end:.4f} (+{((e_KE_end - e_KE_start)/e_KE_start)*100.0:.4f}%)")
    E_start = total_E_density[0]
    E_end = total_E_density[-1]
    print(f"Change in overall energy density: E t=start: {E_start:.4f}, E t=end: {E_end:.4f} (+{((E_end - E_start)/E_start)*100.0:.4f}%)")
        
def analyse_electron_heating(analysisDirectory : Path):

    electronDeltaPct = []
    cellWidth_rLe = []
    cellWidth_dl = []
    totalConservationPct = []

    dataFiles = glob.glob(str(analysisDirectory / "*.nc"))
    
    for simulation in dataFiles:

        data = xr.open_datatree(
            simulation,
            engine="netcdf4"
        )

        cellWidth_dl.append(data.attrs["cellWidth_dL"])
        cellWidth_rLe.append(data.attrs["cellWidth_rLe"])

        energyStats = data["Energy"]
        #electronDeltaEnergy = ((energyStats.electronEnergyDensity_end - energyStats.electronEnergyDensity_start) / energyStats.electronEnergyDensity_start) * 100.0
        electronDeltaEnergy = 100.0 * (energyStats.electronEnergyDensity_delta/energyStats.electronEnergyDensity_start)
        # electronDeltaEnergy = energyStats.electronEnergyDensity_delta
        electronDeltaPct.append(electronDeltaEnergy)
        totalConservationPct.append(energyStats.totalEnergyDensityConservation_pct)

        print(f"{simulation.split('/')[-1]}: cell width: {data.attrs['cellWidth_dL']:.4f} Debye lengths/{data.attrs['cellWidth_rLe']:.4f} electron gyroradii (B0: {data.attrs['B0strength']:.4f}, background density: {data.attrs['backgroundDensity']:.4E}), electron delta energy: {electronDeltaEnergy:.4f}%, total energy change: {energyStats.totalEnergyDensityConservation_pct:.4f}")
    
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax2 = ax1.twiny()
    ax1.scatter(cellWidth_rLe, electronDeltaPct, color="blue")
    # res = linregress(cellWidth_rLe, electronDeltaPct)
    ax2.scatter(cellWidth_dl, electronDeltaPct, color="blue")
    # x = np.linspace(0.0, np.max(cellWidth_rLe), 1000)
    # plt.plot(x, res.intercept + res.slope*x, "g--", label=f"R^2 = {res.rvalue**2:.5f}")
    # plt.yscale("log")
    # plt.legend()
    ax1.set_xlabel("Cell width / electron gyroradii")
    ax2.set_xlabel("Cell width / Debye lengths")
    plt.title("Electrons - run 33 (B0: 0.9549, background density: 7.7898E+19)")
    ax1.set_ylabel("Change in energy density by end of simulation / %")
    plt.tight_layout()
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax2 = ax1.twiny()
    # res = linregress(cellWidth_rLe, totalConservationPct)
    ax1.scatter(cellWidth_rLe, totalConservationPct, color="orange")
    ax2.scatter(cellWidth_dl, totalConservationPct, color="orange")
    # x = np.linspace(0.0, np.max(cellWidth_rLe), 1000)
    # plt.plot(x, res.intercept + res.slope*x, "r--", label=f"R^2 = {res.rvalue**2:.5f}")
    # plt.yscale("log")
    ax1.set_xlabel("Cell width / electron gyroradii")
    ax2.set_xlabel("Cell width / Debye lengths")
    ax1.set_ylabel("Change in energy density by end of simulation / %")
    plt.title("Total energy - (B0: 0.9549, background density: 7.7898E+19)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--dir",
        action="store",
        help="Directory.",
        required = True,
        type=Path
    )
    parser.add_argument(
        "--electronHeating",
        action="store_true",
        help="Analyse electron heating.",
        required = False
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
    parser.add_argument(
        "--pct",
        action="store_true",
        help="Calculate percentages.",
        required = False
    )

    args = parser.parse_args()

    if args.electronHeating:
        analyse_electron_heating(args.dir)
    else:
        calculate_energy(args.dir, args.irb, args.pct)