from sdf_xarray import SDFPreprocess
from pathlib import Path
from scipy import constants
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import argparse
import xrft
import epydeck

class MySdfPreprocess(SDFPreprocess):
    def __init__(self):
        SDFPreprocess.__init__(self)

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        if self.job_id is None:
            self.job_id = ds.attrs["jobid1"]

        if self.job_id != ds.attrs["jobid1"]:
            raise ValueError(
                f"Mismatching job ids (got {ds.attrs['jobid1']}, expected {self.job_id})"
            )
        
        # ds_clean = ds.drop_dims([d for d in ds.dims if ("px" in d or "py" in d or "pz" in d)])

        # new_ds = xr.combine_by_coords(ds_clean, data_vars="minimal", combine_attrs="drop_conflicts")

        #new_ds = ds_clean.expand_dims(time=[ds_clean.attrs["time"]])
        new_ds = ds.expand_dims(time=[ds.attrs["time"]])

        # Particles' spartial coordinates also evolve in time
        for coord, value in new_ds.coords.items():
            if "Particles" in coord:
                new_ds.coords[coord] = value.expand_dims(time=[new_ds.attrs["time"]])

        return new_ds.drop_dims([c for c in new_ds.coords if ("px" in c or "py" in c or "pz" in c)])

#def calculate_CPDR_perp():


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
        "--maxK",
        action="store",
        help="Max value of k to plot on the x-axis.",
        required = True,
        type=float
    )
    args = parser.parse_args()

    directory = args.dir

    ds = xr.open_mfdataset(
        str(directory / "*.sdf"),
        data_vars='minimal', 
        coords='minimal', 
        compat='override', 
        preprocess=MySdfPreprocess()
    )

    # Load input.deck content
    deck = dict()
    with open(args.dir / "input.deck") as inputDeck:
        deck = epydeck.load(inputDeck)

    data_to_plot = []
    # electric_field_ex : xr.DataArray = ds["Electric Field/Ex"]
    # electric_field_ey : xr.DataArray = ds["Electric Field/Ey"]
    # electric_field_ez : xr.DataArray = ds["Electric Field/Ez"]
    magnetic_field_by : xr.DataArray = ds["Magnetic Field/By"]
    magnetic_field_bz : xr.DataArray = ds["Magnetic Field/Bz"]
    # charge_density_all : xr.DataArray = ds["Derived/Charge_Density"]
    # charge_density_irb : xr.DataArray = ds["Derived/Charge_Density/ion_ring_beam"]
    # charge_density_p : xr.DataArray = ds["Derived/Charge_Density/proton"]
    # charge_density_e : xr.DataArray = ds["Derived/Charge_Density/electron"]

    # data_to_plot.append(magnetic_field_by)
    # data_to_plot.append(magnetic_field_bz)
    # data_to_plot.append(electric_field_ex)
    # data_to_plot.append(electric_field_ey)
    # data_to_plot.append(electric_field_ez)
    # data_to_plot.append(charge_density_irb)
    # data_to_plot.append(charge_density_p)
    # data_to_plot.append(charge_density_e)
    # data_to_plot.append(charge_density_all)
    
    # data_to_plot.append(pos_charge)
    # data_to_plot.append(all_charge)

    # for data in data_to_plot:
    #     x = plt.figure()
    #     data.plot()
    #     x.show()

    # plt.show()

    #magnetic_field_by_c = magnetic_field_by.chunk({'time': magnetic_field_by.sizes['time']})
    data_load = magnetic_field_bz.load()
    
    B0 = 0.0
    for direction in deck["fields"]:
        B0 += deck["fields"][direction]**2
    B0 = np.sqrt(B0)

    bkgd_density = deck["constant"]["background_density"]
    bkgd_temp = deck["constant"]["background_temp"]
    ion_mass = constants.proton_mass
    ion_charge = constants.elementary_charge

    if "deuteron" in deck["species"]:
        ion_mass = constants.proton_mass + constants.neutron_mass
        ion_charge *= deck["species"]["deuteron"]["charge"]
    
    mass_density = (bkgd_density * ion_mass) + (bkgd_density * constants.electron_mass)
    alfven_velo = B0 / (np.sqrt(constants.mu_0 * mass_density))
    thermal_velo = np.sqrt((2.0 * constants.k * bkgd_temp) / ion_mass)
    
    omega_pe_squared = (bkgd_density * constants.elementary_charge**2)/(constants.electron_mass * constants.epsilon_0)
    omega_pi_squared = (bkgd_density * constants.elementary_charge**2)/(ion_mass * constants.epsilon_0)
    omega_p_squared = omega_pe_squared + omega_pi_squared
    TWO_PI = 2.0 * np.pi
    ion_gyroperiod = (TWO_PI * ion_mass) / (ion_charge * B0)
    Tci = data_load.coords["time"] / ion_gyroperiod
    vA_Tci = data_load.coords["X_Grid_mid"] / (ion_gyroperiod * alfven_velo)
    #vTh_over_Wci = ds.coords["X_Grid_mid"] * (TWO_PI / B0)
    #Tci = np.array(ds.coords["time"]*B0/TWO_PI)
    data_scaled = xr.DataArray(data_load, coords=[Tci, vA_Tci], dims=["time", "X_Grid_mid"])
    orig_spec : xr.DataArray = xrft.xrft.fft(data_scaled, true_amplitude=True, true_phase=True)
    spec = abs(orig_spec)
    spec = spec.sel(freq_time=spec.freq_time>=0.0)
    #spec = spec.sel(freq_time=spec.freq_time<=30.0)
    spec = spec.sel(freq_X_Grid_mid=spec.freq_X_Grid_mid>=0.0)
    spec = spec.sel(freq_X_Grid_mid=spec.freq_X_Grid_mid<=args.maxK)
    print(f"Max: {np.max(spec)}")
    print(f"Min: {np.min(spec)}")
    spec.plot(norm=colors.LogNorm())
    #spec.plot()
    omega_A = spec.freq_X_Grid_mid

    # Broken, fix
    #omega_magnetosonic = ((spec.freq_X_Grid_mid * (ion_gyroperiod * alfven_velo))**2 * constants.speed_of_light**2 * ((thermal_velo**2 + alfven_velo**2)/(constants.speed_of_light**2 + alfven_velo**2))) / ion_gyroperiod
    cVa = constants.speed_of_light / alfven_velo
    plt.plot(spec.freq_X_Grid_mid, omega_A, "k--", label="w = kVa")
    plt.plot(spec.freq_X_Grid_mid, spec.freq_X_Grid_mid * cVa, "r--", label="w = kc")
    # plt.plot(spec.freq_X_Grid_mid, omega_magnetosonic, "k:")
    plt.xlabel("Wavenumber [Wcp/Va]")
    plt.ylabel("Frequency [Wcp]")
    plt.title("Warm plasma dispersion relation")
    plt.legend()
    plt.show()