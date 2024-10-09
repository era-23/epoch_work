from sdf_xarray import SDFPreprocess
from pathlib import Path
from scipy import constants
import epydeck
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import xarray as xr
import argparse
import xrft

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


if __name__ == "__main__":
    # Run python setup.py -h for list of possible arguments
    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--dir",
        action="store",
        help="Directory containing all sdf files from simulation.",
        required = True,
        type=Path
    )
    parser.add_argument(
        "--fft",
        action="store",
        help="Quantity to fft.",
        required = True,
        type=str
    )
    parser.add_argument(
        "--normalise",
        action="store_true",
        help="Normalise data around 0?",
        required = False
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Log plot",
        required = False
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
    quantity = args.fft
    norm = args.normalise
    logplot = args.log

    # Read dataset
    ds = xr.open_mfdataset(
        str(directory / "*.sdf"),
        data_vars='minimal', 
        coords='minimal', 
        compat='override', 
        preprocess=MySdfPreprocess()
    )

    # Read input deck
    input = {}
    with open(str(directory / "input.deck")) as id:
        input = epydeck.loads(id.read())

    data_to_plot = []
    # electric_field_ex : xr.DataArray = ds["Electric Field/Ex"]
    # electric_field_ey : xr.DataArray = ds["Electric Field/Ey"]
    # electric_field_ez : xr.DataArray = ds["Electric Field/Ez"]
    # magnetic_field_bx : xr.DataArray = ds["Magnetic Field/Bx"]
    # magnetic_field_by : xr.DataArray = ds["Magnetic Field/By"]
    # magnetic_field_bz : xr.DataArray = ds["Magnetic Field/Bz"]
    # charge_density_all : xr.DataArray = ds["Derived/Charge_Density"]
    # charge_density_irb : xr.DataArray = ds["Derived/Charge_Density/ion_ring_beam"]
    # charge_density_p : xr.DataArray = ds["Derived/Charge_Density/proton"]
    # charge_density_e : xr.DataArray = ds["Derived/Charge_Density/electron"]

    quantity_array : xr.DataArray = ds[quantity]

    # data_to_plot.append(magnetic_field_bx)
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

    data_load = quantity_array.load()

    TWO_PI = 2.0 * np.pi
    B0 = input['constant']['b_strength']

    electron_mass = input['species']['electron']['mass'] * constants.electron_mass
    ion_bkgd_mass = input['constant']['ion_mass_e'] * constants.electron_mass 
    ion_ring_mass = input['constant']['ion_mass_e'] * constants.electron_mass # Note: assumes background and ring beam ions are then same species
    ion_ring_frac = input['constant']['frac_beam']

    mass_density = input['constant']['background_density'] * (electron_mass + ion_bkgd_mass + (ion_ring_frac * ion_ring_mass))

    ion_ring_charge = input['species']['ion_ring_beam']['charge'] * constants.elementary_charge
    
    ion_gyroperiod = (TWO_PI * ion_ring_mass) / (ion_ring_charge * B0)
    Tci = ds.coords["time"] / ion_gyroperiod
    alfven_velo = B0 / (np.sqrt(constants.mu_0 * mass_density))
    vA_Tci = ds.coords["X_Grid_mid"] / (ion_gyroperiod * alfven_velo)
    
    print(np.mean(data_load))
    data = xr.DataArray(data_load if not norm else data_load - abs(np.mean(data_load)), coords=[Tci, vA_Tci], dims=["time", "X_Grid_mid"])
    orig_spec : xr.DataArray = xrft.xrft.fft(data, true_amplitude=True, true_phase=True)
    spec = abs(orig_spec)
    
    # Positive times
    spec = spec.sel(freq_time=spec.freq_time>=0.0)
    # Positive wavenumbers
    spec = spec.sel(freq_X_Grid_mid=spec.freq_X_Grid_mid>=0.0)
    spec = spec.sel(freq_X_Grid_mid=spec.freq_X_Grid_mid<=args.maxK)
    if logplot:
        spec.plot(norm=colors.LogNorm())
    else:
        spec.plot()
    if norm:
        plt.title(f"{quantity} (normalised) - {directory.name}")
    else:
        plt.title(f"{quantity} - {directory.name}")
    plt.xlabel("Wavenumber [Wcp/Va]")
    plt.ylabel("Frequency [Wcp]")
    plt.show()