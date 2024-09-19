import dask.array
import dask.array.rechunk
from sdf_xarray import sdf, SDFPreprocess
from pathlib import Path
from scipy import fftpack, constants
from matplotlib import colors
import dask
import numpy as np
import matplotlib.pyplot as plt
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

    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--dir",
        action="store",
        help="Directory containing all sdf files from simulation.",
        required = True,
        type=Path
    )
    args = parser.parse_args()

    directory = args.dir

    ds = xr.open_mfdataset(
        str(directory / "0[0123456]*.sdf"),
        data_vars='minimal', 
        coords='minimal', 
        compat='override', 
        preprocess=MySdfPreprocess()
    )

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
    B0 = 2.0
    bkgd_density = 1e19
    proton_mass = 1836.2 * constants.electron_mass
    proton_charge = constants.elementary_charge
    mass_density = (bkgd_density * proton_mass) + (bkgd_density * constants.electron_mass)
    alfven_velo = B0 / (np.sqrt(constants.mu_0 * mass_density))
    TWO_PI = 2.0 * np.pi
    ion_gyroperiod = (TWO_PI * proton_mass) / (proton_charge * B0)
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
    spec = spec.sel(freq_X_Grid_mid=spec.freq_X_Grid_mid<=120.0)
    spec.plot(norm=colors.LogNorm())
    plt.xlabel("Wavenumber [Wcp/Va]")
    plt.ylabel("Frequency [Wcp]")
    plt.title("Warm plasma dispersion relation")
    plt.show()