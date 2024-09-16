import dask.array
import dask.array.rechunk
from sdf_xarray import sdf, SDFPreprocess
from pathlib import Path
from scipy import fftpack
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
    # Run python setup.py -h for list of possible arguments
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
        str(directory / "*.sdf"),
        data_vars='minimal', 
        coords='minimal', 
        compat='override', 
        preprocess=MySdfPreprocess()
    )

    data_to_plot = []
    electric_field_ex : xr.DataArray = ds["Electric Field/Ex"]
    electric_field_ey : xr.DataArray = ds["Electric Field/Ey"]
    electric_field_ez : xr.DataArray = ds["Electric Field/Ez"]
    magnetic_field_bx : xr.DataArray = ds["Magnetic Field/Bx"]
    magnetic_field_by : xr.DataArray = ds["Magnetic Field/By"]
    magnetic_field_bz : xr.DataArray = ds["Magnetic Field/Bz"]
    charge_density_all : xr.DataArray = ds["Derived/Charge_Density"]
    charge_density_irb : xr.DataArray = ds["Derived/Charge_Density/ion_ring_beam"]
    charge_density_p : xr.DataArray = ds["Derived/Charge_Density/proton"]
    charge_density_e : xr.DataArray = ds["Derived/Charge_Density/electron"]

    data_to_plot.append(magnetic_field_bx)
    data_to_plot.append(magnetic_field_by)
    data_to_plot.append(magnetic_field_bz)
    data_to_plot.append(electric_field_ex)
    data_to_plot.append(electric_field_ey)
    data_to_plot.append(electric_field_ez)
    data_to_plot.append(charge_density_irb)
    data_to_plot.append(charge_density_p)
    data_to_plot.append(charge_density_e)
    data_to_plot.append(charge_density_all)
    # data_to_plot.append(pos_charge)
    # data_to_plot.append(all_charge)

    # for data in data_to_plot:
    #     x = plt.figure()
    #     data.plot()
    #     x.show()

    # plt.show()

    #magnetic_field_by_c = magnetic_field_by.chunk({'time': magnetic_field_by.sizes['time']})
    print(charge_density_p.load())
    print(charge_density_irb.load())
    all_pos = charge_density_p.load() + charge_density_irb.load()
    print(all_pos)

    data_load = charge_density_all.load()
    #data = data - np.mean(data)
    print(magnetic_field_bz.load())
    Bz = np.mean(magnetic_field_bz.load(), axis=1)
    TWO_PI = 2.0 * np.pi
    B0z = np.mean(Bz) # THINK ABOUT THIS, UNSURE IF (MOST) CORRECT
    vTh_over_Wci = ds.coords["X_Grid_mid"] * (TWO_PI / B0z)
    Tci = np.array(ds.coords["time"]*Bz/TWO_PI)
    data = xr.DataArray(data_load - np.mean(data_load), coords=[Tci, vTh_over_Wci], dims=["time", "X_Grid_mid"])
    orig_spec : xr.DataArray = xrft.xrft.fft(data, true_amplitude=True, true_phase=True)
    spec = abs(orig_spec)
    spec = spec.sel(freq_time=spec.freq_time>=0.0)
    spec.plot()
    plt.show()

    #b0z = dataset.B0z
    #vTh_over_Wci = np.arange(0.0, dataset.numCells) * (TWO_PI / b0z)
    #Tci = np.array(dataset.outputTimes*b0z/TWO_PI)
    #charge_density = xr.DataArray(dataset.chargeDensity, coords=[Tci, vTh_over_Wci], dims=["time", "space"])
    #charge_density_irb = charge_density_irb.chunk({'time': 200})
    #charge_density_irb = charge_density_irb.as_numpy()
    #og_spectrum: xr.DataArray = xrft.xrft.fft(charge_density_irb, true_amplitude=False, true_phase=False)
    #spectrum = abs(og_spectrum)
    #spectrum.plot()
    #plt.show()

    #fwd_time_spectrum = spectrum.sel(freq_time=spectrum.freq_time>=0.0)
    #k_vTh_over_Wci = fwd_time_spectrum.freq_space.data * ((4.0 * np.pi**2) / (b0z**2))
    #fwd_time_spectrum = fwd_time_spectrum.assign_coords(k_perp=("freq_space", k_vTh_over_Wci))
    #fwd_time_spectrum.plot()
    #fwd_time_spectrum.plot(size=9, x="k_perp", y = "freq_time", cbar_kwargs={"label": "CHARGE DENSITY"})
    #plt.title("FFT of charge density in cell space and (+ve) time", fontsize=20)
    #plt.yticks(fontsize=24)
    #plt.xticks(fontsize=24)
    #plt.ylabel("$\omega/\omega_{ci}$", fontsize=32)
    #plt.xlabel("$k/k_{\perp}v_{th,i}\omega_{ci}^{-1}$", fontsize=30)
    #plt.grid()
    #plt.show()