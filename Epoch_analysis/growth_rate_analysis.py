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

def calculate_all_growth_rates_in_run(directory : Path, field : str, normalise : bool = False, plotTk : bool = True, maxK = None, numKs = 5, log = False, window : str = None):

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

    field_data_array : xr.DataArray = ds[field]

    field_data = field_data_array.load()

    # Check Nyquist frequencies in SI
    num_t = len(field_data.coords["time"])
    print(f"Num time points: {num_t}")
    sim_time = float(field_data.coords["time"][-1])
    print(f"Sim time in SI: {sim_time}s")
    print(f"Sampling frequency: {num_t/sim_time}Hz")
    print(f"Nyquist frequency: {num_t/(2.0 *sim_time)}Hz")
    num_cells = len(field_data.coords["X_Grid_mid"])
    print(f"Num cells: {num_cells}")
    sim_L = float(field_data.coords["X_Grid_mid"][-1])
    print(f"Sim length: {sim_L}m")
    print(f"Sampling frequency: {num_cells/sim_L}m^-1")
    print(f"Nyquist frequency: {num_cells/(2.0 *sim_L)}m^-1")

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

    # Nyquist frequencies in normalised units
    simtime_Tci = float(Tci[-1])
    print(f"NORMALISED: Sim time in Tci: {simtime_Tci}Tci")
    print(f"NORMALISED: Sampling frequency in Wci: {num_t/simtime_Tci}Wci")
    print(f"NORMALISED: Nyquist frequency in Wci: {num_t/(2.0 *simtime_Tci)}Wci")
    simL_vATci = float(vA_Tci[-1])
    print(f"NORMALISED: Sim L in vA*Tci: {simL_vATci}vA*Tci")
    print(f"NORMALISED: Sampling frequency in Wci/vA: {num_cells/simL_vATci}Wci/vA")
    print(f"NORMALISED: Nyquist frequency in Wci/vA: {num_cells/(2.0 *simL_vATci)}Wci/vA")

    # Clean data and take FFT
    data = xr.DataArray(field_data if not normalise else field_data - abs(np.mean(field_data)), coords=[Tci, vA_Tci], dims=["time", "X_Grid_mid"])
    orig_spec : xr.DataArray = xrft.xrft.fft(data, true_amplitude=True, true_phase=True, window=None)

    # Create t-k spectrum
    zeroed_spec = orig_spec.where(orig_spec.freq_time>0.0, 0.0)
    zeroed_doubled_spec = 2.0 * zeroed_spec # Double spectrum to conserve E
    zeroed_doubled_spec.loc[dict(freq_X_Grid_mid=0.0)] = zeroed_spec.sel(freq_X_Grid_mid=0.0) # Restore original 0-freq amplitude
    spec_tk = xrft.xrft.ifft(zeroed_doubled_spec, dim="freq_time")
    spec_tk : xr.DataArray = abs(spec_tk)
    # spec_cellSpace = spec_cellSpace.assign_coords(k_perp=("freq_X_Grid_mid", vA_Tci))

    # Plot t-k
    if plotTk:
        spec_tk_plot = spec_tk
        if args.maxK is not None:
            spec_tk_plot = spec_tk.sel(freq_X_Grid_mid=spec_tk.freq_X_Grid_mid<=maxK)
            spec_tk_plot = spec_tk_plot.sel(freq_X_Grid_mid=spec_tk_plot.freq_X_Grid_mid>=-maxK)
        if log:
            spec_tk_plot.plot(size=9, x = "freq_X_Grid_mid", y = "time", norm=colors.LogNorm())
        else:
            spec_tk_plot.plot(size=9, x = "freq_X_Grid_mid", y = "time")
        plt.xlabel("Wavenumber [Wci/VA]")
        plt.ylabel("Time [Wci^-1]")
        plt.show()

    sum_over_all_t = np.sum(spec_tk, axis=0)
    max_power_ks = np.argpartition(sum_over_all_t, -numKs)[-numKs:]

    # Calculate growth rates by k
    for i in max_power_ks:
        plt.plot(spec_tk.coords["time"], spec_tk[:,i], label = f"k = {float(spec_tk.coords['freq_X_Grid_mid'][i])}")
    #plt.yscale("log")
    plt.xlabel("Time/Wci^-1")
    plt.ylabel(f"Spectral power [{field_data_array.units}]")
    plt.title(f"Time evolution of {numKs} highest power wavenumbers in {field}")
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
        "--field",
        action="store",
        help="Simulation field to use in Epoch output format, e.g. \'Derived/Charge_Density\', \'Electric Field/Ex\', \'Magnetic Field/Bz\'",
        required = True,
        type=str
    )
    parser.add_argument(
        "--norm",
        action="store_true",
        help="Switch to normalise data before plotting.",
        required = False
    )
    parser.add_argument(
        "--plotTk",
        action="store_true",
        help="Switch to normalise data before plotting.",
        required = False
    )
    parser.add_argument(
        "--maxK",
        action="store",
        help="Max value of k for plotting. Growth rates of all k will still be calculated. Defaults to all k.",
        required = False,
        type=float
    )
    parser.add_argument(
        "--numKs",
        action="store",
        help="Number of wavenumbers to plot in time evolution. Defaults to 5.",
        required = False,
        type=int
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Plot FFT with log of spectral power.",
        required = False
    )
    args = parser.parse_args()

    defaultNumK = 5
    numKs = args.numKs if not None else defaultNumK

    calculate_all_growth_rates_in_run(args.dir, args.field, args.norm, args.plotTk, args.maxK, numKs, args.log)