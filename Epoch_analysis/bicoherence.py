from sdf_xarray import SDFPreprocess
from pathlib import Path
from scipy import constants
from spectrum import bicoherence, plot_bicoherence, bicoherencex, plot_bicoherencex, bispectrumd, bispectrumi, plot_bispectrumd, plot_bispectrumi
from matplotlib import pyplot as plt
import epydeck
import numpy as np
import xarray as xr
import argparse
import xrft
from typing import Tuple, Any, Union

def my_plot_bispectrumd(
    Bspec: np.ndarray[Any, np.dtype[Any]],
    waxis: np.ndarray[Any, np.dtype[Any]],
    axis_label: str,
    grid: bool
) -> None:
    """
    Plot the bispectrum estimate.

    Parameters:
    -----------
    Bspec : np.ndarray[Any, np.dtype[Any]]
        Estimated bispectrum array.
    waxis : np.ndarray[Any, np.dtype[Any]]
        Frequency axis values.

    Returns:
    --------
    None
    """
    cont = plt.contourf(waxis, waxis, abs(Bspec), 100, cmap="viridis")
    plt.colorbar(cont)
    plt.title("Bispectrum estimated via the direct (FFT) method")
    plt.xlabel(axis_label)
    plt.ylabel(axis_label)
    if grid:
        plt.grid()
    plt.show()

def plot_bicoherence_spectra(directory : Path, field : str, time_pt : float, plotTk : bool, irb : bool = True):

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

    field_data_array : xr.DataArray = ds[field]

    field_data = field_data_array.load()

    # Read input deck
    input = {}
    with open(str(directory / "input.deck")) as id:
        input = epydeck.loads(id.read())

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
    B0 = input['constant']['b0_strength']

    electron_mass = input['species']['electron']['mass'] * constants.electron_mass
    ion_bkgd_mass = input['constant']['ion_mass_e'] * constants.electron_mass
    ion_ring_frac = input['constant']['frac_beam']
    mass_density = (input['constant']['background_density'] * (electron_mass + ion_bkgd_mass)) - (ion_ring_frac * ion_bkgd_mass) # Bare minimum
    if irb:
        ion_ring_mass = input['constant']['ion_mass_e'] * constants.electron_mass # Note: assumes background and ring beam ions are then same species
        mass_density += input['constant']['background_density'] * ion_ring_frac * ion_ring_mass
        ion_ring_charge = input['species']['ion_ring_beam']['charge'] * constants.elementary_charge
        ion_gyroperiod = (TWO_PI * ion_ring_mass) / (ion_ring_charge * B0)
    else:
        ion_bkgd_charge = input['species']['proton']['charge'] * constants.elementary_charge
        ion_gyroperiod = (TWO_PI * ion_bkgd_mass) / (ion_bkgd_charge * B0)

    # Interpolate data onto evenly-spaced coordinates
    evenly_spaced_time = np.linspace(ds.coords["time"][0].data, ds.coords["time"][-1].data, len(ds.coords["time"].data))
    field_data = field_data.interp(time=evenly_spaced_time)

    Tci = evenly_spaced_time / ion_gyroperiod
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

    data = xr.DataArray(field_data, coords=[Tci, vA_Tci], dims=["time", "X_Grid_mid"])
    data = data.rename(X_Grid_mid="x_space")
    if plotTk:
        spec_k : xr.DataArray = xrft.xrft.fft(data, dim = "x_space", true_amplitude=True, true_phase=True, window=None)
        spec_k = abs(spec_k)
        spec_k = spec_k.rename(freq_x_space="wavenumber")
        spec_k = spec_k.where(spec_k.wavenumber!=0.0, None)
        #spec_k = spec_k.sel(wavenumber=spec_k.wavenumber<=100.0)
        #spec_k = spec_k.sel(wavenumber=spec_k.wavenumber>=-100.0)
        spec_k.plot(size = 9, x = "wavenumber", y = "time")
        plt.show()
        zeroed_spec = spec_k.where(spec_k.time>0.0, 0.0)
        zeroed_doubled_spec = 2.0 * zeroed_spec # Double spectrum to conserve E
        zeroed_doubled_spec.loc[dict(wavenumber=0.0)] = zeroed_spec.sel(wavenumber=0.0) # Restore original 0-freq amplitude

    jump_factor = 8 # Sample only every this many points from bispectrum
    Bspec, waxis = bispectrumd(data.sel(time = time_pt, method='nearest').to_numpy()[::jump_factor, None], overlap = 0, nfft = 1000)
    waxis = waxis[2:]

    # Upper triangle mask
    upper_mask = np.tri(waxis.shape[0], waxis.shape[0], k=-1)

    # Mask the upper triangle
    Bspec_mask = np.ma.array(Bspec, mask=upper_mask)

    min_k = 2.0 * np.pi /simL_vATci
    nyquist_k = min_k * num_cells / 2.0
    new_nyquist_k = nyquist_k / jump_factor
    my_nyq_k = num_cells/(2.0 *simL_vATci)
    my_new_nyq_k = my_nyq_k / jump_factor
    #plt.pcolormesh(waxis*2.0*new_nyquist_k, waxis*2.0*new_nyquist_k, abs(Bspec))
    #plt.show()

    #plot_bispectrumd(Bspec, waxis*2.0*my_new_nyq_k)
    my_plot_bispectrumd(Bspec_mask, waxis*2.0*my_new_nyq_k, "Wavenumber [Wci/vA]", True)
    #bic, waxis = bicoherence(data.to_numpy())
    #plot_bicoherence(bic, waxis)

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
        help="Simulation field to use in Epoch output format, e.g. \'Derived_Charge_Density\', \'Electric Field_Ex\', \'Magnetic Field_Bz\'",
        required = True,
        type=str
    )
    parser.add_argument(
        "--time",
        action="store",
        help="Time point in gyroperiods to investigate.",
        required = True,
        type=float
    )
    parser.add_argument(
        "--plotTk",
        action="store_true",
        help="Plot t-k.",
        required = False
    )
    parser.add_argument(
        "--irb",
        action="store_true",
        help="Account for properties of an ion ring beam.",
        required = False
    )
    

    args = parser.parse_args()

    plot_bicoherence_spectra(args.dir, args.field, args.time, args.plotTk, args.irb)