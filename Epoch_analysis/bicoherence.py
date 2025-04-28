from sdf_xarray import SDFPreprocess
from pathlib import Path
from scipy import constants
from plasmapy.formulary import frequencies as ppf
from plasmapy.formulary import speeds as pps
from plasmapy.formulary import lengths as ppl
import astropy.units as u
import pybispectra as pbs
import epoch_utils as utils
#from spectrum import bicoherence, plot_bicoherence, bicoherencex, plot_bicoherencex, bispectrumd, bispectrumi, plot_bispectrumd, plot_bispectrumi
from spectrum import bispectrumd
from matplotlib import pyplot as plt
import epydeck
import numpy as np
import xarray as xr
import argparse
import xrft
from typing import Any

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
    cont = plt.contourf(waxis, waxis, abs(Bspec), 100, cmap="plasma")
    plt.colorbar(cont)
    #plt.title("Bispectrum estimated via the direct (FFT) method")
    plt.xlabel(axis_label)
    plt.ylabel(axis_label)
    plt.tight_layout()
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

    plt.rcParams.update({'axes.labelsize': 16})
    plt.rcParams.update({'axes.titlesize': 18})
    plt.rcParams.update({'xtick.labelsize': 14})
    plt.rcParams.update({'ytick.labelsize': 14})

    data = xr.DataArray(field_data, coords=[Tci, vA_Tci], dims=["time", "X_Grid_mid"])
    data = data.rename(X_Grid_mid="x_space")

    # Create t-k spectrum
    original_spec : xr.DataArray = xrft.xrft.fft(data, true_amplitude=True, true_phase=True, window=None)
    #adjusted_data = xrft.xrft.ifft(original_spec)
    original_spec = original_spec.rename(freq_time="frequency", freq_x_space="wavenumber")
    original_spec = original_spec.where(original_spec.wavenumber!=0.0, None)
    original_spec = original_spec.where(original_spec.frequency!=0.0, None)
    zeroed_spec = original_spec.where(original_spec.frequency>0.0, 0.0)

    # Positive ks
    positive_k_spec = zeroed_spec.where(zeroed_spec.wavenumber>0.0, 0.0)
    quadrupled_positive_spec = 4.0 * positive_k_spec # Double spectrum to conserve E
    quadrupled_positive_spec.loc[dict(wavenumber=0.0)] = positive_k_spec.sel(wavenumber=0.0) # Restore original 0-freq amplitude
    quadrupled_positive_spec.loc[dict(frequency=0.0)] = positive_k_spec.sel(frequency=0.0) # Restore original 0-freq amplitude
    positive_data = xrft.xrft.ifft(quadrupled_positive_spec)
    positive_data = positive_data.rename(freq_frequency = "time")
    positive_data = positive_data.rename(freq_wavenumber = "x_space")

    # Negative ks
    negative_k_spec = zeroed_spec.where(zeroed_spec.wavenumber<0.0, 0.0)
    quadrupled_negative_spec = 4.0 * negative_k_spec # Double spectrum to conserve E
    quadrupled_negative_spec.loc[dict(wavenumber=0.0)] = negative_k_spec.sel(wavenumber=0.0) # Restore original 0-freq amplitude
    quadrupled_negative_spec.loc[dict(frequency=0.0)] = negative_k_spec.sel(frequency=0.0) # Restore original 0-freq amplitude
    negative_data = xrft.xrft.ifft(quadrupled_negative_spec)
    negative_data = negative_data.rename(freq_frequency = "time")
    negative_data = negative_data.rename(freq_wavenumber = "x_space")

    if plotTk:
        # Create t-k spectrum
        zero_spec = original_spec.where(original_spec.frequency>0.0, 0.0)
        zeroed_doubled_spec = 2.0 * zero_spec # Double spectrum to conserve E
        zeroed_doubled_spec.loc[dict(wavenumber=0.0)] = zero_spec.sel(wavenumber=0.0) # Restore original 0-freq amplitude
        spec_tk = xrft.xrft.ifft(zeroed_doubled_spec, dim="frequency")
        spec_tk = spec_tk.rename(freq_frequency="time")
        spec_tk : xr.DataArray = abs(spec_tk)
        spec_tk = xrft.xrft.ifft(zeroed_doubled_spec, dim="frequency")
        spec_tk = spec_tk.rename(freq_frequency="time")
        spec_tk_plot : xr.DataArray = abs(spec_tk)
        spec_tk_plot.plot(size=9, x = "wavenumber", y = "time")
        plt.title(f"{directory.name}: Time evolution of spectral power in {field}")
        plt.xlabel("Wavenumber [Wci/vA]")
        plt.ylabel("Time [Wci^-1]")
        plt.show()

    jump_factor = 8 # Sample only every this many points from bispectrum
    Bspec, waxis = bispectrumd(positive_data.sel(time = time_pt, method='nearest').to_numpy()[::jump_factor, None], overlap = 0, nfft = 1000)
    #Bspec, waxis = bicoherence(data.sel(time = time_pt, method='nearest').to_numpy()[::jump_factor, None], overlap = 0, nfft = 1000)
    waxis = waxis[2:]

    # Upper triangle mask
    upper_mask = np.tri(waxis.shape[0], waxis.shape[0], k=-1)
    #lefty_mask = np.flip(upper_mask, axis=0)
    #righty_mask = np.flip(upper_mask, axis=1)

    # Mask the upper triangle
    Bspec_mask_pos = np.ma.array(Bspec, mask=upper_mask)
    #Bspec_mask = np.ma.array(Bspec_mask, mask=righty_mask)

    # min_k = 2.0 * np.pi /simL_vATci
    # nyquist_k = min_k * num_cells / 2.0
    # new_nyquist_k = nyquist_k / jump_factor
    my_nyq_k = num_cells/(2.0 *simL_vATci)
    my_new_nyq_k = my_nyq_k / jump_factor

    #plot_bispectrumd(Bspec, waxis*2.0*my_new_nyq_k)
    #my_plot_bispectrumd(Bspec_mask_pos, waxis*2.0*my_new_nyq_k, r"Wavenumber [$\omega_{ci}/V_A$]", True)
    #plot_bicoherence(Bspec_mask, waxis*2.0*my_new_nyq_k)
    #bic, waxis = bicoherence(data.to_numpy())
    #plot_bicoherence(bic, waxis)

    Bspec, waxis = bispectrumd(negative_data.sel(time = time_pt, method='nearest').to_numpy()[::jump_factor, None], overlap = 0, nfft = 1000)
    #Bspec, waxis = bicoherence(data.sel(time = time_pt, method='nearest').to_numpy()[::jump_factor, None], overlap = 0, nfft = 1000)
    waxis = waxis[2:]

    # Upper triangle mask
    #upper_mask = np.tri(waxis.shape[0], waxis.shape[0], k=-1)
    #righty_mask = np.flip(upper_mask, axis=1)

    # Mask the upper triangle
    Bspec_mask_neg = np.ma.array(Bspec, mask=upper_mask)
    #Bspec_mask = np.ma.array(Bspec_mask, mask=righty_mask)

    #plot_bispectrumd(Bspec, waxis*2.0*my_new_nyq_k)
    #my_plot_bispectrumd(Bspec_mask_neg, waxis*2.0*my_new_nyq_k, r"Wavenumber [$\omega_{ci}/V_A$]", True)

    Bspec_all = Bspec_mask_pos + Bspec_mask_neg
    my_plot_bispectrumd(Bspec_all, waxis*2.0*my_new_nyq_k, r"Wavenumber [$\omega_{ci}/V_A$]", True)

def manual_autobispectrum(directory : Path, time_Tci : float, field : str = "Magnetic_Field_Bz"):
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

    # Load data
    field_data_array : xr.DataArray = ds[field]
    field_data = field_data_array.load()

    # Interpolate data onto evenly-spaced coordinates
    evenly_spaced_time = np.linspace(ds.coords["time"][0].data, ds.coords["time"][-1].data, len(ds.coords["time"].data))
    field_data = field_data.interp(time=evenly_spaced_time)

    # Read input deck
    input = {}
    with open(str(directory / "input.deck")) as id:
        input = epydeck.loads(id.read())

    TWO_PI = 2.0 * np.pi
    B0 = input['constant']['b0_strength']
    ion_gyroperiod = TWO_PI / ppf.gyrofrequency(B = B0 * u.T, particle = "p")
    Tci = evenly_spaced_time / ion_gyroperiod
    ion_ring_frac = input['constant']['frac_beam']
    mass_density = input['constant']['background_density'] * (1.0 - ion_ring_frac) * constants.proton_mass
    alfven_velo = B0 / (np.sqrt(constants.mu_0 * mass_density))
    vA_Tci = ds.coords["X_Grid_mid"] / (ion_gyroperiod * alfven_velo)
    data = xr.DataArray(field_data, coords=[Tci, vA_Tci], dims=["time", "X_Grid_mid"])
    data = data.rename(X_Grid_mid="x_space")

    # Take fft
    og_spec : xr.DataArray = xrft.xrft.fft(data, true_amplitude=True, true_phase=True, window=None)
    og_spec = og_spec.rename(freq_time="frequency", freq_x_space="wavenumber")
    # Remove zero-frequency component
    og_spec = og_spec.where(og_spec.wavenumber!=0.0, None)
    # Get t-k
    tk_spec = utils.create_t_k_spectrum(og_spec, maxK = 100.0)
    # tk_spec = np.log(tk_spec)
    utils.create_t_k_plot(tk_spec, field, field_unit = "T", maxK = 100.0, display = True)
    
    # Make bispectrum
    spec = tk_spec.sel(time = time_Tci, method = "nearest").to_numpy()[::5]
    nfft = spec.size
    # Create all combinations of k1 and k2
    k = np.arange(nfft)
    K1, K2 = np.meshgrid(k, k)
    K3 = (K1 + K2) % nfft

    # Use broadcasting to access X[k1], X[k2], X[k1 + k2]
    bispec = spec[K1] * spec[K2] * np.conj(spec[K3])
    # bispec = np.fft.fftshift(bispec)
    bispec = np.abs(bispec)
    # bispec = np.log(bispec)
    # bispec = np.tril(bispec)
    plt.imshow(bispec, extent=[-100.0, 100.0, -100.0, 100.0], origin="lower", cmap="plasma")
    plt.xlabel('Wavenumber $k_1$')
    plt.ylabel('Wavenumber $k_2$')
    plt.colorbar(label='Magnitude')
    plt.grid(False)
    plt.show()

    utils.create_t_k_plot(tk_spec, field, field_unit = "T", maxK = 100.0, log = True, display = True)

    plt.imshow(np.log(bispec), extent=[-100.0, 100.0, -100.0, 100.0], origin="lower", cmap="plasma")
    plt.xlabel('Wavenumber $k_1$')
    plt.ylabel('Wavenumber $k_2$')
    plt.colorbar(label='Log Magnitude')
    plt.grid(False)
    plt.show()

    # Lower triangle mask
    lower_mask = np.tri(data.shape[0], data.shape[1], k=-1)

    # Upper triangle mask
    upper_mask = lower_mask.T

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

    manual_autobispectrum(args.dir, args.time, args.field)
    #plot_bicoherence_spectra(args.dir, args.field, args.time, args.plotTk, args.irb)