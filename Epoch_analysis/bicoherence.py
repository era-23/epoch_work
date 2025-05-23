from sdf_xarray import SDFPreprocess
from pathlib import Path
from scipy import constants
from scipy.linalg import hankel
from scipy.signal import convolve2d, get_window, hilbert
from plasmapy.formulary import frequencies as ppf
import astropy.units as u
import pybispectra as pbs
import epoch_utils as utils
from spectrum import bicoherence, plot_bicoherence, bispectrumd, bispectrumi, plot_bispectrumd, plot_bispectrumi
from spectrum.tools.matlab import nextpow2, flat_eq
#from spectrum import bispectrumd, bispectrumi
from matplotlib import pyplot as plt
from matplotlib import colors
import epydeck
import numpy as np
import xarray as xr
import argparse
import xrft  # noqa: E402
import logging
from typing import Tuple, Any, Union

log = logging.getLogger(__file__)

def compute_full_bispectrum(signal, fs=1.0, nfft=256, segment_length=256, overlap=0.5, window_type='hann'):
    """
    Optimized bispectrum calculation using vectorized NumPy operations.
    
    Parameters:
    - signal: 1D array-like signal
    - fs: Sampling frequency
    - nfft: Number of FFT points
    - segment_length: Length of each segment
    - overlap: Fraction of overlap between segments
    - window_type: Type of window to apply

    Returns:
    - B: 2D bispectrum matrix
    - freqs: Frequency vector (includes negative and positive frequencies)
    """
    signal = np.asarray(signal)
    step = int(segment_length * (1 - overlap))
    window = get_window(window_type, segment_length)

    # Segment the signal
    segments = np.array([
        signal[i:i + segment_length] * window
        for i in range(0, len(signal) - segment_length + 1, step)
    ])

    # FFT and shift
    fft_segments = np.fft.fftshift(np.fft.fft(segments, n=nfft, axis=1), axes=1)
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1/fs))

    # Precompute bispectrum
    B = np.zeros((nfft, nfft), dtype=complex)
    for i in range(nfft):
        for j in range(nfft):
            k = (i + j - nfft // 2) % nfft  # adjusted for fftshift
            B[i, j] = np.mean(fft_segments[:, i] * fft_segments[:, j] * np.conj(fft_segments[:, k]))

    return B, freqs

def memory_effieicnt_bispectrumd(
    signal : xr.DataArray,
    timePoint : float,
    axis=-1,
    nfft=128,
    wind=5,
    nsamp=None,
    overlap=50,
    max_nsamp=2048
):
    """
    Memory-efficient version of bispectrumd (direct method).
    """
    # Move axis of interest to first dimension
    sigTranspose = signal.transpose()
    y = sigTranspose.to_numpy()
    ly = y.shape[0]

    # Determine segment size
    overlap = min(max(overlap, 0), 99)
    if nsamp is None:
        nsamp = int(np.fix(ly / (8 - 7 * overlap / 100)))
    nsamp = min(nsamp, max_nsamp)

    # Update nfft
    if nfft < nsamp:
        nfft = 2 ** int(np.ceil(np.log2(nsamp)))

    # Segment control
    overlap_samp = int(nsamp * overlap / 100)
    nadvance = nsamp - overlap_samp
    nsegments = (ly - overlap_samp) // nadvance

    # Smoothing window
    if isinstance(wind, int):
        win = get_window("hann", wind)
        winf = np.concatenate((win[::-1], win[1:]))  # symmetric window
        W1 = winf[:, None]
        W2 = winf[None, :]
        Wsum = hankel(winf[::-1], winf)
        opwind = W1 * W2 * Wsum
    elif isinstance(wind, np.ndarray) and wind.ndim == 1:
        win = wind
        winf = np.concatenate((win[::-1], win[1:]))  # symmetric window
        W1 = winf[:, None]
        W2 = winf[None, :]
        Wsum = hankel(winf[::-1], winf)
        opwind = W1 * W2 * Wsum
    elif isinstance(wind, np.ndarray) and wind.ndim == 2:
        opwind = wind
    else:
        opwind = 1.0

    # Allocate output
    Bspec = np.zeros((nfft, nfft), dtype=np.complex64)

    # Segment loop
    for k in range(nsegments):
        start = k * nadvance
        xseg = y[start : start + nsamp, :]

        xseg = xseg - np.mean(xseg, axis=0, keepdims=True)
        # X = np.fft.fft(seg, n=nfft, axis=0) / nsamp
        # Take fft
        coords = []
        for dim in sigTranspose.dims:
            coords.append(sigTranspose.coords[dim][start : start + nsamp])
        fftSignal = xr.DataArray(xseg, coords=coords, dims = signal.dims)
        og_spec : xr.DataArray = xrft.xrft.fft(fftSignal, true_amplitude=True, true_phase=True, window=None)
        og_spec = og_spec.rename(freq_time="frequency", freq_x_space="wavenumber")
        # Remove zero-frequency component
        og_spec = og_spec.where(og_spec.wavenumber!=0.0, None)
        # Get t-k
        tk_spec = utils.create_t_k_spectrum(og_spec, maxK = None)
        X = tk_spec.sel(time = timePoint, method = "nearest").to_numpy()
        for i in range(nfft):
            for j in range(nfft):
                k_idx = (i + j) % nfft
                Bspec[i, j] += np.sum(X[i] * X[j] * np.conj(X[k_idx]))

    Bspec /= nsegments

    if isinstance(opwind, np.ndarray) and opwind.shape[0] > 1:
        Bspec = convolve2d(Bspec, opwind, mode="same")

    # Frequency axis (normalized)
    faxis = np.fft.fftshift(np.fft.fftfreq(nfft, d=1))
    Bspec = np.fft.fftshift(Bspec)

    return Bspec, faxis

def my_bispectrumd(
    signal: xr.DataArray|np.ndarray,
    timePoint: float,
    nfft: int = 128,
    wind: Union[int, np.ndarray[Any, np.dtype[Any]]] = 5,
    nsamp: int = 0,
    overlap: int = 50,
    axisName: str = "wavenumber"
) -> Tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]:
    """
    Estimate the bispectrum using the direct (FFT) method.

    Parameters:
    -----------
    y : np.ndarray[Any, np.dtype[Any]]
        Input data vector or time-series.
    nfft : int, optional
        FFT length (default is 128). The actual size used is the next power of two greater than 'nsamp'.
    wind : Union[int, np.ndarray[Any, np.dtype[Any]]], optional
        Window specification for frequency-domain smoothing (default is 5).
        If 'wind' is a scalar, it specifies the length of the side of the square for the Rao-Gabr optimal window.
        If 'wind' is a vector, a 2D window will be calculated via w2(i,j) = wind(i) * wind(j) * wind(i+j).
        If 'wind' is a matrix, it specifies the 2-D filter directly.
    nsamp : int, optional
        Samples per segment (default is 0, which sets it to have 8 segments).
    overlap : int, optional
        Percentage overlap of segments, range [0, 99] (default is 50).

    Returns:
    --------
    Bspec : np.ndarray[Any, np.dtype[Any]]
        Estimated bispectrum: an nfft x nfft array, with origin at the center,
        and axes pointing down and to the right.
    waxis : np.ndarray[Any, np.dtype[Any]]
        Vector of frequencies associated with the rows and columns of Bspec.
        Sampling frequency is assumed to be 1.

    Notes:
    ------
    The bispectrum is a higher-order spectral analysis technique that provides information
    about the interaction between different frequency components in a signal.
    """

    (ly, nrecs) = signal.shape

    overlap = min(99, max(overlap, 0))
    if nrecs > 1:
        overlap = 0
        nsamp = ly
    if nrecs == 1 and nsamp <= 0:
        nsamp = int(np.fix(ly / (8 - 7 * overlap / 100)))
    if nfft < nsamp:
        nfft = 2 ** nextpow2(nsamp)
    overlap = int(np.fix(nsamp * overlap / 100))
    nadvance = nsamp - overlap
    nrecs = int(np.fix((ly * nrecs - overlap) / nadvance))

    # Create the 2-D window
    if isinstance(wind, (int, np.integer)):
        winsize = wind
        if winsize < 0:
            winsize = 5
        winsize = winsize - (winsize % 2) + 1
        if winsize > 1:
            mwind = np.fix(nfft / winsize)
            lby2 = (winsize - 1) / 2
            theta = np.array([np.arange(-lby2, lby2 + 1)])
            opwind = np.ones((winsize, 1)) * (theta**2)
            opwind = opwind + opwind.T + (theta.T * theta)
            opwind = 1 - ((2 * mwind / nfft) ** 2) * opwind
            Hex = np.ones((winsize, 1)) * theta
            Hex = abs(Hex) + abs(Hex.T) + abs(Hex + Hex.T)
            Hex = Hex < winsize
            opwind = opwind * Hex
            opwind = opwind * (4 * mwind**2) / (7 * np.pi**2)
        else:
            opwind = 1
    elif isinstance(wind, np.ndarray) and wind.ndim == 1:
        windf = np.concatenate((wind[:0:-1], wind))
        opwind = (windf[:, np.newaxis] * windf) * hankel(np.flipud(wind), wind)
        winsize = len(wind)
    elif isinstance(wind, np.ndarray):
        winsize = wind.shape[0]
        if wind.shape[0] != wind.shape[1]:
            log.info("2-D window is not square: window ignored")
            wind = 1
            winsize = wind.shape[0]
        if winsize % 2 == 0:
            log.info("2-D window does not have odd length: window ignored")
            wind = 1
            winsize = wind.shape[0]
        opwind = wind

    # Accumulate triple products
    Bspec = np.zeros((nfft, nfft))
    mask = hankel(np.arange(nfft), np.array([nfft - 1] + list(range(nfft - 1))))
    locseg = np.arange(nsamp).T
    y = signal.to_numpy()
    y = y.ravel(order="F")

    for _ in range(nrecs):
        xseg = y.isel(locseg).data.reshape(1, -1)
        # Xf = np.fft.fft(xseg - np.mean(xseg), nfft) / nsamp
        # Take fft
        fftSignal = xr.DataArray(xseg - np.mean(xseg), coords=signal.coords[locseg], dims = signal.dims)
        og_spec : xr.DataArray = xrft.xrft.fft(fftSignal, true_amplitude=True, true_phase=True, window=None)
        og_spec = og_spec.rename(freq_time="frequency", freq_x_space="wavenumber")
        # Remove zero-frequency component
        og_spec = og_spec.where(og_spec.wavenumber!=0.0, None)
        # Get t-k
        tk_spec = utils.create_t_k_spectrum(og_spec, maxK = None)
        Xf = tk_spec.sel(time = timePoint, method = "nearest").to_numpy()
        CXf = np.conjugate(Xf).ravel(order="F")
        Bspec = Bspec + flat_eq(Bspec, (Xf * Xf.T) * CXf[mask].reshape(nfft, nfft))
        locseg = locseg + int(nadvance)

    Bspec = np.fft.fftshift(Bspec) / nrecs

    # Frequency-domain smoothing
    if winsize > 1:
        lby2 = int((winsize - 1) / 2)
        Bspec = convolve2d(Bspec, opwind, mode="same")
        Bspec = Bspec[lby2 : lby2 + nfft, lby2 : lby2 + nfft]

    if nfft % 2 == 0:
        waxis = np.transpose(np.arange(-nfft // 2, nfft // 2)) / nfft
    else:
        waxis = np.transpose(np.arange(-(nfft - 1) // 2, (nfft - 1) // 2 + 1)) / nfft

    return Bspec, waxis

def hosa_autobispectrum_original(directory : Path, field : str, time_pt : float, plotTk : bool, irb : bool = True):

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
    pos_spec = positive_data.sel(time = time_pt, method='nearest').to_numpy()[::jump_factor, None]
    Bspec, waxis = bispectrumd(pos_spec, overlap = 0, nfft = 1000)
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
    # my_plot_bispectrumd(Bspec_all, waxis*2.0*my_new_nyq_k, r"Wavenumber [$\omega_{ci}/V_A$]", True)
    # my_plot_bispectrumd(np.log(Bspec_all), waxis*2.0*my_new_nyq_k, r"Wavenumber [$\omega_{ci}/V_A$]", True)

def hosa_autobispectrum_2(signal : xr.DataArray, timePoint : float = None, direct = True, mask : bool = True):
    
    y = signal.sel(time = timePoint, method="nearest")
    y = y.reshape(-1,1) if isinstance(y, np.ndarray) else y.data.reshape(-1,1)
    fs = float(y.size/signal.coords['x_space'][-1])
    if direct:
        # bispec, waxis = bispectrumd(y, nfft = 1024, nsamp = 1024, overlap = 50)
        bispec, waxis = bispectrumd(y, nfft = y.size//2, nsamp = y.size//2, overlap = 10)
        # plot_bispectrumd(bispec, waxis[1:-1])
        # plt.show()
    else:
        bispec, waxis = bispectrumi(y, nfft = 1024, nsamp = 1024, overlap = 50)
        # plot_bispectrumi(bispec, waxis)
        # plt.show()
    
    if mask:
        # Lower triangle mask
        lower_mask = np.fliplr(np.tri(bispec.shape[0], bispec.shape[1], k=-1))
        bispec = np.ma.array(bispec, mask = lower_mask)

    return bispec, waxis * fs

def custom_autobispectrum(signal : xr.DataArray, timePoint : float = None, mask : bool = True):

    fs = float(signal.size/signal.coords['x_space'][-1])
    if timePoint is not None:
        y = signal.sel(time=[timePoint-1.0, timePoint+1.0], method="nearest")
    bispec, waxis = memory_effieicnt_bispectrumd(y, timePoint=timePoint)
    
    if mask:
        # Lower triangle mask
        lower_mask = np.tri(bispec.shape[0], bispec.shape[1], k=-1)
        bispec = np.ma.array(bispec, mask = lower_mask)

    return bispec, waxis * fs

def manual_autobispectrum(signal : xr.DataArray, timePoint_tci : float = None, totalWindow_tci = 2.0, fftWindowSize_tci = 0.25, overlap = 0.5, maxK = None, mask : bool = True) -> np.ndarray:
    
    if len(signal.shape) > 1 and (signal.shape[0] != 1 or signal.shape[1] != 1):
        if timePoint_tci is None:
            raise Exception("timePoint must be provided for 2d signal.")
    if timePoint_tci is None:
        sig_fft = xrft.xrft.fft(signal, true_amplitude=True, true_phase=True, window=None)
        kwargs = {"freq_x_space": "wavenumber"}
        spectrum = sig_fft.rename(**kwargs)
        fs = float(signal.size / signal.coords["x_space"][-1])
        y = spectrum.data
        nfft = y.size
        
        # Create all combinations of k1 and k2
        k = np.arange(nfft)
        K1, K2 = np.meshgrid(k, k)
        K3 = (K1 + K2) % nfft

        # Use broadcasting to access X[k1], X[k2], X[k1 + k2]
        bispec = y[K1] * y[K2] * np.conj(y[K3])

        waxis = np.linspace(-fs/2.0, fs/2.0, bispec.shape[0])
    else:
        
        # Take fft
        spec : xr.DataArray = xrft.xrft.fft(signal, true_amplitude=True, true_phase=True, window=None)
        spec = spec.rename(freq_time="frequency", freq_x_space="wavenumber")
        # Remove zero-frequency component
        spec = spec.where(spec.wavenumber!=0.0, None)
        # Get t-k
        spec = utils.create_t_k_spectrum(spec, maxK = maxK, takeAbs = False)
        # spec = spec.fillna(0.0)

        # Work out how many indices in each window
        times = np.array(signal.coords["time"])
        windowSize_indices = int((signal.coords["time"].size / signal.coords["time"][-1]) * fftWindowSize_tci)
        overlap_indices = int(np.round(overlap * windowSize_indices))
        windowStart_tci = timePoint_tci - (0.5 * totalWindow_tci)
        windowStart_idx = np.where(abs(times-windowStart_tci)==abs(times-windowStart_tci).min())[0][0]
        windowEnd_tci = timePoint_tci + (0.5 * totalWindow_tci)
        windowEnd_idx = np.where(abs(times-windowEnd_tci)==abs(times-windowEnd_tci).min())[0][0]
        signalWindows = []
        startIdx = windowStart_idx
        endIdx = startIdx + windowSize_indices
        while endIdx < windowEnd_idx:
            w = spec.isel(time=slice(startIdx, endIdx))
            signalWindows.append(w)
            startIdx = (endIdx + 1) - overlap_indices
            endIdx = startIdx + windowSize_indices

        initBispec = True
        count = 0
        for window in signalWindows:
            
            # Build bispectrum by averaging FFT data around the time point
            if initBispec:
                bispec = np.zeros((window.shape[1], window.shape[1]), dtype=complex)
                initBispec = False
            
            # y = spec.mean(dim="time").to_numpy()
            y = window.to_numpy()
            nfft = window.shape[1]
            # Create all combinations of k1 and k2
            k = np.arange(nfft)
            K1, K2 = np.meshgrid(k, k)
            K3 = K1 + K2

            # Mask
            k_mask = K3 < nfft
            valid_K3 = np.where(k_mask, K3, 0)
            Y3_conj = np.conj(y[:,valid_K3])

            # Use broadcasting to access X[k1], X[k2], X[k1 + k2]
            # b = y[:,K1] * y[:,K2] * Y3_conj
            bispec += np.mean(y[:,K1] * y[:,K2] * Y3_conj, axis = 0)
            bispec[np.logical_not(k_mask)] = 0.0

            count += 1

        # bispec = np.fft.fftshift(bispec) / count
        bispec = bispec / count
        waxis = np.linspace(-maxK, maxK, bispec.shape[0])

    if mask:
        # Lower triangle mask
        lower_mask = np.fliplr(np.tri(bispec.shape[0], bispec.shape[1], k=-1))
        bispec = np.ma.array(bispec, mask = lower_mask)

    return bispec, waxis

def all_autobispectra(directory : Path, time_Tci : float, field : str = "Magnetic_Field_Bz", showPlots = True):
    
    maxK = 100.0

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
    if showPlots:
        utils.create_t_k_plot(tk_spec, field, field_unit = "T", maxK = 100.0, display = True)
    
    # Make bispectrum
    spec = tk_spec.sel(time = time_Tci, method = "nearest").to_numpy()
    spec = np.nan_to_num(spec)
    
    ##### HOSA
    bs_masked = hosa_autobispectrum_2(spec)

    plt.imshow(bs_masked, extent=[-maxK, maxK, -maxK, maxK], origin="lower", cmap="plasma")
    plt.title("HOSA bispectrum implementation")
    plt.xlabel('Wavenumber $k_1$')
    plt.ylabel('Wavenumber $k_2$')
    plt.colorbar(label='Magnitude')
    plt.grid(False)
    plt.show()

    # Log
    plt.imshow(np.log(bs_masked), extent=[-maxK, maxK, -maxK, maxK], origin="lower", cmap="plasma")
    plt.title("HOSA bispectrum implementation")
    plt.xlabel('Wavenumber $k_1$')
    plt.ylabel('Wavenumber $k_2$')
    plt.colorbar(label='Log Magnitude')
    plt.grid(False)
    plt.show()

    ##### Manual
    spec = spec[::3]
    bs_masked = manual_autobispectrum(spec, mask = True)

    plt.imshow(bs_masked, extent=[-maxK, maxK, -maxK, maxK], origin="lower", cmap="plasma")
    plt.title("Manual bispectrum implementation")
    plt.xlabel('Wavenumber $k_1$')
    plt.ylabel('Wavenumber $k_2$')
    plt.colorbar(label='Magnitude')
    plt.grid(False)
    plt.show()

    # Log
    utils.create_t_k_plot(tk_spec, field, field_unit = "T", maxK = maxK, log = True, display = showPlots)

    plt.imshow(np.log(bs_masked), extent=[-maxK, maxK, -maxK, maxK], origin="lower", cmap="plasma")
    plt.title("Manual bispectrum implementation")
    plt.xlabel('Wavenumber $k_1$')
    plt.ylabel('Wavenumber $k_2$')
    plt.colorbar(label='Log Magnitude')
    plt.grid(False)
    plt.show()

def evaluate_single_signal(signal : xr.DataArray, timePoint : float, totalTimeWindow_tci : float, fftWindow_tci : float, fftOverlap : float, maxK : float, showPlots = True):
    
    # Get bispectra
    # manual_bs, m_waxis = manual_autobispectrum(spectrum, fs, mask = False)
    hosa_bs_d_masked, hosa_waxis = hosa_autobispectrum_2(signal, timePoint=timePoint, direct=True, mask = True)
    manual_bs_masked, m_waxis = manual_autobispectrum(signal, timePoint_tci=timePoint, totalWindow_tci=totalTimeWindow_tci, fftWindowSize_tci=fftWindow_tci, overlap=fftOverlap, maxK = maxK, mask = True)
    # hosa_bs_d, hosa_waxis = hosa_autobispectrum_2(signal, direct=True, mask = False)
    
    # custom_bs_d_masked, c_waxis = custom_autobispectrum(signal, timePoint=timePoint, mask = False)

    # INDIRECT BROKEN
    # hosa_bs_i, _ = hosa_autobispectrum_2(signal, direct=False, fs=fs, mask = False)
    # hosa_bs_i_masked, _ = hosa_autobispectrum_2(signal, direct=False, fs=fs, mask = True)
    
    # signal_shift = np.roll(signal, 100)
    # hosa_bs_shifted_d = hosa_autobispectrum_2(signal_shift, direct=True, mask = False)
    # hosa_bs_shifted_d_masked = hosa_autobispectrum_2(signal_shift, direct=True, mask = True)

    if showPlots:
        
        max_freq = m_waxis[-1]
        
        plt.imshow(np.abs(manual_bs_masked), extent=[-max_freq, max_freq, -max_freq, max_freq], origin="lower", cmap="plasma")
        plt.xlim(-100.0, 100.0)
        plt.ylim(-100.0, 100.0)
        plt.title(f"Manual bispectrum implementation t = {timePoint}")
        plt.xlabel('Wavenumber $k_1$')
        plt.ylabel('Wavenumber $k_2$')
        plt.colorbar(label='Magnitude')
        plt.grid(False)
        plt.show()

        # Phase
        plt.imshow(np.angle(manual_bs_masked), extent=[-max_freq, max_freq, -max_freq, max_freq], origin="lower", cmap="plasma")
        plt.xlim(-100.0, 100.0)
        plt.ylim(-100.0, 100.0)
        plt.title(f"Manual bispectrum implementation -- phase spectrum at t = {timePoint}")
        plt.xlabel('Wavenumber $k_1$')
        plt.ylabel('Wavenumber $k_2$')
        plt.colorbar(label='Angle/rad')
        plt.grid(False)
        plt.show()

        # Log
        plt.imshow(np.log(np.abs(manual_bs_masked)), extent=[-max_freq, max_freq, -max_freq, max_freq], origin="lower", cmap="plasma")
        plt.xlim(-100.0, 100.0)
        plt.ylim(-100.0, 100.0)
        plt.title(f"Manual bispectrum implementation (log) t = {timePoint}")
        plt.xlabel('Wavenumber $k_1$')
        plt.ylabel('Wavenumber $k_2$')
        plt.colorbar(label='Log Magnitude')
        plt.grid(False)
        plt.show()

        max_freq = hosa_waxis[-1]

        plt.imshow(np.abs(hosa_bs_d_masked), extent=[-max_freq, max_freq, -max_freq, max_freq], origin="lower", cmap="plasma")
        plt.xlim(-100.0, 100.0)
        plt.ylim(-100.0, 100.0)
        plt.title(f"HOSA bispectrum direct implementation t = {timePoint}")
        plt.xlabel('Wavenumber $k_1$')
        plt.ylabel('Wavenumber $k_2$')
        plt.colorbar(label='Magnitude')
        plt.grid(False)
        plt.show()

        # Phase
        plt.imshow(np.angle(hosa_bs_d_masked), extent=[-max_freq, max_freq, -max_freq, max_freq], origin="lower", cmap="plasma")
        plt.xlim(-100.0, 100.0)
        plt.ylim(-100.0, 100.0)
        plt.title(f"HOSA bispectrum direct implementation -- phase spectrum at t = {timePoint}")
        plt.xlabel('Wavenumber $k_1$')
        plt.ylabel('Wavenumber $k_2$')
        plt.colorbar(label='Angle/rad')
        plt.grid(False)
        plt.show()

        # Log
        plt.imshow(np.log(np.abs(hosa_bs_d_masked)), extent=[-max_freq, max_freq, -max_freq, max_freq], origin="lower", cmap="plasma")
        plt.xlim(-100.0, 100.0)
        plt.ylim(-100.0, 100.0)
        plt.title(f"HOSA bispectrum direct implementation (log) t = {timePoint}")
        plt.xlabel('Wavenumber $k_1$')
        plt.ylabel('Wavenumber $k_2$')
        plt.colorbar(label='Log Magnitude')
        plt.grid(False)
        plt.show()

        # max_freq = c_waxis[-1]

        # plt.imshow(np.abs(custom_bs_d_masked), extent=[-max_freq, max_freq, -max_freq, max_freq], origin="lower", cmap="plasma")
        # plt.xlim(-100.0, 100.0)
        # plt.ylim(-100.0, 100.0)
        # plt.title("Custom bispectrum implementation")
        # plt.xlabel('Wavenumber $k_1$')
        # plt.ylabel('Wavenumber $k_2$')
        # plt.colorbar(label='Magnitude')
        # plt.grid(False)
        # plt.show()

        # # Phase
        # plt.imshow(np.angle(custom_bs_d_masked), extent=[-max_freq, max_freq, -max_freq, max_freq], origin="lower", cmap="plasma")
        # plt.xlim(-100.0, 100.0)
        # plt.ylim(-100.0, 100.0)
        # plt.title("Custom bispectrum implementation -- phase spectrum")
        # plt.xlabel('Wavenumber $k_1$')
        # plt.ylabel('Wavenumber $k_2$')
        # plt.colorbar(label='Angle/rad')
        # plt.grid(False)
        # plt.show()

        # # Log
        # plt.imshow(np.log(np.abs(custom_bs_d_masked)), extent=[-max_freq, max_freq, -max_freq, max_freq], origin="lower", cmap="plasma")
        # plt.xlim(-100.0, 100.0)
        # plt.ylim(-100.0, 100.0)
        # plt.title("Custom bispectrum implementation (log)")
        # plt.xlabel('Wavenumber $k_1$')
        # plt.ylabel('Wavenumber $k_2$')
        # plt.colorbar(label='Log Magnitude')
        # plt.grid(False)
        # plt.show()

        # INDIRECT BROKEN

        # plt.imshow(np.abs(hosa_bs_i_masked), extent=[-max_freq, max_freq, -max_freq, max_freq], origin="lower", cmap="plasma")
        # plt.title("HOSA bispectrum indirect implementation")
        # plt.xlabel('Wavenumber $k_1$')
        # plt.ylabel('Wavenumber $k_2$')
        # plt.colorbar(label='Magnitude')
        # plt.grid(False)
        # plt.show()

        # # Phase
        # plt.imshow(np.angle(hosa_bs_i_masked), extent=[-max_freq, max_freq, -max_freq, max_freq], origin="lower", cmap="plasma")
        # plt.title("HOSA bispectrum indirect implementation -- phase spectrum")
        # plt.xlabel('Wavenumber $k_1$')
        # plt.ylabel('Wavenumber $k_2$')
        # plt.colorbar(label='Angle/rad')
        # plt.grid(False)
        # plt.show()

        # # Log
        # plt.imshow(np.log(np.abs(hosa_bs_i_masked)), extent=[-max_freq, max_freq, -max_freq, max_freq], origin="lower", cmap="plasma")
        # plt.title("HOSA bispectrum indirect implementation (log)")
        # plt.xlabel('Wavenumber $k_1$')
        # plt.ylabel('Wavenumber $k_2$')
        # plt.colorbar(label='Log Magnitude')
        # plt.grid(False)
        # plt.show()

        # plt.imshow(np.abs(hosa_bc_masked), extent=[-max_freq, max_freq, -max_freq, max_freq], origin="lower", cmap="plasma")
        # plt.title("HOSA bicoherence implementation")
        # plt.xlabel('Wavenumber $k_1$')
        # plt.ylabel('Wavenumber $k_2$')
        # plt.colorbar(label='Magnitude')
        # plt.grid(False)
        # plt.show()

        # # Log
        # plt.imshow(np.log(np.abs(hosa_bc_masked)), extent=[-max_freq, max_freq, -max_freq, max_freq], origin="lower", cmap="plasma")
        # plt.title("HOSA bicoherence implementation (log)")
        # plt.xlabel('Wavenumber $k_1$')
        # plt.ylabel('Wavenumber $k_2$')
        # plt.colorbar(label='Log Magnitude')
        # plt.grid(False)
        # plt.show()

    # Hermitian symmetry
    # print(f"Hermitian symmetry (full spectrum, manual):          {np.allclose(manual_bs, np.conj(np.flip(np.flip(manual_bs, axis=0), axis=1)))}")
    # print(f"Hermitian symmetry (masked spectrum, manual):        {np.allclose(manual_bs_masked, np.conj(np.flip(np.flip(manual_bs_masked, axis=0), axis=1)))}")
    # print(f"Hermitian symmetry (full spectrum, HOSA direct):     {np.allclose(hosa_bs_d, np.conj(np.flip(np.flip(hosa_bs_d, axis=0), axis=1)))}")
    # print(f"Hermitian symmetry (masked spectrum, HOSA direct):   {np.allclose(hosa_bs_d_masked, np.conj(np.flip(np.flip(hosa_bs_d_masked, axis=0), axis=1)))}")
    # print(f"Hermitian symmetry (full spectrum, HOSA indirect):   {np.allclose(hosa_bs_i, np.conj(np.flip(np.flip(hosa_bs_i, axis=0), axis=1)))}")
    # print(f"Hermitian symmetry (masked spectrum, HOSA indirect): {np.allclose(hosa_bs_i_masked, np.conj(np.flip(np.flip(hosa_bs_i_masked, axis=0), axis=1)))}")
    # print(f"Hermitian symmetry (full bicoherence, HOSA):         {np.allclose(hosa_bc, np.conj(np.flip(np.flip(hosa_bc, axis=0), axis=1)))}")
    # print(f"Hermitian symmetry (masked bicoherence, HOSA):       {np.allclose(hosa_bc_masked, np.conj(np.flip(np.flip(hosa_bc_masked, axis=0), axis=1)))}")

    # Signal mean
    # print(f"Mean power (full spectrum, manual):          {np.abs(manual_bs).mean()}")
    # print(f"Mean power (masked spectrum, manual):        {np.abs(manual_bs_masked).mean()}")
    # print(f"Mean power (full spectrum, HOSA direct):     {np.abs(hosa_bs_d).mean()}")
    # print(f"Mean power (masked spectrum, HOSA direct):   {np.abs(hosa_bs_d_masked).mean()}")
    # print(f"Mean power (full spectrum, HOSA indirect):   {np.abs(hosa_bs_i).mean()}")
    # print(f"Mean power (masked spectrum, HOSA indirect): {np.abs(hosa_bs_i_masked).mean()}")

    # Signal max
    # print(f"Max power (full spectrum, manual):          {np.abs(manual_bs).max()}")
    # print(f"Max power (masked spectrum, manual):        {np.abs(manual_bs_masked).max()}")
    # print(f"Max power (full spectrum, HOSA direct):     {np.abs(hosa_bs_d).max()}")
    # print(f"Max power (masked spectrum, HOSA direct):   {np.abs(hosa_bs_d_masked).max()}")
    # print(f"Max power (full spectrum, HOSA indirect):   {np.abs(hosa_bs_i).max()}")
    # print(f"Max power (masked spectrum, HOSA indirect): {np.abs(hosa_bs_i_masked).max()}")

    # Time-shift invariance

def evaluate_signals(directory : Path, time_Tci : float, totalTimeWindow_tci : float = 2.0, fftWindow_tci : float = 0.25, fftOverlap : float = 0.5, maxK : float = 100.0, field : str = "Magnetic_Field_Bz", showPlots = True):

    # ##### Synthetic Gaussian noise
    # print("Evaluating synthetic Gaussian noise....")
    # # Generate noise
    # num_points = 2048
    # max_t = 1.0
    # t = np.linspace(0.0, max_t, num_points, endpoint=False)
    # y = np.random.randn(2048)
    # if showPlots:
    #     plt.plot(t, y)
    #     plt.show()

    # evaluate_single_signal(signal=y, xCoords=t, xName="time", fs = num_points/max_t, showPlots=showPlots)
    
    # ##### Known synthetic signal
    # print("Evaluating synthetic signal....")
    # # Generate signal
    # num_points = 2048
    # max_t = 1.0
    # t = np.linspace(0.0, max_t, num_points, endpoint=False)
    # y = np.sin(2.0 * np.pi * 25.0 * t) + np.sin(2.0 * np.pi * 60.0 * t) + np.sin(2.0 * np.pi * 85.0 * t)
    # if showPlots:
    #     plt.plot(t, y)
    #     plt.show()
    # signal = xr.DataArray(y, coords={"x_space" : t}, dims="x_space")

    # evaluate_single_signal(signal=signal, fs = num_points/max_t, showPlots=showPlots)

    ##### Real MCI signal
    print(f"Evaluating signal from real data ({directory}), {field} at {time_Tci}....")
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

    
    #spec_sampling_freq = float(spec.coords["wavenumber"][-1]) * 2.0

    # Debug
    # print(vA_Tci)
    # print(data.coords["x_space"])
    # print(data.shape)
    # early_time = data.sel(time=0.1, method="nearest")
    # early_time.plot()
    # plt.show()
    # mid_time = data.sel(time=time_Tci, method="nearest")
    # mid_time.plot()
    # plt.show()

    evaluate_single_signal(data, time_Tci, totalTimeWindow_tci, fftWindow_tci, fftOverlap, maxK, showPlots=True)

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
        "--totalTimeWindow",
        action="store",
        help="Total time window in gyroperiods to use for bispectra.",
        required = False,
        type=float
    )
    parser.add_argument(
        "--fftTimeWindow",
        action="store",
        help="Window size of individual FFTs in gyroperiods to use for bispectra.",
        required = False,
        type=float
    )
    parser.add_argument(
        "--fftOverlap",
        action="store",
        help="Fraction of overlap in FFT windows to use for bispectra. Must be in the range 0.0 - 1.0.",
        required = False,
        type=float
    )
    parser.add_argument(
        "--maxK",
        action="store",
        help="Max wavenumber to use for bispectra. Bispectra will be shown from +/-maxK. Defaults to 100.0.",
        required = False,
        type=float
    )
    parser.add_argument(
        "--showPlots",
        action="store_true",
        help="Plot t-k.",
        required = False
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate bispectrum methods.",
        required = False
    )

    args = parser.parse_args()

    if args.evaluate:
        evaluate_signals(args.dir, args.time, args.totalTimeWindow, args.fftTimeWindow, args.fftOverlap, args.maxK, args.field, args.showPlots)
    else:
        all_autobispectra(args.dir, args.time, args.field, args.showPlots)
    # hosa_autobispectrum(args.dir, args.field, args.time, args.plotTk, args.irb)
    #plot_bicoherence_spectra(args.dir, args.field, args.time, args.plotTk, args.irb)