from sdf_xarray import SDFPreprocess
from pathlib import Path
from scipy import constants
import matplotlib.pyplot as plt
#from matplotlib import colors
from inference.plotting import matrix_plot
import astropy.units as u
from plasmapy.formulary import frequencies as ppf
#import xrscipy as xrscipy
#import xrscipy.fft
import epoch_utils as utils
import pandas as pd
import epydeck
import numpy as np
import xarray as xr
import argparse
import xrft
import csv
import os
from tqdm import tqdm

def trim_to_middle_pct(x, pct):
    remove_pct = 100.0 - pct
    remove_frac_one_tail = (remove_pct / 100.0) / 2.0
    remove_indices_one_tail = int(remove_frac_one_tail * len(x))
    print(f"trimmed to middle {len(x) - 2 * remove_indices_one_tail} indices")
    return x[remove_indices_one_tail:-remove_indices_one_tail]

def fit_to_middle_percentage(x, y, pct):
    x_trim = trim_to_middle_pct(x, pct)
    y_trim = trim_to_middle_pct(y, pct)
    return np.polyfit(x_trim, y_trim, deg = 1, full = True)

def plot_growth_rate_data(
        directory : Path, 
        field : str, 
        normalise : bool = False, 
        plotOmegaK : bool = False, 
        plotTk : bool = False, 
        plotGrowth : bool = True,
        plotGammas : bool = False,
        maxK = None, 
        maxW = None,
        maxRes = 0.2,
        numKs = 5,
        gammaWindow = 100,
        log = False, 
        deltaField = False, 
        beam = True,
        deuteron = False,
        savePath = None,
        window : str = None):

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

    # ts = []
    # for t in range(1, len(ds.coords["time"])):
    #     ts.append(ds.coords["time"][t] - ds.coords["time"][t-1])

    # plt.scatter(range(len(ts)), ts)
    # plt.xlabel("time index")
    # plt.ylabel("Difference between t and t-1")
    # plt.show()

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
    B0 = input['constant']['b0_strength']

    electron_mass = input['species']['electron']['mass'] * constants.electron_mass
    ion_bkgd_mass = input['constant']['ion_mass_e'] * constants.electron_mass
    ion_bkgd_charge = input['species']['proton']['charge'] * constants.elementary_charge
    ion_ring_frac = input['constant']['frac_beam']
    bkgd_number_density = input['constant']['background_density']
    mass_density = (bkgd_number_density * (electron_mass + ion_bkgd_mass)) - (ion_ring_frac * ion_bkgd_mass) # Bare minimum
    if beam:
        ion_ring_mass = input['constant']['ion_mass_e'] * constants.electron_mass # Note: assumes background and ring beam ions are then same species
        if deuteron:
            ion_ring_mass *= 2.0
        mass_density += bkgd_number_density * ion_ring_frac * ion_ring_mass
        ion_ring_charge = input['species']['ion_ring_beam']['charge'] * constants.elementary_charge
        ion_gyroperiod = (TWO_PI * ion_ring_mass) / (ion_ring_charge * B0)
    else:
        ion_gyroperiod = (TWO_PI * ion_bkgd_mass) / (ion_bkgd_charge * B0)

    pp_tCI = 1.0 / ppf.gyrofrequency(B0 * u.T, 'p+')
    print(f"Ion gyroperiod from plasmapy in s/rad: {pp_tCI}")
    print(f"Ion gyroperiod from plasmapy in s: {pp_tCI * TWO_PI * u.rad}")
    pp_wLH = ppf.lower_hybrid_frequency(B0 * u.T, bkgd_number_density * u.m**-3, 'p+')
    print(f"Lower hybrid freqency from plasmapy in rad/s: {pp_wLH}")
    wLH_cyclo = pp_wLH * pp_tCI
    print(f"Lower hybrid freqency from plasmapy in W_ci: {wLH_cyclo}")

    # Interpolate data onto evenly-spaced coordinates
    evenly_spaced_time = np.linspace(ds.coords["time"][0].data, ds.coords["time"][-1].data, len(ds.coords["time"].data))
    field_data = field_data.interp(time=evenly_spaced_time)

    Tci = evenly_spaced_time / ion_gyroperiod
    alfven_velo = B0 / (np.sqrt(constants.mu_0 * mass_density))
    print(f"Alfven speed: {alfven_velo}m/s")
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
    w_LH_Wci = utils.calculate_lower_hybrid_frequency(ion_bkgd_charge, ion_bkgd_mass, bkgd_number_density, B0, 'cyc')
    print(f"NORMALISED: Lower Hybrid frequency in Wci: {w_LH_Wci}Wci")

    # Clean data and take FFT
    print(f"Normalise == {normalise}")
    if normalise:
        data = xr.DataArray(field_data - abs(np.mean(field_data)), coords=[Tci, vA_Tci], dims=["time", "X_Grid_mid"])
    else:
        data = xr.DataArray(field_data, coords=[Tci, vA_Tci], dims=["time", "X_Grid_mid"])

    print(f"deltaField == {deltaField}")
    if deltaField:
        #field_data = (field_data - B0) / B0
        #data = (data - B0)**2 / B0**2
        data = (data - data[0])**2 / data[0]**2
    
    data = data.rename(X_Grid_mid="x_space")
    original_spec : xr.DataArray = xrft.xrft.fft(data, true_amplitude=True, true_phase=True, window=None)
    # original_spec : xr.DataArray = xrscipy.fft.fftn(data, 'x_space', 'time')
    # original_spec.data = fft.fftshift(original_spec.data)
    original_spec = original_spec.rename(freq_time="frequency", freq_x_space="wavenumber")
    original_spec = original_spec.where(original_spec.wavenumber!=0.0, None)

    plt.rcParams.update({'axes.labelsize': 16})
    plt.rcParams.update({'axes.titlesize': 18})
    plt.rcParams.update({'xtick.labelsize': 14})
    plt.rcParams.update({'ytick.labelsize': 14})
    # Plot omega-k
    if plotOmegaK:

        spec = abs(original_spec)
        print(f"Log == {log}")
        if log:
            spec = np.log(spec)
        spec = spec.sel(frequency=spec.frequency>=0.0)
        if maxK is not None:
            spec = spec.sel(wavenumber=spec.wavenumber<=maxK)
            spec = spec.sel(wavenumber=spec.wavenumber>=-maxK)
        if maxW is not None:
            spec = spec.sel(frequency=spec.frequency<=maxW)
        spec.plot(size=9, cbar_kwargs={'label': f'Spectral power in {field}' if not log else f'Log of spectral power in {field}'}, cmap='plasma')
        #plt.title(f"{directory.name}: Dispersion relation of {field}")
        plt.ylabel(r"Frequency [$\omega_{ci}$]")
        plt.xlabel(r"Wavenumber [$\omega_{ci}/V_A$]")
        if savePath is not None:
            plt.savefig(savePath / f'{directory.name}_wk_dField-{deltaField}_log-{log}_maxK-{maxK if maxK is not None else "all"}_maxW-{maxW if maxW is not None else "all"}.png')
            plt.clf()
        else:
            plt.show()

        spec = spec.sel(wavenumber=spec.wavenumber>0.0)
        spec.plot(size=9, cbar_kwargs={'label': f'Spectral power in {field}' if not log else f'Log of spectral power in {field}'}, cmap='plasma')
        plt.plot(spec.coords['wavenumber'].data, spec.coords['wavenumber'].data, 'k--', label=r'$V_A$ branch')
        plt.axhline(y = wLH_cyclo, color='black', linestyle=':', label=r'Lower hybrid frequency')
        plt.legend()
        plt.ylabel(r"Frequency [$\omega_{ci}$]")
        plt.xlabel(r"Wavenumber [$\omega_{ci}/V_A$]")
        if savePath is not None:
            plt.savefig(savePath / f'{directory.name}_wk_positiveK_dField-{deltaField}_log-{log}_maxK-{maxK if maxK is not None else "all"}_maxW-{maxW if maxW is not None else "all"}.png')
            plt.clf()
        else:
            plt.show()

        f_over_all_k = spec.sum(dim = "wavenumber")
        f_over_all_k.plot(size=9)
        plt.xticks(ticks=np.arange(np.floor(f_over_all_k.coords['frequency'][0]), np.ceil(f_over_all_k.coords['frequency'][-1])+1.0, 1.0), minor=True)
        plt.grid(which='both', axis='x')
        plt.xlabel(r"Frequency [$\omega_{ci}$]")
        plt.ylabel(r"Sum of power in Bz over all k")
        if savePath is not None:
            plt.savefig(savePath / f'{directory.name}_powerByOmega_dField-{deltaField}_log-{log}_maxK-{maxK if maxK is not None else "all"}_maxW-{maxW if maxW is not None else "all"}.png')
            plt.clf()
        else:
            plt.show()

    # Create t-k spectrum
    print(original_spec.sel(wavenumber=0.0))
    zeroed_spec = original_spec.where(original_spec.frequency>0.0, 0.0)
    zeroed_doubled_spec = 2.0 * zeroed_spec # Double spectrum to conserve E
    print(zeroed_spec.sel(wavenumber=0.0))
    zeroed_doubled_spec.loc[dict(wavenumber=0.0)] = zeroed_spec.sel(wavenumber=0.0) # Restore original 0-freq amplitude
    spec_tk = xrft.xrft.ifft(zeroed_doubled_spec, dim="frequency")
    spec_tk = spec_tk.rename(freq_frequency="time")
    spec_tk : xr.DataArray = abs(spec_tk)
    # spec_cellSpace = spec_cellSpace.assign_coords(k_perp=("freq_X_Grid_mid", vA_Tci))

    # Plot t-k
    if plotTk:
        spec_tk_plot = spec_tk
        if log:
            spec_tk_plot = np.log(spec_tk_plot)
        if args.maxK is not None:
            spec_tk_plot = spec_tk_plot.sel(wavenumber=spec_tk_plot.wavenumber<=maxK)
            spec_tk_plot = spec_tk_plot.sel(wavenumber=spec_tk_plot.wavenumber>=-maxK)
        spec_tk_plot.plot(size=9, x = "wavenumber", y = "time", cbar_kwargs={'label': f'Spectral power in {field}' if not log else f'Log of spectral power in {field}'}, cmap='plasma')
        #plt.title(f"{directory.name}: Time evolution of spectral power in {field}")
        plt.grid()
        plt.xlabel(r"Wavenumber [$\omega_{ci}/V_A$]")
        plt.ylabel(r"Time [$\tau_{ci}$]")
        if savePath is not None:
            plt.savefig(savePath / f'{directory.name}_tk_dField-{deltaField}_log-{log}_maxK-{maxK if maxK is not None else "all"}.png')
            plt.clf()
        else:
            plt.show()

    #sum_over_all_t = np.sum(spec_tk, axis=0)
    peak_powers = spec_tk.max(axis=0)

    # Was highest total power sums, now ks with highest peaks
    peak_powers = np.nan_to_num(peak_powers)
    peak_power_k_indices = np.argpartition(peak_powers, -numKs)[-numKs:][::-1]
    #peak_power_t_index = np.argmax(spec_tk[:,peak_power_k_indices[0]].data)

    # fit_to_middle_pct = 60.0

    # Calculate growth rates by k
    if plotGrowth:
        for i in peak_power_k_indices:
            plt.plot(spec_tk.coords["time"], spec_tk[:,i], label = f"k = {float(spec_tk.coords['wavenumber'][i]):.4f}")
        plt.xlabel("Time [Wci^-1]")
        plt.ylabel(f"Spectral power [{field_data_array.units}]")
        plt.title(f"{directory.name}: Time evolution of {numKs} highest power wavenumbers in {field}")
        plt.legend()
        if savePath is not None:
            plt.savefig(savePath / f'{directory.name}_growthRates_dField-{deltaField}_log-{log}_numK-{numKs if numKs is not None else "all"}.png')
            plt.clf()
        else:
            plt.show()

    if plotGammas:
        # Calculate gamma by time for only high power ks
        for k in tqdm(peak_power_k_indices): # For highest peak power ks
            t_k = spec_tk[:,k]
            all_gammas = []
            all_residuals = []
            for i in tqdm(range(len(t_k) - (gammaWindow + 1))): # For each window
                t_k_window = t_k[i:(i + gammaWindow)]
                fit, res, _, _, _ = np.polyfit(x = t_k.coords["time"][i:(i + gammaWindow)], y = np.log(t_k_window), deg = 1, full = True)
                all_gammas.append(utils.LinearGrowthRateByT(time = t_k.coords["time"][i:(i + gammaWindow)][int(gammaWindow/2)], gamma = float(fit[0]), yIntercept=float(fit[1]), residual = float(res[0])))
                all_residuals.append(float(res[0]))
            
            # Normalise residuals and filter
            filtered = utils.filter_by_residuals(all_gammas, all_residuals, maxRes)

            filtered_times = []
            filtered_gammas = []
            filtered_intercepts = []
            for lgr in filtered:
                filtered_times.append(lgr.time)
                filtered_gammas.append(lgr.gamma)
                filtered_intercepts.append(lgr.yIntercept)
            
            plt.title(f'k = {float(spec_tk.coords["wavenumber"][k]):.4f}')
            plt.scatter(filtered_times, filtered_gammas, marker = 'x')
            plt.xlabel(r"Time at centre of window [$\tau_{ci}$]")
            plt.ylabel(r"Gamma [$\omega_{ci}$]")
            #plt.yscale("log")
            #plt.title(f"{directory.name}: k = {float(spec_tk.coords['wavenumber'][k]):.4f} Growth rate within sliding window of size {gammaWindow} ({gammaWindow*100.0/num_t}%)")
            if savePath is not None:
                plt.savefig(savePath / f'{directory.name}_k{float(spec_tk.coords["wavenumber"][k]):.5f}_growthRateSlidingWindow.png')
                plt.clf()
            else:
                plt.show()

            plt.title(f'Highest growth rate of k = {float(spec_tk.coords["wavenumber"][k]):.4f}')
            plt.plot(t_k.coords["time"], np.log(t_k), label="Data")
            maxGammaIndex = np.argmax(filtered_gammas)
            middleTimeIndex = np.absolute(t_k.coords["time"]-filtered_times[maxGammaIndex]).argmin()
            bestFitTimes = t_k.coords["time"][int(middleTimeIndex-(gammaWindow/2)):int(middleTimeIndex+(gammaWindow/2))]
            bestFit = (filtered_gammas[maxGammaIndex] * bestFitTimes) + filtered_intercepts[maxGammaIndex]
            plt.plot(bestFitTimes, bestFit, label = f"gamma = {float(filtered_gammas[maxGammaIndex]):.4f}")
            plt.xlabel("Time/tau_ci")
            plt.legend()
            plt.ylabel("Log of power in k")
            if savePath is not None:
                plt.savefig(savePath / f'{directory.name}_k{float(spec_tk.coords["wavenumber"][k]):.5f}_highestGamma.png')
                plt.clf()
            else:
                plt.show()  

def calculate_max_growth_rate_in_simulation(
        directory : Path, 
        simNum : int,
        field : str,
        outFile : Path,
        maxK : float,
        maxRes = 0.2,
        gammaWindow = 100,
        log = False, 
        deltaField = False, 
        beam = True,
        deuteron = False,
        plot = False,
        writeToLocalCSV = False,
        figureSavePath = None):
    
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
    B0 = input['constant']['b0_strength']
    B0_angle = input['constant']["b0_angle"]
    background_density = input['constant']['background_density']

    electron_mass = input['species']['electron']['mass'] * constants.electron_mass
    ion_bkgd_mass = input['constant']['ion_mass_e'] * constants.electron_mass
    ion_ring_frac = input['constant']['frac_beam']
    mass_density = (background_density * (electron_mass + ion_bkgd_mass)) - (ion_ring_frac * ion_bkgd_mass) # Bare minimum
    if beam:
        ion_ring_mass = input['constant']['ion_mass_e'] * constants.electron_mass # Note: assumes background and ring beam ions are then same species
        if deuteron:
            ion_ring_mass *= 2.0
        mass_density += background_density * ion_ring_frac * ion_ring_mass
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
    print(f"Alfven speed: {alfven_velo}m/s")
    vA_Tci = ds.coords["X_Grid_mid"] / (ion_gyroperiod * alfven_velo)

    # Nyquist frequencies in normalised units
    simtime_Tci = float(Tci[-1])
    print(f"NORMALISED: Sim time in Tci: {simtime_Tci}Tci")
    samp_freq_time = num_t/simtime_Tci
    print(f"NORMALISED: Sampling frequency in Wci: {samp_freq_time}Wci")
    nyq_freq_time = num_t/(2.0 *simtime_Tci)
    print(f"NORMALISED: Nyquist frequency in Wci: {nyq_freq_time}Wci")
    simL_vATci = float(vA_Tci[-1])
    print(f"NORMALISED: Sim L in vA*Tci: {simL_vATci}vA*Tci")
    samp_freq_space = num_cells/simL_vATci
    print(f"NORMALISED: Sampling frequency in Wci/vA: {samp_freq_space}Wci/vA")
    nyq_freq_space = num_cells/(2.0 *simL_vATci)
    print(f"NORMALISED: Nyquist frequency in Wci/vA: {nyq_freq_space}Wci/vA")

    # Clean data and take FFT
    data = xr.DataArray(field_data, coords=[Tci, vA_Tci], dims=["time", "X_Grid_mid"])
    print(f"deltaField == {deltaField}")
    if deltaField:
        data = (data - data[0])**2 / data[0]**2
    
    data = data.rename(X_Grid_mid="x_space")
    original_spec : xr.DataArray = xrft.xrft.fft(data, true_amplitude=True, true_phase=True, window=None)
    original_spec = original_spec.rename(freq_time="frequency", freq_x_space="wavenumber")
    original_spec = original_spec.where(original_spec.wavenumber!=0.0, None)

    # Create t-k spectrum
    zeroed_spec = original_spec.where(original_spec.frequency>0.0, 0.0)
    zeroed_doubled_spec = 2.0 * zeroed_spec # Double spectrum to conserve E
    zeroed_doubled_spec.loc[dict(wavenumber=0.0)] = zeroed_spec.sel(wavenumber=0.0) # Restore original 0-freq amplitude
    spec_tk = xrft.xrft.ifft(zeroed_doubled_spec, dim="frequency")
    spec_tk = spec_tk.rename(freq_frequency="time")
    spec_tk = abs(spec_tk)
    if maxK is not None:
        spec_tk = spec_tk.sel(wavenumber=spec_tk.wavenumber<=maxK)
        spec_tk = spec_tk.sel(wavenumber=spec_tk.wavenumber>=-maxK)

    if plot or figureSavePath is not None:
        spec_tk.plot(size=9, x = "wavenumber", y = "time", cbar_kwargs={'label': f'Spectral power in {field}' if not log else f'Log of spectral power in {field}'})
        plt.title(f"{directory.name}: time-wavenumber up to k = {maxK if maxK is not None else nyq_freq_space}")
        plt.xlabel(r"Wavenumber [$\omega_{ci}/V_A$]")
        plt.ylabel(r"Time [$\tau_{ci}$]")
        if figureSavePath is not None:
            plt.savefig(figureSavePath / f"{directory.name}_tk_maxK{maxK if maxK is not None else nyq_freq_space}.png")
        if plot:
            plt.show()
        plt.clf()

    max_growth_rates = []

    # Calculate max gamma for all ks
    for k in spec_tk.coords["wavenumber"]:
        print(f"Processing k = {float(k)}....")
        t_k = spec_tk.sel(wavenumber = k)[:]
        all_gammas = []
        all_residuals = []
        for i in range(len(t_k) - (gammaWindow + 1)): # For each window
            t_k_window = t_k[i:(i + gammaWindow)]
            fit, res, _, _, _ = np.polyfit(x = t_k.coords["time"][i:(i + gammaWindow)], y = np.log(t_k_window), deg = 1, full = True)
            # plt.scatter(t_k.coords["time"][i:(i + gammaWindow)], np.log(t_k_window))
            # if not np.isnan(fit[0]):
            #     plt.plot(t_k.coords["time"][i:(i + gammaWindow)], fit[0] * t_k.coords["time"][i:(i + gammaWindow)] + fit[1])
            # plt.title(f'k = {float(k):.3f}')
            # plt.show()
            all_gammas.append(utils.LinearGrowthRateByT(time = float(t_k.coords["time"][i:(i + gammaWindow)][int(gammaWindow/2)]), gamma = float(fit[0]), yIntercept=float(fit[1]), residual = float(res[0])))
            all_residuals.append(float(res[0]))
        
        # Normalise residuals and filter
        filtered = utils.filter_by_residuals(all_gammas, all_residuals, maxRes)

        if filtered.size != 0:
            filtered_gammas = [lgr.gamma for lgr in filtered]
            max_gamma_id = np.nanargmax(filtered_gammas)
            max_growth_rates.append(utils.MaxGrowthRate(wavenumber=float(k), time=filtered[max_gamma_id].time, gamma=filtered[max_gamma_id].gamma, yIntercept=filtered[max_gamma_id].yIntercept))

    all_max_gammas = [mgr.gamma for mgr in max_growth_rates]
    max_gamma_in_sim = max_growth_rates[np.nanargmax(all_max_gammas)]

    if figureSavePath is not None:
        plt.title(f'Highest growth rate of k = {max_gamma_in_sim.wavenumber:.4f}')
        t_k = spec_tk.sel(wavenumber = max_gamma_in_sim.wavenumber)[:]
        plt.plot(t_k.coords["time"], np.log(t_k), label="Data")
        maxGamma = max_gamma_in_sim.gamma
        middleTime = max_gamma_in_sim.time
        middleTimeIndex = np.where(t_k.coords["time"] == middleTime)[0][0]
        bestFitTimes = t_k.coords["time"][int(middleTimeIndex-(gammaWindow/2)):int(middleTimeIndex+(gammaWindow/2))]
        bestFit = (maxGamma * bestFitTimes) + max_gamma_in_sim.yIntercept
        plt.plot(bestFitTimes, bestFit, label = f"gamma = {float(maxGamma):.4f}")
        plt.xlabel("Time/tau_ci")
        plt.legend()
        plt.ylabel("Log of power in k")
        if figureSavePath is not None:
            plt.savefig(figureSavePath / f'{directory.name}_k{max_gamma_in_sim.wavenumber:.5f}_highestGammaFit.png')
        if plot:
            plt.show()
        plt.clf()

    if outFile is not None:
        with open(str(outFile.absolute()), mode="a") as csvOut:
            writer = csv.writer(csvOut)
            writer.writerow([simNum, background_density, ion_ring_frac, B0, B0_angle, max_gamma_in_sim.wavenumber, max_gamma_in_sim.time, max_gamma_in_sim.gamma])
    elif writeToLocalCSV:
        with open(directory / f'{directory.name}_growthRate.csv', mode="w") as csvOut:
            writer = csv.writer(csvOut)
            writer.writerow(["simNumber", "background_density", "frac_beam", "b0_strength", "b0_angle", "wavenumber", "time", "maxGamma"])
            writer.writerow([simNum, background_density, ion_ring_frac, B0, B0_angle, max_gamma_in_sim.wavenumber, max_gamma_in_sim.time, max_gamma_in_sim.gamma])

def analyse_growth_rates_across_simulations(csvData : str):

    df = pd.read_csv(csvData, header=0)
    
    #matrix_plot([np.log(df.background_density.to_numpy()), np.log(df.frac_beam.to_numpy()), df.b0_strength.to_numpy(), df.b0_angle.to_numpy(), df.maxGamma.to_numpy()], labels = ["Log(Density)", "Log(Beam Fraction)", r"B0 [$T$]", r"B0 Angle $[^\circ]$", r"Max Gamma [$\omega_{ci}$]"])
    #matrix_plot([np.log(df.background_density.to_numpy()), np.log(df.frac_beam.to_numpy()), df.b0_strength.to_numpy(), df.b0_angle.to_numpy(), df.time.to_numpy()], labels = ["Log(Density)", "Log(Beam Fraction)", r"B0 [$T$]", r"B0 Angle $[^\circ]$", r"Time [$\tau_{ci}$]"])
    #matrix_plot([np.log(df.background_density.to_numpy()), np.log(df.frac_beam.to_numpy()), df.b0_strength.to_numpy(), df.b0_angle.to_numpy(), df.wavenumber.to_numpy()], labels = ["Log(Density)", "Log(Beam Fraction)", r"B0 [$T$]", r"B0 Angle $[^\circ]$", r"Wavenumber [$\omega_{ci}/V_A$]"])
    matrix_plot([np.log(df.background_density.to_numpy()), np.log(df.frac_beam.to_numpy()), df.b0_strength.to_numpy(), df.b0_angle.to_numpy(), df.maxGamma.to_numpy(), df.time.to_numpy(), df.wavenumber.to_numpy()], labels = ["Log(Density)", "Log(Beam Fraction)", r"B0 [$T$]", r"B0 Angle $[^\circ]$", r"Max Gamma [$\omega_{ci}$]", r"Time [$\tau_{ci}$]", r"Wavenumber [$\omega_{ci}/V_A$]"], show_ticks=True)
    #matrix_plot([df.wavenumber.to_numpy(), df.time.to_numpy(), df.maxGamma.to_numpy()], labels = [r"Wavenumber [$\omega_{ci}/V_A$]", r"Time [$\tau_{ci}$]", r"Max Gamma [$\omega_{ci}$]"])
    

def collate_growth_rate_CSVs(overallDirectory : Path):
    overallCsvPath = args.overallDir / "allGrowthRates.csv"
    with open(overallCsvPath, mode="w") as overallCsv:
        print(f'Processing dirs in {overallDirectory}....')
        dirs = next(os.walk(overallDirectory))[1] 
        writer = csv.writer(overallCsv)
        writer.writerow(["simNumber", "background_density", "frac_beam", "b0_strength", "b0_angle", "wavenumber", "time", "maxGamma"])
        for dir in dirs:
            dir = Path(dir)
            print(f'Processing {dir.name}....')
            csvPath = args.overallDir / dir / f'{dir.name}_growthRate.csv'
            with open(csvPath, mode='r') as dataFile:
                reader = csv.reader(dataFile)
                next(reader) # header
                for row in reader:
                    writer.writerow(row)
                    

if __name__ == "__main__":

    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--overallDir",
        action="store",
        help="Directory containing multiple simulation directories for evaluation.",
        required = False,
        type=Path
    )
    parser.add_argument(
        "--dir",
        action="store",
        help="Directory containing all sdf files from simulation.",
        required = False,
        type=Path
    )
    parser.add_argument(
        "--field",
        action="store",
        help="Simulation field to use in Epoch output format, e.g. \'Derived/Charge_Density\', \'Electric Field/Ex\', \'Magnetic Field/Bz\'",
        required = False,
        type=str
    )
    parser.add_argument(
        "--norm",
        action="store_true",
        help="Switch to normalise data before plotting.",
        required = False
    )
    parser.add_argument(
        "--plotOmegaK",
        action="store_true",
        help="Plot dispersion relation (frequency-wavenumber).",
        required = False
    )
    parser.add_argument(
        "--plotTk",
        action="store_true",
        help="Plot power in wavenumber vs time.",
        required = False
    )
    parser.add_argument(
        "--plotGrowth",
        action="store_true",
        help="Plot time evolution of field.",
        required = False
    )
    parser.add_argument(
        "--plotGammas",
        action="store_true",
        help="Plot all growth rates in run.",
        required = False
    )
    parser.add_argument(
        "--plotMatrix",
        action="store_true",
        help="Plot matrix of growth rates across simulations.",
        required = False
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot details of max gamma calculation across multiple simulations.",
        required = False
    )
    parser.add_argument(
        "--calculateAllGrowthRates",
        action="store_true",
        help="Calculate all growth rates across simulations.",
        required = False
    )
    parser.add_argument(
        "--calculateIndividualGrowthRate",
        action="store_true",
        help="Calculate a single growth rate for one of a batch of simulations (useful for array jobs).",
        required = False
    )
    parser.add_argument(
        "--collateGrowthRates",
        action="store_true",
        help="Collate all growth rates stored in separate CSVs across a simulation directory into one CSV in overallDir.",
        required = False
    )
    parser.add_argument(
        "--maxK",
        action="store",
        help="Max value of k for plotting. Defaults to all k.",
        required = False,
        type=float
    )
    parser.add_argument(
        "--maxW",
        action="store",
        help="Max value of omega for plotting. Defaults to all omega.",
        required = False,
        type=float
    )
    parser.add_argument(
        "--maxRes",
        action="store",
        help="Maximum normalised residual for inclusion in plots, analysis. Defaults to 0.2 (max 1).",
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
        "--gammaWindow",
        action="store",
        help="Size of window in indices (time points) to use for calculating growth rates. Defaults to 20.",
        required = False,
        type=int
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Plot FFT with log of spectral power.",
        required = False
    )
    parser.add_argument(
        "--deltaField",
        action="store_true",
        help="Use the normalised squared change in field value.",
        required = False
    )
    parser.add_argument(
        "--beam",
        action="store_true",
        help="Account for properties of an ion ring beam.",
        required = False
    )
    parser.add_argument(
        "--deuteron",
        action="store_true",
        help="Ion ring beam is composed of deuterons.",
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
        "--figureSavePath",
        action="store",
        help="Directory to which figures should be saved when calculating gamma across multiple simulations.",
        required = False,
        type=Path
    )
    parser.add_argument(
        "--simRunNumber",
        action="store",
        help="Simulation run number, i.e. 23 for run_23 folder in epyrunner naming convention.",
        required = False,
        type=int
    )
    args = parser.parse_args()

    if args.plotMatrix:
        analyse_growth_rates_across_simulations(str(args.dir))
    elif args.calculateAllGrowthRates:
        print("Simulation directories:")
        dirs = next(os.walk(args.overallDir))[1] 
        print(dirs)
        with open(str(args.savePath.absolute()), "w") as outF:
            writer = csv.writer(outF)
            writer.writerow(["simNumber", "background_density", "frac_beam", "b0_strength", "b0_angle", "wavenumber", "time", "maxGamma"])
        
        for dir in dirs:
            simNum = int(dir.split('_')[-1])
            path = args.overallDir / Path(dir)
            print(f"Processing {path}; simNum {int(dir.split('_')[-1])}...")
            calculate_max_growth_rate_in_simulation(path, simNum, args.field, args.savePath, args.maxK, args.maxRes, plot=args.plot, figureSavePath=args.figureSavePath)
    elif args.calculateIndividualGrowthRate:
        simNumber = args.simRunNumber
        print(f"Processing simulation number {simNumber}....")
        path = args.overallDir / Path(f"run_{str(simNumber)}")
        calculate_max_growth_rate_in_simulation(path, simNumber, args.field, args.savePath, args.maxK, args.maxRes, plot=args.plot, writeToLocalCSV=True, figureSavePath=args.figureSavePath)
    
    if args.collateGrowthRates:
        collate_growth_rate_CSVs(overallDirectory=args.overallDir)

    if args.dir is not None:
        plot_growth_rate_data(
            args.dir, 
            args.field, 
            args.norm, 
            args.plotOmegaK, 
            args.plotTk, 
            args.plotGrowth,
            args.plotGammas,
            args.maxK, 
            args.maxW, 
            0.2 if args.maxRes is None else args.maxRes,
            args.numKs,
            100 if args.gammaWindow is None else args.gammaWindow,
            args.log, 
            args.deltaField, 
            args.beam,
            args.deuteron,
            args.savePath)