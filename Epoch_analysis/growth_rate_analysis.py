from sdf_xarray import SDFPreprocess
from pathlib import Path
from scipy import constants
from dataclasses import dataclass
import matplotlib.pyplot as plt
import epydeck
import numpy as np
import xarray as xr
import argparse
import xrft
import csv
import os

@dataclass
class LinearGrowthRateByK:
    wavenumber: float
    gamma: float
    residual: float

@dataclass
class LinearGrowthRateByT:
    time: float
    gamma: float
    residual: float

@dataclass
class MaxGrowthRate:
    wavenumber: float
    time: float
    gamma: float

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

def filter_by_residuals(x, residuals, maxRes):
    x = np.array(x)
    min_res = np.nanmin(residuals)
    max_res = np.nanmax(residuals)
    range_res = max_res - min_res
    for lgr in x:
        lgr.residual = (lgr.residual - min_res) / range_res
    
    # Filter growth rates  
    x_low_residuals = np.array([i for i in x if i.residual <= maxRes])

    for i in x_low_residuals:
        i.residual = (i.residual * range_res) + min_res

    return x_low_residuals

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
        gammaWindow = 20,
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
    ion_ring_frac = input['constant']['frac_beam']
    mass_density = (input['constant']['background_density'] * (electron_mass + ion_bkgd_mass)) - (ion_ring_frac * ion_bkgd_mass) # Bare minimum
    if beam:
        ion_ring_mass = input['constant']['ion_mass_e'] * constants.electron_mass # Note: assumes background and ring beam ions are then same species
        if deuteron:
            ion_ring_mass *= 2.0
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
    original_spec = original_spec.rename(freq_time="frequency", freq_x_space="wavenumber")
    original_spec = original_spec.where(original_spec.wavenumber!=0.0, None)

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
        spec.plot(size=9)
        if savePath is not None:
            plt.savefig(savePath / f'{directory.name}_wk_dField-{deltaField}_log-{log}_maxK-{maxK if maxK is not None else "all"}_maxW-{maxK if maxK is not None else "all"}.png')
        plt.title(f"{directory.name}: Dispersion relation of {field}")
        plt.ylabel("Frequency [Wci]")
        plt.xlabel("Wavenumber [Wci/vA]")
        plt.show()

    # Create t-k spectrum
    zeroed_spec = original_spec.where(original_spec.frequency>0.0, 0.0)
    zeroed_doubled_spec = 2.0 * zeroed_spec # Double spectrum to conserve E
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
            spec_tk_plot = spec_tk.sel(wavenumber=spec_tk.wavenumber<=maxK)
            spec_tk_plot = spec_tk_plot.sel(wavenumber=spec_tk_plot.wavenumber>=-maxK)
        spec_tk_plot.plot(size=9, x = "wavenumber", y = "time")
        if savePath is not None:
            plt.savefig(savePath / f'{directory.name}_tk_dField-{deltaField}_log-{log}_maxK-{maxK if maxK is not None else "all"}.png')
        plt.title(f"{directory.name}: Time evolution of spectral power in {field}")
        plt.xlabel("Wavenumber [Wci/vA]")
        plt.ylabel("Time [Wci^-1]")
        plt.show()

    #sum_over_all_t = np.sum(spec_tk, axis=0)
    peak_powers = spec_tk.max(axis=0)

    # Was highest total power sums, now ks with highest peaks
    peak_powers = np.nan_to_num(peak_powers)
    peak_power_k_indices = np.argpartition(peak_powers, -numKs)[-numKs:][::-1]
    peak_power_t_index = np.argmax(spec_tk[:,peak_power_k_indices[0]].data)

    fit_to_middle_pct = 60.0

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
        plt.show()

        print(f"Max power at index {peak_power_t_index}, t = {spec_tk.coords['time'][peak_power_t_index]}")
        for i in peak_power_k_indices:
            plt.plot(spec_tk.coords["time"][:peak_power_t_index], spec_tk[:peak_power_t_index,i], label = f"k = {float(spec_tk.coords['wavenumber'][i]):.4f}")
        plt.xlabel("Time [Wci^-1]")
        plt.ylabel(f"Spectral power [{field_data_array.units}]")
        plt.title(f"{directory.name}: Time evolution of {numKs} highest power wavenumbers in {field} up to time of max power")
        plt.legend()
        if savePath is not None:
            plt.savefig(savePath / f'{directory.name}_growthRates_earlyTime_dField-{deltaField}_log-{log}_numK-{numKs if numKs is not None else "all"}.png')
        plt.show()

        for i in peak_power_k_indices:
            plt.plot(spec_tk.coords["time"][:peak_power_t_index], spec_tk[:peak_power_t_index,i], label = f"k = {float(spec_tk.coords['wavenumber'][i]):.4f}")
        plt.yscale("log")
        plt.xlabel("Time [Wci^-1]")
        plt.ylabel(f"Log of spectral power [{field_data_array.units}]")
        plt.title(f"{directory.name}: Time evolution of {numKs} highest power wavenumbers in {field} up to time of max power (log scale)")
        plt.legend()
        if savePath is not None:
            plt.savefig(savePath / f'{directory.name}_LogGrowthRates_earlyTime_dField-{deltaField}_log-{log}_numK-{numKs if numKs is not None else "all"}.png')
        plt.show()

        plt.plot(spec_tk.coords["time"][:peak_power_t_index], np.log(spec_tk[:peak_power_t_index,peak_power_k_indices[0]]), label = f"k = {float(spec_tk.coords['wavenumber'][peak_power_k_indices[0]]):.4f}")
        #plt.yscale("log")
        t_middle = trim_to_middle_pct(spec_tk.coords["time"][:peak_power_t_index], fit_to_middle_pct)
        power_middle = trim_to_middle_pct(np.log(spec_tk[:peak_power_t_index,peak_power_k_indices[0]]), fit_to_middle_pct)
        fit = np.polyfit(x = t_middle, y = power_middle, deg = 1)
        plt.title(f"{directory.name}: Linear fit of middle {fit_to_middle_pct}% of highest log power wavenumber in {field} up to time of max power")
        plt.xlabel("Time [Wci^-1]")
        plt.ylabel("Log of spectral power [au]")
        print(f"y = {float(fit[0]):.4f}x + {float(fit[1]):.4f}")
        plt.plot(t_middle, np.polyval(fit, t_middle), label = f"gamma = {float(fit[0]):.4f}")
        plt.legend()
        plt.show()

    # Calculate this gamma for all k and plot
    all_linear_growth_rates = []
    all_residuals = []
    filtered_spec_tk = spec_tk.sel(wavenumber=spec_tk.wavenumber<=maxK)
    filtered_spec_tk = filtered_spec_tk.sel(wavenumber=filtered_spec_tk.wavenumber>=-maxK)
    for k in range(len(filtered_spec_tk.coords['wavenumber'])):
        fit, res, _, _, _ = np.polyfit(x = filtered_spec_tk.coords["time"][:peak_power_t_index], y = np.log(filtered_spec_tk[:peak_power_t_index,k]), deg = 1, full=True)
        all_linear_growth_rates.append(LinearGrowthRateByK(wavenumber = float(filtered_spec_tk.coords['wavenumber'][k].data), gamma = float(fit[0]), residual = float(res[0])))
        all_residuals.append(float(res[0]))
    
    # Normalise growth rates
    lgr_low_residuals = filter_by_residuals(all_linear_growth_rates, all_residuals, maxRes = maxRes)
    k_filtered = []
    gamma_filtered = []
    res_filtered = []
    for lgr in lgr_low_residuals:
        k_filtered.append(lgr.wavenumber)
        gamma_filtered.append(lgr.gamma)
        res_filtered.append(lgr.residual)

    if plotGammas:
        plt.scatter(k_filtered, gamma_filtered, marker='x')
        plt.title(f"{directory.name}: Gamma during initial linear growth phase, by wavenumber")
        plt.xlabel("Wavenumber [Wci/vA]")
        plt.ylabel("Gamma [Wci]")
        plt.show()

        plt.scatter(k_filtered, res_filtered, marker='x')
        plt.title(f"{directory.name}: Residuals of Gamma fit during initial linear growth phase, by wavenumber")
        plt.xlabel("Wavenumber [Wci/vA]")
        plt.ylabel("Sum of squared errors [au]")
        plt.show()

    # plt.scatter(gamma_filtered, res_filtered, marker='x')
    # plt.title(f"{directory.name}: Residuals of Gamma fit during initial linear growth phase, by growth rate")
    # plt.xlabel("Gamma [Wci]")
    # plt.ylabel("Sum of squared errors [au]")
    # plt.show()

    # fit, res, _, _, _ = np.polyfit(x = gamma_filtered, y = np.log(res_filtered), deg = 1, full=True)
    # plt.scatter(gamma_filtered, res_filtered, marker='x')
    # gamma_filtered.sort()
    # plt.plot(gamma_filtered, np.exp(np.polyval(fit, gamma_filtered)), color='orange', label = f'Fit: g = {float(fit[0]):.3f}')
    # plt.title(f"{directory.name}: Residuals of Gamma fit during initial linear growth phase, by growth rate")
    # plt.xlabel("Gamma [Wci]")
    # plt.ylabel("Sum of squared errors [au]")
    # plt.legend()
    # plt.show()

    # Calculate gamma by time for only high power ks
    for k in peak_power_k_indices: # For highest peak power ks
        t_k = spec_tk[:,k]
        all_gammas = []
        all_residuals = []
        for i in range(len(t_k) - (gammaWindow + 1)): # For each window
            t_k_window = t_k[i:(i + gammaWindow)]
            fit, res, _, _, _ = np.polyfit(x = t_k.coords["time"][i:(i + gammaWindow)], y = np.log(t_k_window), deg = 1, full = True)
            all_gammas.append(LinearGrowthRateByT(time = t_k.coords["time"][i:(i + gammaWindow)][int(gammaWindow/2)], gamma = float(fit[0]), residual = float(res[0])))
            all_residuals.append(float(res[0]))
        
        # Normalise residuals and filter
        filtered = filter_by_residuals(all_gammas, all_residuals, maxRes)

        filtered_times = []
        filtered_gammas = []
        for lgr in filtered:
            filtered_times.append(lgr.time)
            filtered_gammas.append(lgr.gamma)
        
        plt.scatter(filtered_times, filtered_gammas, marker = 'x')
        plt.xlabel("Time at centre of window [ion_gyroperiods]")
        plt.ylabel("Gamma [Wci]")
        #plt.yscale("log")
        plt.title(f"{directory.name}: k = {float(spec_tk.coords['wavenumber'][k]):.4f} Growth rate within sliding window of size {gammaWindow} ({gammaWindow*100.0/num_t}%)")
        #plt.savefig(savePath / f'{directory.name}_k{k:.5f}_growthRateSlidingWindow.png')
        plt.show()

def calculate_max_growth_rate_in_simulation(
        directory : Path, 
        simNum : int,
        field : str,
        outFile : Path,
        maxK : float,
        maxRes = 0.2,
        gammaWindow = 20,
        log = False, 
        deltaField = False, 
        beam = True,
        deuteron = False):
    
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

    max_growth_rates = []

    # Calculate max gamma for all ks
    for k in spec_tk.coords["wavenumber"]:
        t_k = spec_tk.sel(wavenumber = k)[:]
        all_gammas = []
        all_residuals = []
        for i in range(len(t_k) - (gammaWindow + 1)): # For each window
            t_k_window = t_k[i:(i + gammaWindow)]
            fit, res, _, _, _ = np.polyfit(x = t_k.coords["time"][i:(i + gammaWindow)], y = np.log(t_k_window), deg = 1, full = True)
            all_gammas.append(LinearGrowthRateByT(time = float(t_k.coords["time"][i:(i + gammaWindow)][int(gammaWindow/2)]), gamma = float(fit[0]), residual = float(res[0])))
            all_residuals.append(float(res[0]))
        
        # Normalise residuals and filter
        filtered = filter_by_residuals(all_gammas, all_residuals, maxRes)

        if filtered.size != 0:
            filtered_gammas = [lgr.gamma for lgr in filtered]
            max_gamma_id = np.nanargmax(filtered_gammas)
            max_growth_rates.append(MaxGrowthRate(wavenumber=float(k), time=filtered[max_gamma_id].time, gamma=filtered[max_gamma_id].gamma))

    all_max_gammas = [mgr.gamma for mgr in max_growth_rates]
    max_gamma_in_sim = max_growth_rates[np.nanargmax(all_max_gammas)]

    with open(str(outFile.absolute()), mode="a") as csvOut:
        writer = csv.writer(csvOut)
        writer.writerow([simNum, background_density, ion_ring_frac, B0, B0_angle, max_gamma_in_sim.wavenumber, max_gamma_in_sim.time, max_gamma_in_sim.gamma])


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
    args = parser.parse_args()

    if args.overallDir is not None:
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
            calculate_max_growth_rate_in_simulation(path, simNum, args.field, args.savePath, args.maxK, maxRes = 0.05, deuteron = True)

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
            20 if args.gammaWindow is None else args.gammaWindow,
            args.log, 
            args.deltaField, 
            args.beam,
            args.deuteron,
            args.savePath)