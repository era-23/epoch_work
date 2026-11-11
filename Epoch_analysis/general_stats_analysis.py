import glob
from pathlib import Path
import h5py
from matplotlib import pyplot as plt
import numpy as np
import argparse
import netCDF4 as nc
from scipy import signal
import xarray as xr
import xrft
import epoch_utils as eu
from scipy.stats import linregress
from scipy.signal import find_peaks, correlate
from scipy.optimize import minimize_scalar
from sklearn.metrics import root_mean_squared_error, r2_score
import plasmapy.formulary.frequencies as ppf
import plasmapy.particles as ppp
import astropy.units as u

import warnings

import ml_utils
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas

def find_subgroup_with_var(filepath, target_var, explicit_group=None):
    if explicit_group is not None:
        group_name = explicit_group.lstrip("/")
        return xr.open_dataset(filepath, group=group_name)

    # Inspect groups using netCDF4
    with nc.Dataset(filepath, 'r') as root:
        # Check root first
        if target_var in root.variables:
            return xr.open_dataset(filepath)

        # Walk all subgroups
        def walk_groups(g, current_path=""):
            for name, subgroup in g.groups.items():
                path = f"{current_path}/{name}"
                if target_var in subgroup.variables:
                    return path
                found = walk_groups(subgroup, path)
                if found:
                    return found
            return None

        found_path = walk_groups(root)
        if found_path is not None:
            return xr.open_dataset(filepath, group=found_path.lstrip("/"))

    raise KeyError(f"Variable '{target_var}' not found in root or any subgroup of {filepath}")

def estimate_B0_from_netcdf_spectra(
        netcdf_directory : Path, 
        fields : list, 
        particle : str = "He-4 2+", 
        max_frequency_mHz : float = None,
    ):

    plt.rcParams.update({'axes.titlesize': 20.0})
    plt.rcParams.update({'axes.labelsize': 20.0})
    plt.rcParams.update({'xtick.labelsize': 20.0})
    plt.rcParams.update({'ytick.labelsize': 20.0})
    plt.rcParams.update({'legend.fontsize': 20.0})

    combined_statsFiles = glob.glob(str(netcdf_directory / "*.nc"))
    originalB0s = {f : [] for f in fields}
    recoveredB0s_max = {f : [] for f in fields}
    recoveredB0s_firstPeak = {f : [] for f in fields}
    recoveredB0s_peakSeparation = {f : [] for f in fields}
    recoveredB0s_hybrid = {f : [] for f in fields}
    variedParams = {"pitch" : [], "backgroundDensity" : [], "beamFraction" : []}

    for simFile in combined_statsFiles:
        data_nc = nc.Dataset(simFile, mode="a")
        data_xr = xr.open_datatree(simFile, engine="netcdf4")
        for key, value in variedParams.items():
            value.append(data_xr.attrs[key])

        for field in fields:
            simName = Path(simFile).name
            # print(f"File: {filename}, Field: {field}")
            data : xr.DataArray = data_xr[field]

            # print(data["frequency"])

            # Convert to SI (on original spectrum)
            known_B0 = float(data_xr.B0strength)
            originalB0s[field].append(known_B0)
            
            # print(f"Gyrofrequency in SI: {gyrofrequency_in_SI}")
            if data.coords["frequency"].max() < 100.0: # Not already in SI
                gyrofrequency_in_SI = ppf.gyrofrequency(known_B0 * u.T, particle = particle)
                si_coords = data.coords["frequency"] * (gyrofrequency_in_SI / (2.0 * np.pi))
                # print(f"Frequency coords in SI: {si_coords}")
                data = data.assign_coords({"frequency" : si_coords})

            # data.plot()
            # plt.title(Path(simFile).name)
            # plt.show()

            if max_frequency_mHz:
                data = data[data["frequency"] <= (max_frequency_mHz * 1e6)]
                data_np = data.to_numpy()
                if np.argmin(data_np) == 0: 
                    stationary_point = int(next(i for i,(v0,v1) in enumerate(zip(data_np[:-1], data_np[1:])) if v0>v1))
                elif np.argmax(data_np) == 0: # Else if the first point is a maximum
                    stationary_point = int(next(i for i,(v0,v1) in enumerate(zip(data_np[:-1], data_np[1:])) if v0<v1))
                else:
                    continue
                after_sp = data_np[stationary_point:]
                before_sp = np.linspace(np.mean(data_np), data_np[stationary_point], stationary_point)
                new_data = np.concatenate((before_sp, after_sp), axis = 0)

                global_min = 2.2308287782833357e-12
                new_data = 10.0 * np.log10(new_data / global_min)
                data = xr.DataArray(new_data, coords = {"frequency" : data["frequency"]})
                
                # data.plot()
                # plt.title(Path(simFile).name)
                # plt.show()
                # data = xr.DataArray(resamp_series[1:])
                # data = data.assign_coords({"frequency" : np.linspace(resamp_coords[0], resamp_coords[-1], data.shape[0])})

            # data.plot()
            # plt.title(Path(simFile).name)
            # plt.show()
            
            og_spec : xr.DataArray = xrft.xrft.fft(data, true_amplitude=False, true_phase=True, window=None)
            spec : xr.DataArray = np.abs(og_spec)
            # print(spec.coords["freq_frequency"])
            spec = spec.sel(freq_frequency = slice(0.0, None))
            # spec = spec.isel(freq_frequency = slice(10, None))
            # spec.plot()
            # plt.show()
            spec = spec.where(spec.freq_frequency >= float(0.1*spec.freq_frequency.max().data), other=0.0)
            # spec = spec.isel(freq_frequency = slice(int(0.13*len(spec.coords["freq_frequency"])), None))

            # spec.plot()
            # plt.show()
            
            # Max freq method
            maxFreqFreq = float(spec.idxmax().data)
            maxFreq = 1.0 / maxFreqFreq
            maxPower = spec.max().data
            # Recover B0
            recovered_B0 = maxFreq * 2.0 * np.pi * (ppp.alpha.mass / ppp.alpha.charge)

            # Peak finding method
            height_proportion = 0.9
            prominence_proportion = 0.9
            p, pd = find_peaks(spec.data, height=float(height_proportion*spec.max().data), prominence=float(prominence_proportion*spec.max().data), distance = np.max([1.0, 0.1*len(spec)]))
            if len(p) > 0:
                peakFreqFreq = float(spec.freq_frequency[p[0]])
                peakFreq = 1.0/peakFreqFreq
                recovered_B0_2 = peakFreq * 2.0 * np.pi * (ppp.alpha.mass / ppp.alpha.charge)

                # Peak separation method
                sepFreqFreq = float(spec.freq_frequency[p[0]])
                sepFreq = 1.0 / sepFreqFreq
                if len(p) > 1:
                    seps = []
                    for i in range(1, len(p)):
                        seps.append(float(spec.freq_frequency[p[i]]) - float(spec.freq_frequency[p[i-1]]))
                    sepFreq = 1.0 / np.mean(seps)
                recovered_B0_3 = sepFreq * 2.0 * np.pi * (ppp.alpha.mass / ppp.alpha.charge)
            else:
                recovered_B0_2 = recovered_B0
                recovered_B0_3 = recovered_B0
            
            # # print(f"Max: {maxPower}, Max index: {spec.argmax().data}, Max coord: {maxFreqFreq * 1/gyrofrequency_in_SI.unit} Max coord in OG units: {maxFreq * gyrofrequency_in_SI.unit}")
            # assert spec.coords["freq_frequency"][-1] == np.max(spec.coords["freq_frequency"])
            # plt.scatter(maxFreqFreq, maxPower, color = "red", marker = "x", label = "Max power")
            # plt.scatter(spec.freq_frequency[p].data, spec[p].data, color = "green", marker = "+", label = "Power peaks")
            # spec.plot()
            # plt.legend()
            # plt.title(f"True B0: {known_B0:.3f}, Max B0: {recovered_B0.value:.3f}, Peak B0: {recovered_B0_2.value:.3f}, Sep B0: {recovered_B0_3.value:.3f}")
            # plt.show()
            
            # # print(f"{simName}: Original B0: {data_xr.B0strength * u.T}, recovered B0 : {recovered_B0}")
            # if abs(recovered_B0.value - data_xr.B0strength) > 1.0:
            #     print("-------------------------------------------------------")
            #     print(f"{field}: Poor recovery of B0: {simName} -- error = {recovered_B0.value - data_xr.B0strength}")
            #     print(f"{simName} parameters: B0 {data_xr.B0strength}, pitch {data_xr.pitch}, density {data_xr.backgroundDensity}, beam frac {data_xr.beamFraction}.")
            #     print(f"Percentiles: B0 {int(np.rint(((data_xr.B0strength - 1.0) / 4.0) * 100.0))}, pitch {int(np.rint(((data_xr.pitch - 0.01) / (0.99-0.01)) * 100.0))}, density {int(np.rint(((np.log10(data_xr.backgroundDensity) - 19) / 1.0) * 100.0))} beam frac {int(np.rint(((np.log10(data_xr.beamFraction) + 2) / -2.0) * 100.0))}")
            #     print("-------------------------------------------------------")

            # Record
            recoveredB0s_max[field].append(recovered_B0.value)
            recoveredB0s_firstPeak[field].append(recovered_B0_2.value)
            recoveredB0s_peakSeparation[field].append(recovered_B0_3.value)
            recoveredB0s_hybrid[field].append(np.mean([recovered_B0_3.value, recovered_B0_2.value]))
            # data_nc[field].recovered_B0 = recovered_B0

        data_xr.close()
        data_nc.close()

    # Stats
    for field in fields:
        print(f"{field}: Max peak B0 recovery R2:        {r2_score(originalB0s[field], recoveredB0s_max[field])}")
        print(f"{field}: First peak B0 recovery R2:      {r2_score(originalB0s[field], recoveredB0s_firstPeak[field])}")
        print(f"{field}: Peak separation B0 recovery R2: {r2_score(originalB0s[field], recoveredB0s_peakSeparation[field])}")
        print(f"{field}: Hybrid B0 recovery R2:          {r2_score(originalB0s[field], recoveredB0s_hybrid[field])}")

    plt.figure(figsize=(8,6))
    
    for field in fields:

        colour = "red" if "Bz" in field else "green" if "Ex" in field else "blue"
        mk = "o" if "Bz" in field else "x" if "Ex" in field else "+"

        ogs = originalB0s[field]
        recs = recoveredB0s_hybrid[field]
        absErrors = np.abs(np.array(recs) - np.array(ogs))
        squared_errors = (np.array(recs) - np.array(ogs))**2
        
        # Plot original vs. recovered B0s
        result = linregress(ogs, recs)
        print(f"Plotting predictions vs true values for {field}")
        print(f"{field}: B0 r2 = {result.rvalue**2:.3f}, S.E. = {result.stderr:.3f}, rmse = {root_mean_squared_error(ogs, recs)}")
        threshold = 0.2
        print(f"{field}: {sum(v <= threshold for v in absErrors)}/{len(absErrors)} predictions are within {threshold}T.")
        
        # plt.title(f"{field}: B0\n(r2 = {result.rvalue**2:.3f}, S.E. = {result.stderr:.3f})")
        plt.scatter(ogs, recs, color = colour, alpha = 0.9, marker = mk, s = 100, label = f"{eu.fieldNameToText(field)}")
        sortB0 = sorted(ogs)
        
    plt.plot(sortB0, sortB0, color = "black", alpha = 0.9, linestyle = "dashed", label = "ideal prediction")
    plt.xlabel(r"Original $B_0$ [T]")
    plt.ylabel(r"Recovered $B_0$ [T]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

        # Plot errors with other varied quantities
        # for param, values in variedParams.items():
        #     result = linregress(values, squared_errors)
        #     plt.figure(figsize=(8,8))
        #     plt.title(f"{field}: Errors\n(r2 = {result.rvalue**2:.3f}, S.E. = {result.stderr:.3f})")
        #     plt.scatter(values, squared_errors, marker = "x", color = "red")
        #     plt.plot(values, (result.slope * np.array(values)) + result.intercept, color = "black", alpha = 0.9, linestyle = "dashed", label = "fit")
        #     plt.xlabel(f"{param} [{eu.fieldNameToUnit(param)}]")
        #     # if param == "backgroundDensity" or param == "beamFraction":
        #     #     plt.xscale("log")
        #     plt.ylabel(r"Squared error [$T^2$]")
        #     plt.grid()
        #     plt.legend()
        #     plt.tight_layout()
        #     plt.show()
    
    # print(f"Max frequency mean: {np.mean(maxCoords)} median: {np.median(maxCoords)} var: {np.var(maxCoords)} sd: {np.std(maxCoords)}")

def estimate_B0_from_csv_dat_spectra(csv_dat_directory : Path, known_B0 = 1.581, particle : str = "D+"):

    plt.rcParams.update({'axes.titlesize': 20.0})
    plt.rcParams.update({'axes.labelsize': 20.0})
    plt.rcParams.update({'xtick.labelsize': 20.0})
    plt.rcParams.update({'ytick.labelsize': 20.0})
    plt.rcParams.update({'legend.fontsize': 20.0})

    spectra_paths = glob.glob(str(csv_dat_directory / "*.dat"))
    spectra_names = [Path(p).name for p in spectra_paths]
    originalB0s = {f : [] for f in spectra_names}
    recoveredB0s_max = {f : [] for f in spectra_names}
    recoveredB0s_firstPeak = {f : [] for f in spectra_names}
    recoveredB0s_peakSeparation = {f : [] for f in spectra_names}
    recoveredB0s_hybrid = {f : [] for f in spectra_names}

    max_freq = 120.5
    clean_resample = True

    for path in spectra_paths:

        path_name = Path(path).name

        original_data = pandas.read_csv(path, sep = r"\s+", names = ["frequency", "power"])
        max_freq = float(original_data.iloc[-1]["frequency"]) if max_freq == None else max_freq
        trunc_data = original_data[(original_data["frequency"] > 0.0) & (original_data["frequency"] < max_freq)].sort_values(by = "frequency")
        equal_samples = np.linspace(float(trunc_data["frequency"].min()), max_freq, len(trunc_data))
        trunc_data = trunc_data.drop_duplicates(subset=["frequency"], keep="first").sort_values("frequency")

        if clean_resample:
            freqs_old = trunc_data["frequency"].to_numpy()
            power_old = trunc_data["power"].to_numpy()

            # Binning for max-pooling operation
            # Midpoints define the boundaries between adjacent target frequencies
            midpoints = (equal_samples[:-1] + equal_samples[1:]) / 2.0
            bin_edges = np.concatenate([
                [equal_samples[0] - (midpoints[0] - equal_samples[0])], # Left boundary of first bin
                midpoints,
                [equal_samples[-1] + (equal_samples[-1] - midpoints[-1])] # Right boundary of last bin
            ])

            # Apply Max-Pooling over each bin window
            power_max_pooled = np.zeros_like(equal_samples)
            for i in range(len(equal_samples)):
                bin_min = bin_edges[i]
                bin_max = bin_edges[i + 1]
                
                # Identify original points falling inside the current frequency bin
                mask = (freqs_old >= bin_min) & (freqs_old < bin_max)
                
                if np.any(mask):
                    # Extract peak height inside this bin
                    power_max_pooled[i] = np.max(power_old[mask])
                else:
                    # Fallback to linear interpolation for empty bins (sparse regions without points)
                    power_max_pooled[i] = np.interp(equal_samples[i], freqs_old, power_old, left=0.0, right=0.0)

            clean_data = pandas.DataFrame({
                "frequency": equal_samples,
                "power": power_max_pooled
            })

        peakFind_data = clean_data["power"]
        p, pd = find_peaks(clean_data["power"], height=float(0.05*peakFind_data.max()), prominence=float(0.05*peakFind_data.max()), distance = 0.01*len(peakFind_data))
        peak_seps = []
        for i in range(1, len(p)):
            peak_seps.append(float(clean_data["frequency"][p[i]] - clean_data["frequency"][p[i-1]]))
        peak_seps = np.array(peak_seps)
        ps_mean = peak_seps.mean()
        ps_sd = peak_seps.std()
        mean_mode_peak_sep = peak_seps[(peak_seps >= ps_mean - ps_sd) & (peak_seps <= ps_mean + ps_sd)].mean()
        print(f"Peak positions (MHz): {clean_data["frequency"][p].to_numpy()}")
        print(f"Peak seprtions (MHz): {peak_seps}")
        print(f"Mean(mode) of peak separations: {mean_mode_peak_sep}")
        print(f"Recovered B0 from Mean(mode) of peak separations: {(mean_mode_peak_sep * 1e6 * 2.0 * np.pi * ppp.deuteron.mass) / ppp.deuteron.charge}")
        plt.plot(trunc_data["frequency"], trunc_data["power"], label = "original")
        plt.plot(clean_data["frequency"], clean_data["power"], label = "clean")
        plt.scatter(clean_data["frequency"][p], clean_data["power"][p], marker = "+", color = "green", label = "peaks")
        plt.legend()
        plt.title(path_name)
        plt.show()

        data = clean_data.set_index("frequency").to_xarray().to_dataarray().sel(variable="power")

        og_spec : xr.DataArray = xrft.xrft.fft(data, dim="frequency", true_amplitude=False, true_phase=True, window=None)
        spec : xr.DataArray = np.abs(og_spec)
        spec.plot()
        plt.show()
        spec = spec.sel(freq_frequency = slice(0.0, None))
        spec = spec.where(spec.freq_frequency >= float(0.01*spec.freq_frequency.max().data), other=0.0)
        # spec = spec.isel(freq_frequency = slice(int(0.13*len(spec.coords["freq_frequency"])), None))
        spec.plot()
        plt.show()
        
        # Max freq method
        maxFreqFreq = float(spec.idxmax().data)
        maxFreq_MHz = 1.0 / maxFreqFreq # MHz
        maxFreq_Hz = maxFreq_MHz * 1e6
        maxFreq_radPs = maxFreq_Hz * 2.0 * np.pi
        maxPower = spec.max().data
        # Recover B0
        recovered_B0 = (maxFreq_radPs * ppp.deuteron.mass) / ppp.deuteron.charge

        # Peak finding method
        p, pd = find_peaks(spec.data, height=float(0.05*spec.max().data), prominence=float(0.05*spec.max().data), distance = 0.01*len(spec))
        peakFreqFreq = float(spec.freq_frequency[p[0]])
        peakFreq_MHz = 1.0/peakFreqFreq
        peakFreq_Hz = peakFreq_MHz * 1e6
        peakFreq_radPs = peakFreq_Hz * 2.0 * np.pi
        recovered_B0_2 = (peakFreq_radPs * ppp.deuteron.mass) / ppp.deuteron.charge

        # Peak separation method
        sepFreqFreq = float(spec.freq_frequency[p[0]])
        sepFreq_MHz = 1.0 / sepFreqFreq
        sepFreq_Hz = sepFreq_MHz * 1e6
        sepFreq_radPs = sepFreq_Hz * 2.0 * np.pi
        if len(p) > 1:
            seps = []
            for i in range(1, len(p)):
                seps.append(float(spec.freq_frequency[p[i]]) - float(spec.freq_frequency[p[i-1]]))
            sepFreq_radPs = (1.0 / np.mean(seps)) * 1e6 * 2.0 * np.pi
        recovered_B0_3 = (sepFreq_radPs * ppp.alpha.mass) / ppp.alpha.charge
        
        print(f"Max: {maxPower}, Max index: {spec.argmax().data}, Max coord: {maxFreqFreq}, Max coord in OG units: {maxFreq_MHz}")
        assert spec.coords["freq_frequency"][-1] == np.max(spec.coords["freq_frequency"])
        plt.scatter(maxFreqFreq, maxPower, color = "red", marker = "x", label = "Max power")
        plt.scatter(spec.freq_frequency[p].data, spec[p].data, color = "green", marker = "+", label = "Power peaks")
        spec.plot()
        plt.legend()
        plt.title(f"True B0: {known_B0:.3f}, Max B0: {recovered_B0.value:.3f}, Peak B0: {recovered_B0_2.value:.3f}, Sep B0: {recovered_B0_3.value:.3f}")
        plt.show()
        
        print(f"{path_name}: Original B0: {known_B0 * u.T}, recovered B0 : {recovered_B0}")
        if abs(recovered_B0.value - known_B0) > 1.0:
            print("-------------------------------------------------------")
            print(f"{path_name}: Poor recovery of B0 -- error = {recovered_B0.value - known_B0}")
            # print(f"{simName} parameters: B0 {known_B0}, pitch {data_xr.pitch}, density {data_xr.backgroundDensity}, beam frac {data_xr.beamFraction}.")
            # print(f"Percentiles: B0 {int(np.rint(((known_B0 - 1.0) / 4.0) * 100.0))}, pitch {int(np.rint(((data_xr.pitch - 0.01) / (0.99-0.01)) * 100.0))}, density {int(np.rint(((np.log10(data_xr.backgroundDensity) - 19) / 1.0) * 100.0))} beam frac {int(np.rint(((np.log10(data_xr.beamFraction) + 2) / -2.0) * 100.0))}")
            print("-------------------------------------------------------")

        # Record
        recoveredB0s_max[path_name] = recovered_B0.value
        recoveredB0s_firstPeak[path_name] = recovered_B0_2.value
        recoveredB0s_peakSeparation[path_name] = recovered_B0_3.value
        recoveredB0s_hybrid[path_name] = np.mean([recovered_B0_3.value, recovered_B0_2.value])
        originalB0s[path_name] = known_B0

    # # Stats
    # print(f"{path_name}: Max peak B0 recovery R2:        {r2_score(originalB0s[path_name], recoveredB0s_max[path_name])}")
    # print(f"{path_name}: First peak B0 recovery R2:      {r2_score(originalB0s[path_name], recoveredB0s_firstPeak[path_name])}")
    # print(f"{path_name}: Peak separation B0 recovery R2: {r2_score(originalB0s[path_name], recoveredB0s_peakSeparation[path_name])}")
    # print(f"{path_name}: Hybrid B0 recovery R2:          {r2_score(originalB0s[path_name], recoveredB0s_hybrid[path_name])}")

    for k, v in recoveredB0s_max.items():
        print(f"{k} -- Actual B0: {known_B0:.5f} T, Recovered B0: {v:.5f} T, Error = {(v - known_B0):.5f} T ({(100.0 * (v - known_B0) / known_B0):.5f}%)")

    # plt.figure(figsize=(8,6))

    # # colour = "red" if "Bz" in field else "green" if "Ex" in field else "blue"
    # # mk = "o" if "Bz" in field else "x" if "Ex" in field else "+"
    # ogs = list(originalB0s.values())
    # recs = list(recoveredB0s_hybrid.values())
    # absErrors = np.abs(np.array(recs) - np.array(ogs))
    # squared_errors = (np.array(recs) - np.array(ogs))**2
    
    # # Plot original vs. recovered B0s
    # result = linregress(ogs, recs)
    # print(f"Plotting predictions vs true values for {path_name}")
    # print(f"{path_name}: B0 r2 = {result.rvalue**2:.3f}, S.E. = {result.stderr:.3f}, rmse = {root_mean_squared_error(ogs, recs)}")
    # threshold = 0.2
    # print(f"{path_name}: {sum(v <= threshold for v in absErrors)}/{len(absErrors)} predictions are within {threshold}T.")
    
    # # plt.title(f"{field}: B0\n(r2 = {result.rvalue**2:.3f}, S.E. = {result.stderr:.3f})")
    # plt.scatter(ogs, recs, alpha = 0.9, s = 100, label = path_name)
    # sortB0 = sorted(ogs)
        
    # plt.plot(sortB0, sortB0, color = "black", alpha = 0.9, linestyle = "dashed", label = "ideal prediction")
    # plt.xlabel(r"Original $B_0$ [T]")
    # plt.ylabel(r"Recovered $B_0$ [T]")
    # plt.grid()
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # Plot errors with other varied quantities
    # for param, values in variedParams.items():
    #     result = linregress(values, squared_errors)
    #     plt.figure(figsize=(8,8))
    #     plt.title(f"{field}: Errors\n(r2 = {result.rvalue**2:.3f}, S.E. = {result.stderr:.3f})")
    #     plt.scatter(values, squared_errors, marker = "x", color = "red")
    #     plt.plot(values, (result.slope * np.array(values)) + result.intercept, color = "black", alpha = 0.9, linestyle = "dashed", label = "fit")
    #     plt.xlabel(f"{param} [{eu.fieldNameToUnit(param)}]")
    #     # if param == "backgroundDensity" or param == "beamFraction":
    #     #     plt.xscale("log")
    #     plt.ylabel(r"Squared error [$T^2$]")
    #     plt.grid()
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()
    
    # print(f"Max frequency mean: {np.mean(maxCoords)} median: {np.median(maxCoords)} var: {np.var(maxCoords)} sd: {np.std(maxCoords)}")

def estimate_B0_from_h5_spectra(h5_directory : Path, particle : str):

    print_plot_all = False
    
    plt.rcParams.update({'axes.titlesize': 20.0})
    plt.rcParams.update({'axes.labelsize': 20.0})
    plt.rcParams.update({'xtick.labelsize': 20.0})
    plt.rcParams.update({'ytick.labelsize': 20.0})
    plt.rcParams.update({'legend.fontsize': 20.0})

    spectrum_paths = glob.glob(str(h5_directory / "*.h5"))
    originalB0s = []
    recoveredB0s_rawFirstPeak = []
    recoveredB0s_rawPeakSeparation = []
    recoveredB0s_max = []
    recoveredB0s_firstPeak = []
    recoveredB0s_fftPeakSeparation = []
    recoveredB0s_hybrid = []

    inputData = []
    for f in spectrum_paths:
        data : xr.DataTree = xr.open_datatree(f)
        inputData.append(data)
        data.close()

    if len(inputData) == 0:
        raise Exception(f"No input data found at '{h5_directory}' (searching for '*.h5')")

    inputFreqs = np.array(inputData[0]["spectra"].F)
    if len(inputData) > 1:
        assert np.allclose(inputData[0]["spectra"].F.data, inputData[1]["spectra"].F.data)
        inputParams = np.concat((np.array(inputData[0]["parameters"].X.data, dtype=np.float64), np.array(inputData[1]["parameters"].X.data, dtype=np.float64)), axis=1)
        inputSpectra = np.concat((np.array(inputData[0]["spectra"].Y.data, dtype=np.float64), np.array(inputData[1]["spectra"].Y.data, dtype=np.float64)), axis=1).T
    else:
        inputParams = np.array(inputData[0]["parameters"].X.data, dtype=np.float64)
        inputSpectra = np.array(inputData[0]["spectra"].Y.data, dtype=np.float64).T

    max_freq = inputFreqs[-1]
    print(set(np.diff(inputFreqs)))
    originalB0s = inputParams[0]
    mean_B0 = np.mean(originalB0s)

    for s in range(inputSpectra.shape[0]):
        spectrum = inputSpectra[s]
        peakFind_data = spectrum
        p, pd = find_peaks(spectrum, height=float(0.05*peakFind_data.max()), prominence=float(0.05*peakFind_data.max()), distance = 0.03*len(peakFind_data))
        peak_seps = []
        for i in range(1, len(p)):
            peak_seps.append(float(inputFreqs[p[i]] - inputFreqs[p[i-1]]))
        peak_seps = np.array(peak_seps)
        if len(peak_seps) > 1:
            ps_mean = peak_seps.mean()
            ps_sd = peak_seps.std()
            mean_mode_peak_sep = peak_seps[(peak_seps >= ps_mean - ps_sd) & (peak_seps <= ps_mean + ps_sd)].mean()
            recovered_B0_rawPeakSeps = (mean_mode_peak_sep * 1e6 * 2.0 * np.pi * ppp.deuteron.mass) / ppp.deuteron.charge
        elif len(peak_seps) > 0:
            recovered_B0_rawPeakSeps = (inputFreqs[p] * 1e6 * 2.0 * np.pi * ppp.deuteron.mass) / ppp.deuteron.charge
            mean_mode_peak_sep = 0.0
        else:
            print("No peaks identified, writing dataset mean")
            recovered_B0_rawPeakSeps = (mean_B0 * 1e6 * 2.0 * np.pi * ppp.deuteron.mass) / ppp.deuteron.charge
            mean_mode_peak_sep = 0.0
        if print_plot_all:
            print(f"Peak positions   (MHz): {inputFreqs[p]}")
            print(f"Peak separations (MHz): {peak_seps}")
            print(f"Mean(mode) of peak separations: {mean_mode_peak_sep}")
            print(f"Recovered B0 from Mean(mode) of peak separations: {(mean_mode_peak_sep * 1e6 * 2.0 * np.pi * ppp.deuteron.mass) / ppp.deuteron.charge}")
            plt.plot(inputFreqs, spectrum, label = "original")
            plt.scatter(inputFreqs[p], spectrum[p], marker = "+", color = "green", label = "peaks")
            plt.legend()
            plt.title(f"Spectrum #{s}")
            plt.show()

        recovered_B0_rawPeakSeps = (mean_mode_peak_sep * 1e6 * 2.0 * np.pi * ppp.deuteron.mass) / ppp.deuteron.charge

        data = xr.DataArray(data = spectrum, coords = {"frequency" : inputFreqs}, name = "power", dims = "frequency")
        og_spec : xr.DataArray = xrft.xrft.fft(data, dim="frequency", true_amplitude=False, true_phase=True, window=None)
        spec : xr.DataArray = np.abs(og_spec)
        # spec.plot()
        # plt.show()
        spec = spec.sel(freq_frequency = slice(0.0, None))
        spec = spec.where(spec.freq_frequency >= float(0.005*spec.freq_frequency.max().data), other=0.0)
        if print_plot_all:
            spec.plot()
            plt.show()

        # Max freq method
        maxFreqFreq = float(spec.idxmax().data)
        maxFreq_MHz = 1.0 / maxFreqFreq # MHz
        maxFreq_Hz = maxFreq_MHz * 1e6
        maxFreq_radPs = maxFreq_Hz * 2.0 * np.pi
        maxPower = spec.max().data
        # Recover B0
        recovered_B0_1 = (maxFreq_radPs * ppp.deuteron.mass) / ppp.deuteron.charge

        # Peak finding method
        p, pd = find_peaks(spec.data, height=float(0.05*spec.max().data), prominence=float(0.05*spec.max().data), distance = 0.01*len(spec))
        peakFreqFreq = float(spec.freq_frequency[p[0]])
        peakFreq_MHz = 1.0/peakFreqFreq
        peakFreq_Hz = peakFreq_MHz * 1e6
        peakFreq_radPs = peakFreq_Hz * 2.0 * np.pi
        recovered_B0_2 = (peakFreq_radPs * ppp.deuteron.mass) / ppp.deuteron.charge

        # Peak separation method
        sepFreqFreq = float(spec.freq_frequency[p[0]])
        sepFreq_MHz = 1.0 / sepFreqFreq
        sepFreq_Hz = sepFreq_MHz * 1e6
        sepFreq_radPs = sepFreq_Hz * 2.0 * np.pi
        if len(p) > 1:
            seps = []
            for i in range(1, len(p)):
                seps.append(float(spec.freq_frequency[p[i]]) - float(spec.freq_frequency[p[i-1]]))
            sepFreq_radPs = (1.0 / np.mean(seps)) * 1e6 * 2.0 * np.pi
        recovered_B0_3 = (sepFreq_radPs * ppp.deuteron.mass) / ppp.deuteron.charge

        known_B0 = originalB0s[s]

        # print(f"Max: {maxPower}, Max index: {spec.argmax().data}, Max coord: {maxFreqFreq}, Max coord in OG units: {maxFreq_MHz}")
        assert spec.coords["freq_frequency"][-1] == np.max(spec.coords["freq_frequency"])
        if print_plot_all:
            plt.scatter(maxFreqFreq, maxPower, color = "red", marker = "x", label = "Max power")
            plt.scatter(spec.freq_frequency[p].data, spec[p].data, color = "green", marker = "+", label = "Power peaks")
            spec.plot()
            plt.legend()
            plt.title(f"True B0: {known_B0:.3f}, Max B0: {recovered_B0_1.value:.3f}, Peak B0: {recovered_B0_2.value:.3f}, Sep B0: {recovered_B0_3.value:.3f}")
            plt.show()

            print(f"Spectrum #{s}: Original B0: {known_B0 * u.T}, recovered B0 : {recovered_B0_1}")
            if abs(recovered_B0_1.value - known_B0) > 1.0:
                print("-------------------------------------------------------")
                print(f"Spectrum #{s}: Poor recovery of B0 -- error = {recovered_B0_1.value - known_B0}")
                # print(f"{simName} parameters: B0 {known_B0}, pitch {data_xr.pitch}, density {data_xr.backgroundDensity}, beam frac {data_xr.beamFraction}.")
                # print(f"Percentiles: B0 {int(np.rint(((known_B0 - 1.0) / 4.0) * 100.0))}, pitch {int(np.rint(((data_xr.pitch - 0.01) / (0.99-0.01)) * 100.0))}, density {int(np.rint(((np.log10(data_xr.backgroundDensity) - 19) / 1.0) * 100.0))} beam frac {int(np.rint(((np.log10(data_xr.beamFraction) + 2) / -2.0) * 100.0))}")
                print("-------------------------------------------------------")

        # Record
        recoveredB0s_rawPeakSeparation.append(recovered_B0_rawPeakSeps.value)
        recoveredB0s_max.append(recovered_B0_1.value)
        recoveredB0s_firstPeak.append(recovered_B0_2.value)
        recoveredB0s_fftPeakSeparation.append(recovered_B0_3.value)
        recoveredB0s_hybrid.append(np.mean([recovered_B0_3.value, recovered_B0_2.value]))

    plt.figure(figsize=(12,10))
    plt.scatter(originalB0s, recoveredB0s_max, label = "Recovered by max FFT peak")
    plt.plot(sorted(originalB0s), sorted(originalB0s), "k--", label = "ideal B0 recovery")
    plt.xlabel("Original B0")
    plt.ylabel("Recovered B0")
    plt.legend()
    plt.title(f"Linear data, JET range -- r^2: {r2_score(originalB0s, recoveredB0s_max)}")
    plt.show()

    plt.figure(figsize=(12,10))
    plt.scatter(originalB0s, recoveredB0s_firstPeak, label = "Recovered by first FFT peak")
    plt.plot(sorted(originalB0s), sorted(originalB0s), "k--", label = "ideal B0 recovery")
    plt.xlabel("Original B0")
    plt.ylabel("Recovered B0")
    plt.legend()
    plt.title(f"Linear data, JET range -- r^2: {r2_score(originalB0s, recoveredB0s_firstPeak)}")
    plt.show()

    plt.figure(figsize=(12,10))
    plt.scatter(originalB0s, recoveredB0s_fftPeakSeparation, label = "Recovered by FFT peak separation")
    plt.plot(sorted(originalB0s), sorted(originalB0s), "k--", label = "ideal B0 recovery")
    plt.xlabel("Original B0")
    plt.ylabel("Recovered B0")
    plt.legend()
    plt.title(f"Linear data, JET range -- r^2: {r2_score(originalB0s, recoveredB0s_fftPeakSeparation)}")
    plt.show()

    plt.figure(figsize=(12,10))
    plt.scatter(originalB0s, recoveredB0s_hybrid, label = "Recovered by mean(first FFT peak, peak separation)")
    plt.plot(sorted(originalB0s), sorted(originalB0s), "k--", label = "ideal B0 recovery")
    plt.xlabel("Original B0")
    plt.ylabel("Recovered B0")
    plt.legend()
    plt.title(f"Linear data, JET range -- r^2: {r2_score(originalB0s, recoveredB0s_hybrid)}")
    plt.show()

    plt.figure(figsize=(12,10))
    plt.scatter(originalB0s, recoveredB0s_rawPeakSeparation, label = "Recovered by raw peak separation")
    plt.plot(sorted(originalB0s), sorted(originalB0s), "k--", label = "ideal B0 recovery")
    plt.xlabel("Original B0")
    plt.ylabel("Recovered B0")    
    plt.legend()
    plt.title(f"Linear data, JET range -- r^2: {r2_score(originalB0s, recoveredB0s_rawPeakSeparation)}")
    plt.show()

    # for k, v in recoveredB0s_max.items():
    #     print(f"{k} -- Actual B0: {known_B0:.5f} T, Recovered B0: {v:.5f} T, Error = {(v - known_B0):.5f} T ({(100.0 * (v - known_B0) / known_B0):.5f}%)")

def estimate_B0_robust(
    h5_directory: Path, 
    particle : str = "deuteron",
    b0_search_range: tuple = (0.5, 3.0),
    num_comb_harmonics: int = 15,
    plot_methods: bool = True
) -> dict:
    """
    Recovers B0 from ICE spectra using Comb-Template Matching, Autocorrelation,
    and Median Peak Spacing.
    
    Parameters:
    -----------
    h5_directory : Path
        Directory containing .h5 spectra files.
    particle : str
    b0_search_range : tuple
        (min_B0, max_B0) expected in Tesla for template matching bounds.
    num_comb_harmonics : int
        Number of harmonics to include in the comb template.
    plot_results : bool
        If True, renders diagnostic plots for each spectrum.
        
    Returns:
    --------
    dict containing lists of original and recovered B0 values per method.
    """

    # Fetch physical properties from PlasmaPy
    p_obj = ppp.Particle(particle)
    particle_mass = p_obj.mass.value  # in kg
    particle_charge = abs(p_obj.charge.value)  # in Coulombs

    spectrum_paths = glob.glob(str(h5_directory / "*.h5"))
    if not spectrum_paths:
        raise FileNotFoundError(f"No '.h5' files found in {h5_directory}")

    # Load datasets
    inputData = [xr.open_datatree(f) for f in spectrum_paths]
    
    inputFreqs = np.array(inputData[0]["spectra"].F)  # Expected in MHz
    if len(inputData) > 1:
        inputParams = np.concat(
            (np.array(inputData[0]["parameters"].X.data, dtype=np.float64), 
             np.array(inputData[1]["parameters"].X.data, dtype=np.float64)), 
            axis=1
        )
        inputSpectra = np.concat(
            (np.array(inputData[0]["spectra"].Y.data, dtype=np.float64), 
             np.array(inputData[1]["parameters"].X.data, dtype=np.float64)), 
            axis=1
        ).T
    else:
        inputParams = np.array(inputData[0]["parameters"].X.data, dtype=np.float64)
        inputSpectra = np.array(inputData[0]["spectra"].Y.data, dtype=np.float64).T

    # Close open xarray trees
    for tree in inputData:
        tree.close()

    originalB0s = inputParams[0]
    
    # Storage for output results
    recoveredB0s_comb = []
    recoveredB0s_acf = []
    recoveredB0s_medianPeak = []

    # Conversion factor: fc [MHz] -> B0 [T]
    # fc = (q * B0) / (2 * pi * m)  =>  B0 = (fc * 1e6 * 2 * pi * m) / q
    fc_to_B0 = lambda fc_mhz: (fc_mhz * 1e6 * 2.0 * np.pi * particle_mass) / particle_charge
    B0_to_fc = lambda B0_T: (particle_charge * B0_T) / (2.0 * np.pi * particle_mass * 1e6)

    df = np.median(np.diff(inputFreqs))  # Sampling step in MHz

    for s in range(inputSpectra.shape[0]):
        spectrum = inputSpectra[s]
        # Normalize spectrum baseline and peak amplitude
        spec_norm = spectrum - np.min(spectrum)
        if np.max(spec_norm) > 0:
            spec_norm /= np.max(spec_norm)

        min_fc_mhz = B0_to_fc(b0_search_range[0])
        max_fc_mhz = B0_to_fc(b0_search_range[1])
        all_peaks, _ = find_peaks(
            spec_norm, 
            height=float(0.1 * spec_norm.max()), 
            prominence=float(0.3 * spec_norm.max()), 
            distance=int((min_fc_mhz * 0.9) / df)
        )
        # if len(all_peaks) > 5:
        #     result = np.zeros_like(spec_norm)
        #     result[all_peaks] = spec_norm[all_peaks]
        #     spec_norm = result
        
        # -------------------------------------------------------------
        # METHOD 1: Comb Template Matching (Global Optimization)
        # -------------------------------------------------------------
        def comb_objective(trial_B0):
            trial_fc = B0_to_fc(trial_B0)
            harmonics = np.arange(1, num_comb_harmonics + 1) * trial_fc
            
            # Create Gaussian comb centered at harmonic positions
            comb = np.zeros_like(inputFreqs)
            sigma = 0.05 * trial_fc  # Width of each comb peak (~5% of fundamental)
            for h_freq in harmonics:
                if h_freq <= inputFreqs[-1]:
                    comb += np.exp(-0.5 * ((inputFreqs - h_freq) / sigma) ** 2)
            
            # Maximize dot product between comb and normalized spectrum
            return -np.sum(spec_norm * comb)

        res = minimize_scalar(comb_objective, bounds=b0_search_range, method='bounded')
        b0_comb = res.x
        recoveredB0s_comb.append(b0_comb)

        # -------------------------------------------------------------
        # METHOD 2: Autocorrelation Function (ACF)
        # -------------------------------------------------------------
        corr = correlate(spec_norm, spec_norm, mode='full')
        corr = corr[len(spec_norm)-1:]  # Keep positive lags
        lags_mhz = np.arange(len(corr)) * df

        # Exclude immediate zero-lag peak region (search above half minimum expected fc)
        valid_idx = (lags_mhz >= (min_fc_mhz * 0.5)) & (lags_mhz <= (max_fc_mhz * 1.1))

        if np.any(valid_idx):
            acf_peaks, peak_properties = find_peaks(corr[valid_idx], height = 0.05 * corr.max(), prominence=0.05 * corr.max())
            if len(acf_peaks) > 0:
                # first_lag_idx = np.where(valid_idx)[0][acf_peaks[0]]
                # fc_acf = lags_mhz[first_lag_idx]
                max_lag_idx = np.where(valid_idx)[0][acf_peaks[np.argmax(peak_properties["peak_heights"])]]
                fc_acf = lags_mhz[max_lag_idx]
                b0_acf = fc_to_B0(fc_acf)
            else:
                b0_acf = np.nan
        else:
            b0_acf = np.nan
        recoveredB0s_acf.append(b0_acf)

        # -------------------------------------------------------------
        # METHOD 3: Robust Median Peak Separation
        # -------------------------------------------------------------
        all_peaks, _ = find_peaks(
            spec_norm, 
            height=float(0.1 * spec_norm.max()), 
            prominence=float(0.3 * spec_norm.max()), 
            distance=int((min_fc_mhz * 0.9) / df)
        )
        
        if len(all_peaks) > 1:
            peak_freqs = inputFreqs[all_peaks]
            peak_seps = np.diff(peak_freqs)
            
            # Using median instead of mean avoids sensitivity to skipped harmonics
            fc_median = np.median(peak_seps)
            b0_median = fc_to_B0(fc_median)
            if b0_median > 5.0:
                b0_median = np.nan
        else:
            b0_median = np.nan
        recoveredB0s_medianPeak.append(b0_median)

        # -------------------------------------------------------------
        # Optional Diagnostic Plotting
        # -------------------------------------------------------------
        if plot_methods:
            fig, ax = plt.subplots(1, 2, figsize=(14, 5))
            
            # Spectrum & Comb
            best_fc = B0_to_fc(b0_comb)
            ax[0].plot(inputFreqs, spec_norm, label="ICE Spectrum")
            if len(all_peaks) > 0:
                ax[0].scatter(inputFreqs[all_peaks], spec_norm[all_peaks], color='red', marker='x', label="Detected Peaks")
            
            # Overlay fitted comb markers
            for h in range(1, num_comb_harmonics + 1):
                ax[0].axvline(h * best_fc, color='k', linestyle='--', alpha=0.3, label = "comb harmonics")
                ax[0].axvline(h * B0_to_fc(originalB0s[s]), color='purple', linestyle=':', alpha=0.8, label = "true harmonics")
                
            ax[0].set_title(f"Spectrum #{s} | Comb B0: {b0_comb:.3f} T (True: {originalB0s[s]:.3f} T)")
            ax[0].set_xlabel("Frequency (MHz)")
            ax[0].set_ylabel("Normalized Power")
            ax[0].set_xlim(0,inputFreqs.max())
            # Get all handles and labels from the axes
            handles, labels = ax[0].get_legend_handles_labels()
            # Filter out duplicates while preserving order
            by_label = dict(zip(labels, handles))
            ax[0].legend(by_label.values(), by_label.keys())

            # Autocorrelation
            ax[1].plot(lags_mhz, corr / corr.max())
            if not np.isnan(b0_acf):
                best_fc = B0_to_fc(b0_acf)
                ax[1].axvline(best_fc, color='r', linestyle='--', label=f"1st Lag Peak ({best_fc:.2f} MHz)")
            ax[1].axvline(B0_to_fc(originalB0s[s]), color='purple', alpha=0.8, linestyle=':', label=f"True G.F. ({B0_to_fc(originalB0s[s]):.2f} MHz)")
            ax[1].set_title("Autocorrelation (Lag Space)")
            ax[1].set_xlabel(r"Frequency Lag $\Delta f$ (MHz)")
            ax[1].set_xlim(0, max(lags_mhz) * 0.5)
            ax[1].legend()

            plt.tight_layout()
            plt.show()

    total_cases = len(originalB0s)

    # Convert results to arrays
    recoveredB0s_comb = np.array(recoveredB0s_comb)
    recoveredB0s_acf = np.array(recoveredB0s_acf)
    recoveredB0s_medianPeak = np.array(recoveredB0s_medianPeak)

    # Global plot style configuration matching original script
    plt.rcParams.update({'axes.titlesize': 20.0})
    plt.rcParams.update({'axes.labelsize': 20.0})
    plt.rcParams.update({'xtick.labelsize': 20.0})
    plt.rcParams.update({'ytick.labelsize': 20.0})
    plt.rcParams.update({'legend.fontsize': 20.0})

    # Plot 1: Comb Template Matching
    plt.figure(figsize=(12, 10))
    plt.scatter(originalB0s, recoveredB0s_comb, label="Recovered by Comb Template Fitting")
    plt.plot(sorted(originalB0s), sorted(originalB0s), "k--", label="ideal B0 recovery")
    plt.xlabel("Original B0")
    plt.ylabel("Recovered B0")
    plt.legend()
    plt.title(f"Linear data, LHD range -- r^2: {r2_score(originalB0s, recoveredB0s_comb):.4f}")
    plt.show()

    # Plot 2: Autocorrelation (handling NaN dropouts if any)
    valid_acf = ~np.isnan(recoveredB0s_acf)
    pct_recovered = (np.count_nonzero(valid_acf) / total_cases) * 100.0
    plt.figure(figsize=(12, 10))
    plt.scatter(originalB0s[valid_acf], recoveredB0s_acf[valid_acf], label="Recovered by Autocorrelation Peak")
    plt.plot(sorted(originalB0s), sorted(originalB0s), "k--", label="ideal B0 recovery")
    plt.xlabel("Original B0")
    plt.ylabel("Recovered B0")
    plt.legend()
    acf_r2 = r2_score(originalB0s[valid_acf], recoveredB0s_acf[valid_acf]) if np.sum(valid_acf) > 1 else 0.0
    plt.title(f"Linear data, LHD range -- r^2: {acf_r2:.4f} ({pct_recovered:.2f}% recovered)")
    plt.show()

    # Plot 3: Median Peak Separation
    valid_med = ~np.isnan(recoveredB0s_medianPeak)
    pct_recovered = (np.count_nonzero(valid_med) / total_cases) * 100.0
    plt.figure(figsize=(12, 10))
    plt.scatter(originalB0s[valid_med], recoveredB0s_medianPeak[valid_med], label="Recovered by Median Peak Separation")
    plt.plot(sorted(originalB0s), sorted(originalB0s), "k--", label="ideal B0 recovery")
    plt.xlabel("Original B0")
    plt.ylabel("Recovered B0")
    plt.legend()
    med_r2 = r2_score(originalB0s[valid_med], recoveredB0s_medianPeak[valid_med]) if np.sum(valid_med) > 1 else 0.0
    plt.title(f"Linear data, LHD range -- r^2: {med_r2:.4f} ({pct_recovered:.2f}% recovered)")
    plt.show()

    return {
        "original_B0": originalB0s,
        "recovered_B0_comb": recoveredB0s_comb,
        "recovered_B0_acf": recoveredB0s_acf,
        "recovered_B0_medianPeak": recoveredB0s_medianPeak
    }

def estimate_B0_robust_netcdf(
    netcdf_directory : Path, 
    fields : list, 
    particle : str = "He-4 2+", 
    max_frequency_mHz : float = None,
    b0_search_range: tuple = (0.5, 5.0),
    num_harmonics: int = 15,
) -> dict:

    plt.rcParams.update({"axes.titlesize": 20.0})
    plt.rcParams.update({"axes.labelsize": 20.0})
    plt.rcParams.update({"xtick.labelsize": 20.0})
    plt.rcParams.update({"ytick.labelsize": 20.0})
    plt.rcParams.update({"legend.fontsize": 20.0})

    combined_statsFiles = glob.glob(str(netcdf_directory / "*.nc"))
    originalB0s = {f: [] for f in fields}
    recoveredB0s_max = {f: [] for f in fields}
    recoveredB0s_firstPeak = {f: [] for f in fields}
    recoveredB0s_peakSeparation = {f: [] for f in fields}
    recoveredB0s_hybrid = {f: [] for f in fields}

    # NEW: Storage for Comb and Autocorrelation methods
    recoveredB0s_comb = {f: [] for f in fields}
    recoveredB0s_autocorr = {f: [] for f in fields}

    variedParams = {"pitch": [], "backgroundDensity": [], "beamFraction": []}

    # Extract target particle mass and charge in SI units
    particle_obj = ppp.Particle(particle)
    q = particle_obj.charge.value
    m = particle_obj.mass.value

    for simFile in combined_statsFiles:
        data_xr = xr.open_datatree(simFile, engine="netcdf4")
        for key, value in variedParams.items():
            if key in data_xr.attrs:
                value.append(data_xr.attrs[key])

        for field in fields:
            simName = Path(simFile).name
            data: xr.DataArray = data_xr[field]

            # Convert coordinates to SI (Hz) if needed
            known_B0 = float(data_xr.B0strength)
            originalB0s[field].append(known_B0)

            gyrofrequency_in_SI = ppf.gyrofrequency(
                known_B0 * u.T, particle=particle
            )

            if data.coords["frequency"].max() < 100.0:  # Not already in SI
                si_coords = data.coords["frequency"] * gyrofrequency_in_SI
                data = data.assign_coords({"frequency": si_coords})

            if max_frequency_mHz:
                data = data[data["frequency"] <= (max_frequency_mHz * 1e6)]

            # Compute FFT Power Spectrum using XRFT
            og_spec: xr.DataArray = xrft.xrft.fft(
                data, true_amplitude=False, true_phase=True, window=None
            )
            spec: xr.DataArray = np.abs(og_spec)
            spec = spec.sel(freq_frequency=slice(0.0, None))
            spec = spec.isel(freq_frequency=slice(10, None))

            # --- 1. Existing Methods (Max, First Peak, Peak Separation) ---
            maxFreqFreq = float(spec.idxmax().data)
            maxFreq = 1.0 / maxFreqFreq
            recovered_B0 = (maxFreq * m) / q

            height_proportion = 0.1
            prominence_proportion = 0.1
            p, _ = find_peaks(
                spec.data,
                height=float(height_proportion * spec.max().data),
                prominence=float(prominence_proportion * spec.max().data),
                distance=np.max([1.0, 0.1 * len(spec)]),
            )

            if len(p) > 0:
                peakFreqFreq = float(spec.freq_frequency[p[0]])
                peakFreq = 1.0 / peakFreqFreq
                recovered_B0_2 = (peakFreq * m) / q

                sepFreqFreq = float(spec.freq_frequency[p[0]])
                sepFreq = 1.0 / sepFreqFreq
                if len(p) > 1:
                    seps = [
                        float(spec.freq_frequency[p[i]])
                        - float(spec.freq_frequency[p[i - 1]])
                        for i in range(1, len(p))
                    ]
                    sepFreq = 1.0 / np.mean(seps)
                recovered_B0_3 = (sepFreq * m) / q
            else:
                recovered_B0_2 = recovered_B0
                recovered_B0_3 = recovered_B0

            # --- 2. NEW METHOD A: Comb Evaluation ---
            # Search candidate B0 values and find which fundamental gyrofrequency
            # matches the largest power sum across its integer harmonics.
            freq_axis = spec.freq_frequency.data  # Time period axis (1/f)
            psd_data = spec.data

            # Convert candidate B0 space to fundamental frequencies
            b0_candidates = np.linspace(
                b0_search_range[0], b0_search_range[1], 1000
            )
            f_ci_candidates = (q * b0_candidates) / (2 * np.pi * m)

            best_comb_score = -1.0
            recovered_B0_comb = b0_candidates[0]

            for cand_b0, f_ci in zip(b0_candidates, f_ci_candidates):
                # Calculate harmonic frequencies for this candidate B0
                harmonics = np.arange(1, num_harmonics + 1) * f_ci
                harmonic_periods = 1.0 / harmonics

                # Accumulate power near each harmonic period
                comb_power = 0.0
                for period in harmonic_periods:
                    idx = np.argmin(np.abs(freq_axis - period))
                    comb_power += psd_data[idx]

                if comb_power > best_comb_score:
                    best_comb_score = comb_power
                    recovered_B0_comb = cand_b0

            # --- 3. NEW METHOD B: Autocorrelation (Fundamental Period Finder) ---
            # Compute autocorrelation of the spectrum array to detect repeating harmonic spacing
            norm_psd = psd_data - np.mean(psd_data)
            autocorr = signal.fftconvolve(norm_psd, norm_psd[::-1], mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]  # Keep positive lags

            # Find primary autocorrelation peak after lag 0
            ac_peaks, _ = find_peaks(
                autocorr,
                height=0.05 * np.max(autocorr),
                distance=max(1, int(0.02 * len(autocorr))),
            )

            if len(ac_peaks) > 0:
                lag_idx = ac_peaks[0]
                period_spacing = abs(freq_axis[lag_idx] - freq_axis[0])
                if period_spacing > 0:
                    f_fundamental = 1.0 / period_spacing
                    recovered_B0_autocorr = (
                        2 * np.pi * f_fundamental * m
                    ) / q
                else:
                    recovered_B0_autocorr = recovered_B0
            else:
                recovered_B0_autocorr = recovered_B0

            # --- Record Results ---
            recoveredB0s_max[field].append(recovered_B0)
            recoveredB0s_firstPeak[field].append(recovered_B0_2)
            recoveredB0s_peakSeparation[field].append(recovered_B0_3)
            recoveredB0s_hybrid[field].append(
                np.mean([recovered_B0_3, recovered_B0_2])
            )
            recoveredB0s_comb[field].append(recovered_B0_comb)
            recoveredB0s_autocorr[field].append(recovered_B0_autocorr)

        data_xr.close()

    # --- Print Statistical Summary ---
    for field in fields:
        print(f"\n================ {field} Results ================")
        print(
            f"Max peak B0 recovery R2:        {r2_score(originalB0s[field], recoveredB0s_max[field]):.3f}"
        )
        print(
            f"First peak B0 recovery R2:      {r2_score(originalB0s[field], recoveredB0s_firstPeak[field]):.3f}"
        )
        print(
            f"Peak separation B0 recovery R2: {r2_score(originalB0s[field], recoveredB0s_peakSeparation[field]):.3f}"
        )
        print(
            f"Hybrid B0 recovery R2:          {r2_score(originalB0s[field], recoveredB0s_hybrid[field]):.3f}"
        )
        print(
            f"Comb Method B0 recovery R2:     {r2_score(originalB0s[field], recoveredB0s_comb[field]):.3f}"
        )
        print(
            f"Autocorr Method B0 recovery R2: {r2_score(originalB0s[field], recoveredB0s_autocorr[field]):.3f}"
        )

    # --- Plot Comparisons ---
    plt.figure(figsize=(9, 6))

    for field in fields:
        colour = "red" if "Bz" in field else "green" if "Ex" in field else "blue"
        mk = "o" if "Bz" in field else "x" if "Ex" in field else "+"

        ogs = originalB0s[field]
        recs_comb = recoveredB0s_comb[field]

        result = linregress(ogs, recs_comb)
        plt.scatter(
            ogs,
            recs_comb,
            color=colour,
            alpha=0.9,
            marker=mk,
            s=100,
            label=f"{field} (Comb Method R²={result.rvalue**2:.2f})",
        )
        sortB0 = sorted(ogs)

    plt.plot(
        sortB0,
        sortB0,
        color="black",
        alpha=0.9,
        linestyle="dashed",
        label="Ideal Prediction",
    )
    plt.xlabel(r"Original $B_0$ [T]")
    plt.ylabel(r"Recovered $B_0$ [T]")
    plt.title(r"$B_0$ Recovery Performance Comparison")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    SMALL_SIZE = 10
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 22
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Run python setup.py -h for list of possible arguments
    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--dir",
        action="store",
        help="Directory containing analysis of all simulations. Expects folders of angles, each with data and plots folders, or .dat files for experimental data.",
        required = True,
        type=Path
    )
    parser.add_argument(
        "--outputDir",
        action="store",
        help="Output directory.",
        required = False,
        type=Path
    )
    parser.add_argument(
        "--plotIciness",
        action="store_true",
        help="Plot already calculated and saved ICEiness characteristics for combined spectra.",
        required = False
    )
    parser.add_argument(
        "--recoverB0",
        action="store_true",
        help="Attempt to analytically recover B0 from ICE analysis spectra.",
        required = False
    )
    parser.add_argument(
        "--fields",
        action="store",
        help="Fields on which to calculate iciness.",
        required = False,
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--particle",
        action="store",
        help="Particle used as fast ion in simulation -- likely to be either alpha ('He-4 2+') or deuterium ('D+').",
        required = False,
        type=str
    )
    parser.add_argument(
        "--truncateToMHz",
        action="store",
        help="Truncate spectra to this frequency in MHz.",
        required = False,
        type=float,
        default=None
    )
    
    args = parser.parse_args()

    if args.recoverB0:
        if len(glob.glob(str(args.dir / "*.dat"))) > 0:
            estimate_B0_from_csv_dat_spectra(args.dir)
        elif len(glob.glob(str(args.dir / "*.csv"))) > 0:
            estimate_B0_from_netcdf_spectra(args.dir, args.fields)
        elif len(glob.glob(str(args.dir / "*.h5"))) > 0:
            # estimate_B0_from_h5_spectra(args.dir, args.particle)
            estimate_B0_robust(args.dir, args.particle)
        elif len(glob.glob(str(args.dir / "*.nc"))) > 0:
            estimate_B0_from_netcdf_spectra(args.dir, args.fields, args.particle, args.truncateToMHz)
            # estimate_B0_robust_netcdf(netcdf_directory = args.dir, fields = args.fields, particle = args.particle)