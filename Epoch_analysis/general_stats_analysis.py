import glob
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import argparse
import netCDF4 as nc
import xarray as xr
import xrft
import epoch_utils as eu
from scipy.stats import linregress
from scipy.signal import find_peaks
from sklearn.metrics import root_mean_squared_error, r2_score
import plasmapy.formulary.frequencies as ppf
import plasmapy.particles as ppp
import astropy.units as u

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas

def estimate_B0_from_netcdf_spectra(netcdf_directory : Path, fields : list, particle : str = "He-4 2+"):

    plt.rcParams.update({'axes.titlesize': 20.0})
    plt.rcParams.update({'axes.labelsize': 20.0})
    plt.rcParams.update({'xtick.labelsize': 20.0})
    plt.rcParams.update({'ytick.labelsize': 20.0})
    plt.rcParams.update({'legend.fontsize': 20.0})

    combined_statsFiles = glob.glob(str(netcdf_directory / "data" / "*_combined_stats.nc"))
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

            # Convert to SI (on original spectrum)
            known_B0 = float(data_xr.B0strength)
            originalB0s[field].append(known_B0)
            gyrofrequency_in_SI = ppf.gyrofrequency(known_B0 * u.T, particle = particle)
            # print(f"Gyrofrequency in SI: {gyrofrequency_in_SI}")
            si_coords = data.coords["frequency"] * gyrofrequency_in_SI
            # print(f"Frequency coords in SI: {si_coords}")
            data = data.assign_coords({"frequency" : si_coords})
            # data.plot()
            # plt.show()
            
            og_spec : xr.DataArray = xrft.xrft.fft(data, true_amplitude=False, true_phase=True, window=None)
            spec : xr.DataArray = np.abs(og_spec)
            # print(spec.coords["freq_frequency"])
            spec = spec.sel(freq_frequency = slice(0.0, None))
            spec = spec.where(spec.freq_frequency >= float(0.11*spec.freq_frequency.max().data), other=0.0)
            # spec = spec.isel(freq_frequency = slice(int(0.13*len(spec.coords["freq_frequency"])), None))
            
            # Max freq method
            maxFreqFreq = float(spec.idxmax().data)
            maxFreq = 1.0 / maxFreqFreq
            maxPower = spec.max().data
            # Recover B0
            recovered_B0 = (maxFreq * ppp.alpha.mass) / ppp.alpha.charge

            # Peak finding method
            p, pd = find_peaks(spec.data, height=float(0.1*spec.max().data), prominence=float(0.1*spec.max().data), distance = 0.1*len(spec))
            peakFreqFreq = float(spec.freq_frequency[p[0]])
            peakFreq = 1.0/peakFreqFreq
            recovered_B0_2 = (peakFreq * ppp.alpha.mass) / ppp.alpha.charge

            # Peak separation method
            sepFreqFreq = float(spec.freq_frequency[p[0]])
            sepFreq = 1.0 / sepFreqFreq
            if len(p) > 1:
                seps = []
                for i in range(1, len(p)):
                    seps.append(float(spec.freq_frequency[p[i]]) - float(spec.freq_frequency[p[i-1]]))
                sepFreq = 1.0 / np.mean(seps)
            recovered_B0_3 = (sepFreq * ppp.alpha.mass) / ppp.alpha.charge
            
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
    
    args = parser.parse_args()

    if args.recoverB0:
        if len(glob.glob(str(args.dir / "*.dat"))) > 0:
            estimate_B0_from_csv_dat_spectra(args.dir)
        else:
            estimate_B0_from_netcdf_spectra(args.dir, args.fields)