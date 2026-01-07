import csv
import glob
import os
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
import pycatch22
import plasmapy.formulary.frequencies as ppf
import plasmapy.particles as ppp
import astropy.units as u

def estimate_B0_from_spectra(combined_directory : Path, fields : list, particle : str = "He-4 2+"):
    combined_statsFiles = glob.glob(str(combined_directory / "data" / "*_combined_stats.nc"))
    originalB0s = {f : [] for f in fields}
    recoveredB0s = {f : [] for f in fields}
    variedParams = {"pitch" : [], "backgroundDensity" : [], "beamFraction" : []}

    for simFile in combined_statsFiles:
        data_nc = nc.Dataset(simFile, mode="a")
        data_xr = xr.open_datatree(simFile, engine="netcdf4")
        for key, value in variedParams.items():
            value.append(data_xr.attrs[key])

        for field in fields:
            filename = Path(simFile).name
            print(f"File: {filename}, Field: {field}")
            data : xr.DataArray = data_xr[field]

            # Convert to SI (on original spectrum)
            known_B0 = float(data_xr.B0strength)
            originalB0s[field].append(known_B0)
            gyrofrequency_in_SI = ppf.gyrofrequency(known_B0 * u.T, particle = particle)
            print(f"Gyrofrequency in SI: {gyrofrequency_in_SI}")
            si_coords = data.coords["frequency"] * gyrofrequency_in_SI
            # print(f"Frequency coords in SI: {si_coords}")
            data = data.assign_coords({"frequency" : si_coords})
            # data.plot()
            # plt.show()
            
            og_spec : xr.DataArray = xrft.xrft.fft(data, true_amplitude=False, true_phase=True, window=None)
            spec = np.abs(og_spec)           
            spec = spec.sel(freq_frequency = slice(0.0, None))
            spec = spec.isel(freq_frequency = slice(50, None))
            maxFreqFreq = float(spec.idxmax().data)
            maxFreq = 1.0 / maxFreqFreq
            maxPower = spec.max().data
            print(f"Max: {maxPower}, Max index: {spec.argmax().data}, Max coord: {maxFreqFreq * 1/gyrofrequency_in_SI.unit} Max coord in OG units: {maxFreq * gyrofrequency_in_SI.unit}")
            assert spec.coords["freq_frequency"][-1] == np.max(spec.coords["freq_frequency"])
            # plt.scatter(maxFreqFreq, maxPower, color = "red", marker = "x", label = "peak power")
            # spec.plot()
            # plt.legend()
            # plt.show()

            # Recover B0
            recovered_B0 = (maxFreq * ppp.alpha.mass) / ppp.alpha.charge
            print(f"Original B0: {data_xr.B0strength * u.T}, recovered B0 : {recovered_B0}")

            # Record
            recoveredB0s[field].append(recovered_B0.value)
            # data_nc[field].recovered_B0 = recovered_B0

        data_xr.close()
        data_nc.close()

    for field in fields:
        ogs = originalB0s[field]
        recs = recoveredB0s[field]
        squared_errors = (np.array(recs) - np.array(ogs))**2
        
        # Plot original vs. recovered B0s
        result = linregress(ogs, recs)
        plt.figure(figsize=(8,8))
        plt.title(f"{field}: B0\n(r2 = {result.rvalue**2:.3f}, S.E. = {result.stderr:.3f})")
        plt.scatter(ogs, recs, marker = "x", color = "red")
        sortB0 = sorted(ogs)
        plt.plot(sortB0, sortB0, color = "blue", alpha = 0.9, linestyle = "dashed", label = "perfect prediciton")
        plt.xlabel("Original B0 [T]")
        plt.ylabel("Recovered B0 [T]")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

        # # Plot errors with other varied quantities
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

def plot_iciness(combined_directory : Path, fields : list, iceMetrics : list = None):

    combined_statsFiles = glob.glob(str(combined_directory / "data" / "*_combined_stats.nc"))
    plots_folder = combined_directory / "plots" / "iciness"
    first_file = xr.open_datatree(combined_statsFiles[0], engine="netcdf4")
    
    # Check spectra and energy groups for ICE metrics
    metrics_fully_qualified = set()
    if iceMetrics:
        for f in fields:
            for attribute in first_file[f].attrs:
                if str(attribute) in iceMetrics:
                    metrics_fully_qualified.add(f"{f}_{attribute}")
        for attribute in first_file["/Energy"].attrs:
            if str(attribute) in iceMetrics:
                metrics_fully_qualified.add(f"Energy_{attribute}")
    else:
        for f in fields:
            for attribute in first_file[f].attrs:
                if str(attribute).startswith("ICEmetric"):
                    metrics_fully_qualified.add(f"{f}_{attribute}")
        for attribute in first_file["/Energy"].attrs:
            if str(attribute).startswith("ICEmetric"):
                metrics_fully_qualified.add(f"Energy_{attribute}")

    iciness_by_field = { 
        m : {file : [] for file in [Path(c).name for c in combined_statsFiles]} for m in metrics_fully_qualified
    }
    # energyTransfer = {file : [] for file in [Path(c).name for c in combined_statsFiles]}
    B0strength = {file : [] for file in [Path(c).name for c in combined_statsFiles]}
    pitch = {file : [] for file in [Path(c).name for c in combined_statsFiles]}
    density = {file : [] for file in [Path(c).name for c in combined_statsFiles]}
    beamFrac = {file : [] for file in [Path(c).name for c in combined_statsFiles]}

    for simFile in combined_statsFiles:
        data_xr = xr.open_datatree(simFile, engine="netcdf4")

        filename = Path(simFile).name
        print(f"File: {filename}")

        # Record
        for metric in iciness_by_field.keys():
            metric_parts = metric.split('_ICEmetric_')
            metric_short = f"ICEmetric_{metric_parts[1]}" if len(metric_parts) > 1 else metric_parts
            spectrum_name = metric_parts[0]
            iciness_by_field[metric][filename].append(float(data_xr[spectrum_name].attrs[metric_short]))

        # # Energy
        # energyTransfer[filename].append(float(data_xr["Energy"].attrs["ICEmetric_energyTransfer"]))
        # iciness_by_field["ICEmetric_energyTransfer"][filename].append(float(data_xr["Energy"].attrs["ICEmetric_energyTransfer"]))

        # Simulation params
        B0strength[filename].append(data_xr.B0strength)
        pitch[filename].append(data_xr.pitch)
        density[filename].append(data_xr.backgroundDensity)
        beamFrac[filename].append(data_xr.beamFraction)

        data_xr.close()

    baselines = {
        "B0" : B0strength,
        "pitch" : pitch,
        "density" : density,
        "beam_frac" : beamFrac
    }
    correlations = []
    iciness_by_metric = {k : [d[0] for d in v.values()] for k, v in iciness_by_field.items()}
    for baseline_name, baseline_data in baselines.items():
        correlations = eu.correlate_and_plot_iciness_vs_baseline(iciness_by_metric, baseline_name, [v[0] for v in baseline_data.values()], plots_folder, "all", correlations, doPlot=True)

    with open(plots_folder / "iciness_correlations.csv", 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, correlations[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(correlations)

    metric_correlations = []
    print("METRIC CORRELATIONS: ")
    for metric in iciness_by_metric.keys():
        metric_parts = metric.split('_ICEmetric_')
        metric_short = f"ICEmetric_{metric_parts[1]}"
        data = [c["r2"] for c in correlations if c["metric"] == metric_short]
        mean = np.mean(data)
        meanAbs = np.mean(np.abs(data))
        print(f"{metric}:   | raw r2 = {mean:.5f}, abs r2 = {meanAbs:.5f}")
        metric_correlations.append({"metric" : metric, "mean_r2" : mean, "mean_abs_r2" : meanAbs})

    with open(plots_folder / "mean_iciness_correlations_by_metric.csv", 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, metric_correlations[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(metric_correlations)

    baseline_correlations = []
    print("BASELINE CORRELATIONS: ")
    for baseline in baselines.keys():
        data = [c["r2"] for c in correlations if c["baseline"] == baseline]
        mean = np.mean(data)
        meanAbs = np.mean(np.abs(data))
        print(f"{baseline}:   | raw r2 = {mean:.5f}, abs r2 = {meanAbs:.5f}")
        baseline_correlations.append({"baseline" : baseline, "mean_r2" : mean, "mean_abs_r2" : meanAbs})

    with open(plots_folder / "mean_iciness_correlations_by_baseline.csv", 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, baseline_correlations[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(baseline_correlations)

def calculate_iciness(combined_directory : Path, fields : list):

    combined_statsFiles = glob.glob(str(combined_directory / "data" / "*_combined_stats.nc"))
    maxCoords = []
    removeFirst = 0.5
    harmonic_tolerance = 0.1

    for simFile in combined_statsFiles:
        data_nc = nc.Dataset(simFile, mode="a")
        data_xr = xr.open_datatree(simFile, engine="netcdf4")

        for field in fields:
            fundamental_index_count = 0
            harmonic_index_count = 0
            filename = Path(simFile).name
            print(f"File: {filename}, Field: {field}")
            data : xr.DataArray = data_xr[field]
            spec : xr.DataArray = xrft.xrft.fft(data, true_amplitude=False, true_phase=True, window=None)
            spec = np.abs(spec)
            spec = spec.sel(freq_frequency = slice(removeFirst, None)) # Eliminate very low-frequency components (noise)
            maxFreq = spec.idxmax().data
            maxCoords.append(maxFreq)
            print(f"Max: {spec.max().data}, Max index: {spec.argmax().data}, Max coord: {maxFreq}")
            assert spec.coords["freq_frequency"][-1] == np.max(spec.coords["freq_frequency"])
            spec.plot()
            plt.show()

            # Isolate indices of fundamental frequency
            fundamentals = spec.sel(freq_frequency=slice(1.0 - harmonic_tolerance, 1.0 + harmonic_tolerance))
            fundamental_index_count = fundamentals.size
            expected_pwr_in_fundamental = spec.sum().data * (fundamental_index_count / spec.size) # Power that would be expected in this range
            pwr_in_fundamental = fundamentals.sum().data - expected_pwr_in_fundamental
            pwr_in_fundamental_pct = 100.0 * (fundamentals.sum().data / spec.sum().data)
            
            harmonic_nums = np.arange(1.0, np.rint(spec.coords["freq_frequency"].max()) + 1)
            harmonics = [spec.sel(freq_frequency=slice(h - harmonic_tolerance,h + harmonic_tolerance)) for h in harmonic_nums]
            harmonic_peaks = []
            spec.plot()
            for h in harmonics:
                p, pd = find_peaks(h.data, height=float(0.05*h.max().data))
                if len(p) > 0:
                    harmonic_peaks.append(h.data[p[np.argmax(pd["peak_heights"])]])
                    plt.scatter(h.coords["freq_frequency"][p], h.data[p], color="r", marker='x')
                plt.axvspan(h.coords["freq_frequency"][0], h.coords["freq_frequency"][-1], color = "orange", alpha = 0.5)
            plt.show()
            expected_pwr_in_harmonics = spec.sum().data * (harmonic_index_count / spec.size)
            pwr_in_harmonics = (np.sum([d.sum().data for d in harmonics])) - expected_pwr_in_harmonics
            pwr_in_harmonics_pct = 100.0 * (pwr_in_harmonics / spec.sum().data)
            print(f"Power in fundamental gyrofrequency : {pwr_in_fundamental} units ({pwr_in_fundamental_pct:.3f}%), Power in harmonics of gyrofrequency: {pwr_in_harmonics} units ({pwr_in_harmonics_pct:.3f}%)")
            print(f"Peak harmonic powers: {harmonic_peaks}")

            fundamental_peak_mean_ratio = float(harmonic_peaks[0] / spec.mean().data)
            fundamental_peak_floor_ratio = float(harmonic_peaks[0] / spec.min().data)
            harmonics_peak_mean_ratio = float(np.mean(harmonic_peaks) / spec.mean().data)
            harmonics_peak_floor_ratio = float(np.mean(harmonic_peaks) / spec.min().data)

            # Get periodicity indicators from pycatch22
            c22 = pycatch22.catch22_all(data.to_numpy())
            ### Incremendal difference measures
            ##### High-fluctuation, proportion of difference mags that are greater than 4% of the SD. Lower is more extreme peaks, so take
            ##### 1.0 - high fluct to indicate more Icey
            high_fluct_inv = 1.0 - c22["values"][c22["names"].index("MD_hrv_classic_pnn40")]
            ### Linear autocorrelations
            acf_ts = c22["values"][c22["names"].index("CO_f1ecac")]
            acf_firstmin_ts = c22["values"][c22["names"].index("CO_FirstMin_ac")]

            # Record
            data_nc["Energy"].ICEmetric_energyTransfer = data_xr["Energy"].fastIonToBackgroundTransfer_percentage
            data_nc[field].ICEmetric_fundamentalPower = fundamentals.sum().data
            data_nc[field].ICEmetric_fundamentalPower_pct = pwr_in_fundamental_pct
            data_nc[field].ICEmetric_fundamentalPeakMeanRatio = fundamental_peak_mean_ratio
            data_nc[field].ICEmetric_fundamentalPeakFloorRatio = fundamental_peak_floor_ratio
            data_nc[field].ICEmetric_harmonicPower = np.sum([d.sum().data for d in harmonics])
            data_nc[field].ICEmetric_harmonicPower_pct = pwr_in_harmonics_pct
            data_nc[field].ICEmetric_harmonicPeakMeanRatio = harmonics_peak_mean_ratio
            data_nc[field].ICEmetric_harmonicPeakFloorRatio = harmonics_peak_floor_ratio
            data_nc[field].ICEmetric_c22HighFluct = high_fluct_inv
            data_nc[field].ICEmetric_c22Acf = acf_ts
            data_nc[field].ICEmetric_c22AcfFirstMin = acf_firstmin_ts

        data_xr.close()
        data_nc.close()
    
    print(f"Max frequency mean: {np.mean(maxCoords)} median: {np.median(maxCoords)} var: {np.var(maxCoords)} sd: {np.std(maxCoords)}")

def combine_spectra(dataDirectory : Path, outputDirectory : Path):

    # Get folders for each angle
    angle_folders = glob.glob(str(dataDirectory / "9[0-9]/data/"))

    assert len(angle_folders) > 0

    # Get first angle
    first_angle_folder = Path(angle_folders[0])

    data_files = glob.glob(str(first_angle_folder / "*.nc"))
    individual_data_filenames = [Path(n).name for n in data_files]

    outputDataDirectory = outputDirectory / "data"
    if not os.path.exists(outputDataDirectory):
        os.mkdir(outputDataDirectory)
    outputPlotDirectory = outputDirectory / "plots"
    if not os.path.exists(outputPlotDirectory):
        os.mkdir(outputPlotDirectory)

    # Each simulation (at one angle)
    for filename in individual_data_filenames:

        sim_id = filename.removesuffix("_stats.nc")
        print(f"Processing {sim_id}...")
        
        # sim_id = filename.removeprefix("run_").removesuffix("_stats.nc")

        all_sim_angle_datapaths = [Path(fp) / filename for fp in angle_folders]

        num_angles = len(all_sim_angle_datapaths)
        print(f"Composing spectrum from {num_angles} B0 angles...")

        # Create single NC file for summary of combined spectra
        # Get existing spectra as xarray
        data_xr = xr.open_datatree(all_sim_angle_datapaths[0], engine="netcdf4")
        # New data without growth rates as they no longer make sense
        # Energy may make sense but needs further thought
        newData_xr = xr.DataTree(dataset=data_xr.to_dataset(inherit=False), children={
            "Energy" : data_xr.children["Energy"],
            "Magnetic_Field_Bz" : xr.DataTree(dataset=data_xr.children["Magnetic_Field_Bz"].to_dataset(), children={
                "power" : data_xr.children["Magnetic_Field_Bz"].children["power"]
            }),
            "Electric_Field_Ex" : xr.DataTree(dataset=data_xr.children["Electric_Field_Ex"].to_dataset(), children={
                "power" : data_xr.children["Electric_Field_Ex"].children["power"]
            }),
            "Electric_Field_Ey" : xr.DataTree(dataset=data_xr.children["Electric_Field_Ey"].to_dataset(), children={
                "power" : data_xr.children["Electric_Field_Ey"].children["power"]
            }),
        }, name = "combined_simulation_data")
        data_xr.close()
        newData_xr.attrs.pop("B0angle") # Now multivalued
        
        # Remove frequencies of max power as they're no longer correct
        # Broken, figure out how to do this
        # newData_xr["Magnetic_Field_Bz/power"].drop_vars("frequencyOfMaxPowerByK")
        # newData_xr["Electric_Field_Ex/power"].drop_vars("frequencyOfMaxPowerByK")
        # newData_xr["Electric_Field_Ey/power"].drop_vars("frequencyOfMaxPowerByK")

        # Set spectral variables to 0
        newData_xr["Magnetic_Field_Bz/power/powerByFrequency"] *= 0.0
        newData_xr["Magnetic_Field_Bz/power/powerByWavenumber"] *= 0.0
        newData_xr["Electric_Field_Ex/power/powerByFrequency"] *= 0.0
        newData_xr["Electric_Field_Ex/power/powerByWavenumber"] *= 0.0
        newData_xr["Electric_Field_Ey/power/powerByFrequency"] *= 0.0
        newData_xr["Electric_Field_Ey/power/powerByWavenumber"] *= 0.0
        newData_xr["Energy/backgroundIonMeanEnergyDensity"] *= 0.0
        newData_xr["Energy/electronMeanEnergyDensity"] *= 0.0
        newData_xr["Energy/electricFieldMeanEnergyDensity"] *= 0.0
        newData_xr["Energy/magneticFieldMeanEnergyDensity"] *= 0.0
        newData_xr["Energy/fastIonMeanEnergyDensity"] *= 0.0
        
        simulation_total_original_power = 0.0

        fig, (axBz, axEx, axEy) = plt.subplots(3, sharex=True, figsize=(16, 16))
        fig.suptitle(filename)
        axBz.set_title("Magnetic_Field_Bz")
        axBz.set_ylabel(r"Spectral power [$T \cdot \omega_{c, \alpha}$]")
        axEx.set_title("Electric_Field_Ex")
        axEx.set_ylabel(r"Spectral power [$\frac{V}{m} \cdot \omega_{c, \alpha}$]")
        axEy.set_title("Electric_Field_Ey")
        axEy.set_ylabel(r"Spectral power [$\frac{V}{m} \cdot \omega_{c, \alpha}$]")
        axEy.set_xlabel(r"Frequency/$\omega_{c, \alpha}$")
        for angle_sim_path in all_sim_angle_datapaths:

            print(f"Processing {sim_id} -- {angle_sim_path.absolute()}...")

            # Get this angle simulation's data
            angle_data_xr = xr.open_datatree(angle_sim_path, engine="netcdf4")

            # Ensure key parameters are equal to those already in the new dataset
            assert angle_data_xr.attrs["B0strength"] == newData_xr.attrs["B0strength"]
            assert angle_data_xr.attrs["backgroundDensity"] == newData_xr.attrs["backgroundDensity"]
            assert angle_data_xr.attrs["beamFraction"] == newData_xr.attrs["beamFraction"]
            assert angle_data_xr.attrs["pitch"] == newData_xr.attrs["pitch"]

            # angle_data_xr.to_netcdf("testToBeAdded.nc")

            # Check coordinates are equal
            assert (angle_data_xr["Magnetic_Field_Bz/power"].coords["wavenumber"] == newData_xr["Magnetic_Field_Bz/power"].coords["wavenumber"]).all()
            assert (angle_data_xr["Magnetic_Field_Bz/power"].coords["frequency"] == newData_xr["Magnetic_Field_Bz/power"].coords["frequency"]).all()
            assert (angle_data_xr["Electric_Field_Ex/power"].coords["wavenumber"] == newData_xr["Electric_Field_Ex/power"].coords["wavenumber"]).all()
            assert (angle_data_xr["Electric_Field_Ex/power"].coords["frequency"] == newData_xr["Electric_Field_Ex/power"].coords["frequency"]).all()
            assert (angle_data_xr["Electric_Field_Ey/power"].coords["wavenumber"] == newData_xr["Electric_Field_Ey/power"].coords["wavenumber"]).all()
            assert (angle_data_xr["Electric_Field_Ey/power"].coords["frequency"] == newData_xr["Electric_Field_Ey/power"].coords["frequency"]).all()
            assert (angle_data_xr["Energy/time"] == newData_xr["Energy/time"]).all()

            # Add spectra
            # Electric and magnetic fields
            newData_xr["Magnetic_Field_Bz/power/powerByWavenumber"] += angle_data_xr["Magnetic_Field_Bz/power/powerByWavenumber"]
            newData_xr["Magnetic_Field_Bz/power/powerByFrequency"] += angle_data_xr["Magnetic_Field_Bz/power/powerByFrequency"]
            newData_xr["Electric_Field_Ex/power/powerByWavenumber"] += angle_data_xr["Electric_Field_Ex/power/powerByWavenumber"]
            newData_xr["Electric_Field_Ex/power/powerByFrequency"] += angle_data_xr["Electric_Field_Ex/power/powerByFrequency"]
            newData_xr["Electric_Field_Ey/power/powerByWavenumber"] += angle_data_xr["Electric_Field_Ey/power/powerByWavenumber"]
            newData_xr["Electric_Field_Ey/power/powerByFrequency"] += angle_data_xr["Electric_Field_Ey/power/powerByFrequency"]

            # Plots
            axBz.plot(
                angle_data_xr["Magnetic_Field_Bz/power"].coords["frequency"].data, 
                angle_data_xr["Magnetic_Field_Bz/power/powerByFrequency"].data, 
                label = f"{angle_data_xr.attrs['B0angle']} degrees"
            )
            axEx.plot(
                angle_data_xr["Electric_Field_Ex/power"].coords["frequency"].data, 
                angle_data_xr["Electric_Field_Ex/power/powerByFrequency"].data, 
                label = f"{angle_data_xr.attrs['B0angle']} degrees"
            )
            axEy.plot(
                angle_data_xr["Electric_Field_Ey/power"].coords["frequency"].data, 
                angle_data_xr["Electric_Field_Ey/power/powerByFrequency"].data, 
                label = f"{angle_data_xr.attrs['B0angle']} degrees"
            )

            # Energy
            newData_xr["Energy/backgroundIonMeanEnergyDensity"] += angle_data_xr["Energy/backgroundIonMeanEnergyDensity"]
            newData_xr["Energy/electronMeanEnergyDensity"] += angle_data_xr["Energy/electronMeanEnergyDensity"]
            newData_xr["Energy/electricFieldMeanEnergyDensity"] += angle_data_xr["Energy/electricFieldMeanEnergyDensity"]
            newData_xr["Energy/magneticFieldMeanEnergyDensity"] += angle_data_xr["Energy/magneticFieldMeanEnergyDensity"]
            newData_xr["Energy/fastIonMeanEnergyDensity"] += angle_data_xr["Energy/fastIonMeanEnergyDensity"]

            simulation_total_original_power += angle_data_xr["Magnetic_Field_Bz/power/powerByWavenumber"].sum()
            simulation_total_original_power += angle_data_xr["Magnetic_Field_Bz/power/powerByFrequency"].sum()
            simulation_total_original_power += angle_data_xr["Electric_Field_Ex/power/powerByWavenumber"].sum()
            simulation_total_original_power += angle_data_xr["Electric_Field_Ex/power/powerByFrequency"].sum()
            simulation_total_original_power += angle_data_xr["Electric_Field_Ey/power/powerByWavenumber"].sum()
            simulation_total_original_power += angle_data_xr["Electric_Field_Ey/power/powerByFrequency"].sum()
            simulation_total_original_power += angle_data_xr["Energy/backgroundIonMeanEnergyDensity"].sum()
            simulation_total_original_power += angle_data_xr["Energy/electronMeanEnergyDensity"].sum()
            simulation_total_original_power += angle_data_xr["Energy/electricFieldMeanEnergyDensity"].sum()
            simulation_total_original_power += angle_data_xr["Energy/magneticFieldMeanEnergyDensity"].sum()
            simulation_total_original_power += angle_data_xr["Energy/fastIonMeanEnergyDensity"].sum()

        # Power check
        simulation_total_new_power = newData_xr["Magnetic_Field_Bz/power/powerByWavenumber"].sum()
        simulation_total_new_power += newData_xr["Magnetic_Field_Bz/power/powerByFrequency"].sum()
        simulation_total_new_power += newData_xr["Electric_Field_Ex/power/powerByWavenumber"].sum()
        simulation_total_new_power += newData_xr["Electric_Field_Ex/power/powerByFrequency"].sum()
        simulation_total_new_power += newData_xr["Electric_Field_Ey/power/powerByWavenumber"].sum()
        simulation_total_new_power += newData_xr["Electric_Field_Ey/power/powerByFrequency"].sum()
        simulation_total_new_power += newData_xr["Energy/backgroundIonMeanEnergyDensity"].sum()
        simulation_total_new_power += newData_xr["Energy/electronMeanEnergyDensity"].sum()
        simulation_total_new_power += newData_xr["Energy/electricFieldMeanEnergyDensity"].sum()
        simulation_total_new_power += newData_xr["Energy/magneticFieldMeanEnergyDensity"].sum()
        simulation_total_new_power += newData_xr["Energy/fastIonMeanEnergyDensity"].sum()
        assert np.allclose(simulation_total_original_power.data, simulation_total_new_power.data)

        # Make energy an average
        newData_xr["Energy/backgroundIonMeanEnergyDensity"] /= num_angles
        newData_xr["Energy/electronMeanEnergyDensity"] /= num_angles
        newData_xr["Energy/electricFieldMeanEnergyDensity"] /= num_angles
        newData_xr["Energy/magneticFieldMeanEnergyDensity"] /= num_angles
        newData_xr["Energy/fastIonMeanEnergyDensity"] /= num_angles

        # Plot
        axBz.plot(
            newData_xr["Magnetic_Field_Bz/power"].coords["frequency"].data, 
            newData_xr["Magnetic_Field_Bz/power/powerByFrequency"].data, 
            label = "Combined"
        )
        axEx.plot(
            newData_xr["Electric_Field_Ex/power"].coords["frequency"].data, 
            newData_xr["Electric_Field_Ex/power/powerByFrequency"].data, 
            label = "Combined"
        )
        axEy.plot(
            newData_xr["Electric_Field_Ey/power"].coords["frequency"].data, 
            newData_xr["Electric_Field_Ey/power/powerByFrequency"].data, 
            label = "Combined"
        )
        axBz.set_ylim(bottom=0)
        axEx.set_ylim(bottom=0)
        axEy.set_ylim(bottom=0)
        handles, labels = axBz.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        axBz.legend(handles, labels)
        axBz.grid()
        axEx.grid()
        axEy.grid()
        plt.savefig(outputPlotDirectory / f"{sim_id}_combined_spectra.png")

        # Fix energy stats
        for k, _ in newData_xr["Energy"].attrs.items():
            newData_xr["Energy"].attrs[k] = 0.0
        newData_xr["Energy"].attrs["fastIonEnergyDensity_delta"] = newData_xr["Energy/fastIonMeanEnergyDensity"][-1].data - newData_xr["Energy/fastIonMeanEnergyDensity"][0].data
        newData_xr["Energy"].attrs["backgroundIonEnergyDensity_delta"] = newData_xr["Energy/backgroundIonMeanEnergyDensity"][-1].data - newData_xr["Energy/backgroundIonMeanEnergyDensity"][0].data
        newData_xr["Energy"].attrs["electronEnergyDensity_delta"] = newData_xr["Energy/electronMeanEnergyDensity"][-1].data - newData_xr["Energy/electronMeanEnergyDensity"][0].data
        newData_xr["Energy"].attrs["electricFieldEnergyDensity_delta"] = newData_xr["Energy/electricFieldMeanEnergyDensity"][-1].data - newData_xr["Energy/electricFieldMeanEnergyDensity"][0].data
        newData_xr["Energy"].attrs["magneticFieldEnergyDensity_delta"] = newData_xr["Energy/magneticFieldMeanEnergyDensity"][-1].data - newData_xr["Energy/magneticFieldMeanEnergyDensity"][0].data
        newData_xr["Energy"].attrs["totalEnergyDensity_delta"] = newData_xr["Energy"].attrs["fastIonEnergyDensity_delta"] + newData_xr["Energy"].attrs["backgroundIonEnergyDensity_delta"] + newData_xr["Energy"].attrs["electronEnergyDensity_delta"] + newData_xr["Energy"].attrs["electricFieldEnergyDensity_delta"] + newData_xr["Energy"].attrs["magneticFieldEnergyDensity_delta"]
        newData_xr["Energy"].attrs["fastIonToBackgroundTransfer"] = newData_xr["Energy"].attrs["backgroundIonEnergyDensity_delta"] - newData_xr["Energy"].attrs["fastIonEnergyDensity_delta"]
        newData_xr["Energy"].attrs["fastIonToBackgroundTransfer_percentage"] = 100.0 * (newData_xr["Energy"].attrs["fastIonToBackgroundTransfer"] / newData_xr["Energy/fastIonMeanEnergyDensity"][0].data)

        # Write new file
        outFname = outputDataDirectory / f"{sim_id}_combined_stats.nc"
        newData_xr.to_netcdf(outFname)
        newData_xr.close()

        print(f"{filename} complete and written to {outFname.absolute()}.")

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
        help="Directory containing analysis of all simulations. Expects folders of angles, each with data and plots folders.",
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
        "--combineSpectra",
        action="store_true",
        help="Combine Ex, Ey and Bz spectra.",
        required = False
    )
    parser.add_argument(
        "--calculateIciness",
        action="store_true",
        help="Calculate ICEiness characteristics for combined spectra.",
        required = False
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
        "--iceMetrics",
        action="store",
        help="Ice metrics to plot.",
        required = False,
        type=str,
        nargs="*"
    )
    
    args = parser.parse_args()

    if args.combineSpectra:
        combine_spectra(args.dir, args.outputDir)
        plt.close("all")
    
    combined_dir = args.outputDir

    if args.calculateIciness:
        calculate_iciness(combined_dir, args.fields)

    if args.plotIciness:
        plot_iciness(combined_dir, args.fields, args.iceMetrics)

    if args.recoverB0:
        estimate_B0_from_spectra(combined_dir, args.fields)

