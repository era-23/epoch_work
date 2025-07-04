import argparse
import glob
import os
import copy
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import xarray as xr
import shutil as sh
from pathlib import Path

def correlate_growth_rate_with_frequency(
        dir : Path, 
        outputDir : Path, 
        fields : list,
        rSquaredThreshold : float = 0.95,
        rawRSquaredThreshold : float = 0.5,
        stdErrThreshold : float = 0.002,
        rawStdErrThreshold : float = 0.003,
        k_transition : float = 20.0,
        displayPlots : bool = False,
    ):
    
    data_files = glob.glob(str(dir / "*.nc"))

    # Create folder structure
    # Use hybrid frequency calibration curve if possible
    if "Magnetic_Field_Bz" in fields and "Electric_Field_Ex" in fields:
        if os.path.exists(outputDir / "growth_rate_analysis"):
            sh.rmtree(outputDir / "growth_rate_analysis")
        os.makedirs(outputDir / "growth_rate_analysis" / "all")
        os.mkdir(outputDir / "growth_rate_analysis" / "all_coloured")
        os.mkdir(outputDir / "growth_rate_analysis" / "rSquaredFilter")
        os.mkdir(outputDir / "growth_rate_analysis" / "rawRSquaredFilter")
        os.mkdir(outputDir / "growth_rate_analysis" / "stdErrFilter")
        os.mkdir(outputDir / "growth_rate_analysis" / "rawStdErrFilter")
    else:
        for field in fields:
            if os.path.exists(outputDir / field / "growth_rate_analysis"):
                sh.rmtree(outputDir / field / "growth_rate_analysis")
            os.makedirs(outputDir / field / "growth_rate_analysis" / "all")
            os.mkdir(outputDir / field / "growth_rate_analysis" / "all_coloured")
            os.mkdir(outputDir / field / "growth_rate_analysis" / "rSquaredFilter")
            os.mkdir(outputDir / field / "growth_rate_analysis" / "rawRSquaredFilter")
            os.mkdir(outputDir / field / "growth_rate_analysis" / "stdErrFilter")
            os.mkdir(outputDir / field / "growth_rate_analysis" / "rawStdErrFilter")

    for simulation in data_files:

        data = xr.open_datatree(
            simulation,
            engine="netcdf4"
        )

        sim_id = Path(simulation).name.split("/")[-1].split(".")[0]

        print(f"Processing {sim_id}....")

        # Create plots
        # Use hybrid frequency calibration curve if possible
        if "Magnetic_Field_Bz" in fields and "Electric_Field_Ex" in fields:

            saveDir = Path(outputDir / "growth_rate_analysis")

            wavenumbers = data["/Electric_Field_Ex/growthRates/positive/wavenumber"]
            max_freq = data["/Magnetic_Field_Bz/power/frequency"].max()

            # Frequencies
            frequencies = clean_frequencies(data, k_transition)
            growthRates = combine_lowKBz_highKEx(
                data["/Electric_Field_Ex/growthRates/positive/growthRate"].to_dataset().to_dataarray(),
                data["/Magnetic_Field_Bz/growthRates/positive/growthRate"].to_dataset().to_dataarray(),
                k_transition)
            rSquareds = combine_lowKBz_highKEx(
                data["/Electric_Field_Ex/growthRates/positive/rSquared"].to_dataset().to_dataarray(),
                data["/Magnetic_Field_Bz/growthRates/positive/rSquared"].to_dataset().to_dataarray(),
                k_transition)
            rawRSquareds = combine_lowKBz_highKEx(
                data["/Electric_Field_Ex/growthRates/positive/rawRSquared"].to_dataset().to_dataarray(),
                data["/Magnetic_Field_Bz/growthRates/positive/rawRSquared"].to_dataset().to_dataarray(),
                k_transition)
            stdErrs = combine_lowKBz_highKEx(
                data["/Electric_Field_Ex/growthRates/positive/stdErr"].to_dataset().to_dataarray(),
                data["/Magnetic_Field_Bz/growthRates/positive/stdErr"].to_dataset().to_dataarray(),
                k_transition)
            rawStdErrs = combine_lowKBz_highKEx(
                data["/Electric_Field_Ex/growthRates/positive/rawStdErr"].to_dataset().to_dataarray(),
                data["/Magnetic_Field_Bz/growthRates/positive/rawStdErr"].to_dataset().to_dataarray(),
                k_transition)
            
            frequencies = xr.DataArray(frequencies, dims=["wavenumber"], coords=[wavenumbers])
            growthRates = xr.DataArray(growthRates, dims=["wavenumber"], coords=[wavenumbers])
            
            # All data
            dataDir = saveDir / "all"
            plt.scatter(frequencies.data, growthRates.data, marker='x', color = "blue")
            plt.title(f"{sim_id}: Best fit growth rate by frequency\n(all data)")
            plt.xlabel(r"Frequency/$\omega_{cp}$")
            plt.ylabel(r"Growth rate/$\omega_{cp}$")
            plt.xlim((-1.0, max_freq + 1.0))
            plt.tight_layout()
            plt.savefig(dataDir / f"{sim_id}_gamma_by_omega_all.png")
            if displayPlots:
                plt.show()
            plt.close("all")

            # All data, coloured
            dataDir = saveDir / "all_coloured"
            # rSquaredVals = (10**np.nan_to_num(rawRSquareds.to_numpy()).clip(0.0, 1.0))/10.0
            rSquaredVals = np.nan_to_num(rawRSquareds)
            alpha_vals = rSquaredVals.clip(0.0, 1.0)
            r, g, b = to_rgb("blue")
            colours = [(r, g, b, alpha) for alpha in alpha_vals]
            plt.scatter(frequencies.data, growthRates.data, marker='x', color = colours)
            plt.title(f"{sim_id}: Best fit growth rate by frequency\n(all data -- alpha scaled to raw R^2)")
            plt.xlabel(r"Frequency/$\omega_{cp}$")
            plt.ylabel(r"Growth rate/$\omega_{cp}$")
            plt.xlim((-1.0, max_freq + 1.0))
            plt.tight_layout()
            plt.savefig(dataDir / f"{sim_id}_gamma_by_omega_all_coloured.png")
            if displayPlots:
                plt.show()
            plt.close("all")

            # Filter by r-squared
            dataDir = saveDir / "rSquaredFilter"
            elements = np.where(rSquareds > rSquaredThreshold)[0]
            filtered_freqs = frequencies.isel(wavenumber=elements)
            filtered_gammas = growthRates.isel(wavenumber=elements)
            plt.scatter(filtered_freqs.data, filtered_gammas.data, marker='x', color="blue")
            plt.title(f"{sim_id}: Best fit growth rate by frequency\n(data filtered by r-squared > {rSquaredThreshold})")
            plt.xlabel(r"Frequency/$\omega_{cp}$")
            plt.ylabel(r"Growth rate/$\omega_{cp}$")
            plt.xlim((-1.0, max_freq + 1.0))
            plt.tight_layout()
            plt.savefig(dataDir / f"{sim_id}_gamma_by_omega_rSqrdFilter.png")
            if displayPlots:
                plt.show()
            plt.close("all")

            # Filter by raw r-squared
            dataDir = saveDir / "rawRSquaredFilter"
            elements = np.where(rawRSquareds > rawRSquaredThreshold)[0]
            filtered_freqs = frequencies.isel(wavenumber=elements)
            filtered_gammas = growthRates.isel(wavenumber=elements)
            plt.scatter(filtered_freqs.data, filtered_gammas.data, marker='x', color="blue")
            plt.title(f"{sim_id}: Best fit growth rate by frequency\n(data filtered by raw r-squared > {rawRSquaredThreshold})")
            plt.xlabel(r"Frequency/$\omega_{cp}$")
            plt.ylabel(r"Growth rate/$\omega_{cp}$")
            plt.xlim((-1.0, max_freq + 1.0))
            plt.tight_layout()
            plt.savefig(dataDir / f"{sim_id}_gamma_by_omega_rawRSqrdFilter.png")
            if displayPlots:
                plt.show()
            plt.close("all")

            # Filter by stdErr
            dataDir = saveDir / "stdErrFilter"
            elements = np.where(stdErrs < stdErrThreshold)[0]
            filtered_freqs = frequencies.isel(wavenumber=elements)
            filtered_gammas = growthRates.isel(wavenumber=elements)
            plt.scatter(filtered_freqs.data, filtered_gammas.data, marker='x', color="blue")
            plt.title(f"{sim_id}: Best fit growth rate by frequency\n(data filtered by std error < {stdErrThreshold})")
            plt.xlabel(r"Frequency/$\omega_{cp}$")
            plt.ylabel(r"Growth rate/$\omega_{cp}$")
            plt.xlim((-1.0, max_freq + 1.0))
            plt.tight_layout()
            plt.savefig(dataDir / f"{sim_id}_gamma_by_omega_stdErrFilter.png")
            if displayPlots:
                plt.show()
            plt.close("all")

            # Filter by raw stdErr
            dataDir = saveDir / "rawStdErrFilter"
            elements = np.where(rawStdErrs < rawStdErrThreshold)[0]
            filtered_freqs = frequencies.isel(wavenumber=elements)
            filtered_gammas = growthRates.isel(wavenumber=elements)
            plt.scatter(filtered_freqs.data, filtered_gammas.data, marker='x', color="blue")
            plt.title(f"{sim_id}: Best fit growth rate by frequency\n(data filtered by raw std error < {rawStdErrThreshold})")
            plt.xlabel(r"Frequency/$\omega_{cp}$")
            plt.ylabel(r"Growth rate/$\omega_{cp}$")
            plt.xlim((-1.0, max_freq + 1.0))
            plt.tight_layout()
            plt.savefig(dataDir / f"{sim_id}_gamma_by_omega_rawStdErrFilter.png")
            if displayPlots:
                plt.show()
            plt.close("all")
        
        else:
            for field in fields:

                saveDir = Path(outputDir / field / "growth_rate_analysis")

                gammaData = data["/" + field + "/growthRates/positive"].to_dataset()

                frequencies = xr.DataArray(gammaData.variables["frequencyOfMaxPowerInK"], dims=["wavenumber"], coords=[gammaData.variables["wavenumber"]])
                growthRates = xr.DataArray(gammaData.variables["growthRate"], dims=["wavenumber"], coords=[gammaData.variables["wavenumber"]])
                rSquareds = xr.DataArray(gammaData.variables["rSquared"], dims=["wavenumber"], coords=[gammaData.variables["wavenumber"]])
                rawRSquareds = xr.DataArray(gammaData.variables["rawRSquared"], dims=["wavenumber"], coords=[gammaData.variables["wavenumber"]])
                stdErrs = xr.DataArray(gammaData.variables["stdErr"], dims=["wavenumber"], coords=[gammaData.variables["wavenumber"]])
                rawStdErrs = xr.DataArray(gammaData.variables["rawStdErr"], dims=["wavenumber"], coords=[gammaData.variables["wavenumber"]])

                # All data
                dataDir = saveDir / "all"
                plt.scatter(frequencies.data, growthRates.data, marker='x', color = "blue")
                plt.title(f"{sim_id} {field}: Best fit growth rate by frequency\n(all data)")
                plt.xlabel(r"Frequency/$\omega_{cp}$")
                plt.ylabel(r"Growth rate/$\omega_{cp}$")
                plt.xlim((-1.0, max_freq + 1.0))
                plt.tight_layout()
                plt.savefig(dataDir / f"{sim_id}_{field}_gamma_by_omega_all.png")
                if displayPlots:
                    plt.show()
                plt.close("all")

                # All data, coloured
                dataDir = saveDir / "all_coloured"
                # rSquaredVals = (10**np.nan_to_num(rawRSquareds.to_numpy()).clip(0.0, 1.0))/10.0
                rSquaredVals = np.nan_to_num(rawRSquareds.to_numpy())
                alpha_vals = rSquaredVals.clip(0.0, 1.0)
                r, g, b = to_rgb("blue")
                colours = [(r, g, b, alpha) for alpha in alpha_vals]
                plt.scatter(frequencies.data, growthRates.data, marker='x', color = colours)
                plt.title(f"{sim_id} {field}: Best fit growth rate by frequency\n(all data -- alpha scaled to raw R^2)")
                plt.xlabel(r"Frequency/$\omega_{cp}$")
                plt.ylabel(r"Growth rate/$\omega_{cp}$")
                plt.xlim((-1.0, max_freq + 1.0))
                plt.tight_layout()
                plt.savefig(dataDir / f"{sim_id}_{field}_gamma_by_omega_all_coloured.png")
                if displayPlots:
                    plt.show()
                plt.close("all")

                # Filter by r-squared
                dataDir = saveDir / "rSquaredFilter"
                elements = np.where(rSquareds.data > rSquaredThreshold)[0]
                filtered_freqs = frequencies.isel(wavenumber=elements)
                filtered_gammas = growthRates.isel(wavenumber=elements)
                plt.scatter(filtered_freqs.data, filtered_gammas.data, marker='x', color="blue")
                plt.title(f"{sim_id} {field}: Best fit growth rate by frequency\n(data filtered by r-squared > {rSquaredThreshold})")
                plt.xlabel(r"Frequency/$\omega_{cp}$")
                plt.ylabel(r"Growth rate/$\omega_{cp}$")
                plt.xlim((-1.0, max_freq + 1.0))
                plt.tight_layout()
                plt.savefig(dataDir / f"{sim_id}_{field}_gamma_by_omega_rSqrdFilter.png")
                if displayPlots:
                    plt.show()
                plt.close("all")

                # Filter by raw r-squared
                dataDir = saveDir / "rawRSquaredFilter"
                elements = np.where(rawRSquareds.data > rawRSquaredThreshold)[0]
                filtered_freqs = frequencies.isel(wavenumber=elements)
                filtered_gammas = growthRates.isel(wavenumber=elements)
                plt.scatter(filtered_freqs.data, filtered_gammas.data, marker='x', color="blue")
                plt.title(f"{sim_id} {field}: Best fit growth rate by frequency\n(data filtered by raw r-squared > {rawRSquaredThreshold})")
                plt.xlabel(r"Frequency/$\omega_{cp}$")
                plt.ylabel(r"Growth rate/$\omega_{cp}$")
                plt.xlim((-1.0, max_freq + 1.0))
                plt.tight_layout()
                plt.savefig(dataDir / f"{sim_id}_{field}_gamma_by_omega_rawRSqrdFilter.png")
                if displayPlots:
                    plt.show()
                plt.close("all")

                # Filter by stdErr
                dataDir = saveDir / "stdErrFilter"
                elements = np.where(stdErrs.data < stdErrThreshold)[0]
                filtered_freqs = frequencies.isel(wavenumber=elements)
                filtered_gammas = growthRates.isel(wavenumber=elements)
                plt.scatter(filtered_freqs.data, filtered_gammas.data, marker='x', color="blue")
                plt.title(f"{sim_id} {field}: Best fit growth rate by frequency\n(data filtered by std error < {stdErrThreshold})")
                plt.xlabel(r"Frequency/$\omega_{cp}$")
                plt.ylabel(r"Growth rate/$\omega_{cp}$")
                plt.xlim((-1.0, max_freq + 1.0))
                plt.tight_layout()
                plt.savefig(dataDir / f"{sim_id}_{field}_gamma_by_omega_stdErrFilter.png")
                if displayPlots:
                    plt.show()
                plt.close("all")

                # Filter by raw stdErr
                dataDir = saveDir / "rawStdErrFilter"
                elements = np.where(rawStdErrs.data < rawStdErrThreshold)[0]
                filtered_freqs = frequencies.isel(wavenumber=elements)
                filtered_gammas = growthRates.isel(wavenumber=elements)
                plt.scatter(filtered_freqs.data, filtered_gammas.data, marker='x', color="blue")
                plt.title(f"{sim_id} {field}: Best fit growth rate by frequency\n(data filtered by raw std error < {rawStdErrThreshold})")
                plt.xlabel(r"Frequency/$\omega_{cp}$")
                plt.ylabel(r"Growth rate/$\omega_{cp}$")
                plt.xlim((-1.0, max_freq + 1.0))
                plt.tight_layout()
                plt.savefig(dataDir / f"{sim_id}_{field}_gamma_by_omega_rawStdErrFilter.png")
                if displayPlots:
                    plt.show()
                plt.close("all")

def combine_lowKBz_highKEx(Ex_quantity : xr.DataArray, Bz_quantity : xr.DataArray, k_transition):
    ExLow = Ex_quantity.sel(wavenumber=slice(None,-k_transition)).to_numpy().flatten()
    BzMid = Bz_quantity.sel(wavenumber=slice(-k_transition,k_transition)).to_numpy().flatten()
    ExHigh = Ex_quantity.sel(wavenumber=slice(k_transition,None)).to_numpy().flatten()
    return np.concatenate((ExLow, BzMid, ExHigh))   

def remove_outliers(
    signal, 
    x_vals, 
    n_neighbours_factor = 0.02, 
    outlier_prominence_SDs = 1.0, 
    max_iter = 50,
    plot = False
):
    
    sig = copy.deepcopy(signal)
    clf = LocalOutlierFactor(n_neighbors=int(n_neighbours_factor*len(sig)), contamination='auto')
    sig_r = sig.reshape(-1, 1)

    total_removed = []

    for i in range(max_iter):
        
        _ = clf.fit_predict(sig_r)
        outlier_scores = -clf.negative_outlier_factor_/np.max(-clf.negative_outlier_factor_)
        scaled_outlier_scores = 40.0*outlier_scores
        mean_nof = np.mean(outlier_scores)
        sd_nof = np.std(outlier_scores)

        interpolating = False
        interp_indices = []
        x_indices = np.arange(stop=len(x_vals))
        num_removed = 0
        
        for s in range(len(outlier_scores)):
            score = outlier_scores[s]
            if score >= mean_nof + (outlier_prominence_SDs * sd_nof): # If value is more than 1 S.D. above the mean
                # Need to remove value
                interpolating = True
                interp_indices.append(s)
            else:
                # End of outliers
                if interpolating:
                    # Replace values
                    start_interp_at = interp_indices[0]-1
                    end_interp_at = interp_indices[-1]+1
                    # print(f"Replacing indices {interp_indices} (values {sig[interp_indices]}) with linear interpolation between indices {start_interp_at} ({sig[start_interp_at]}) and {end_interp_at} ({sig[end_interp_at]})")
                    newPoints = np.interp(x_indices[interp_indices], [x_indices[start_interp_at], x_indices[end_interp_at]], [sig[start_interp_at], sig[end_interp_at]])
                    sig[interp_indices] = newPoints
                    num_removed += len(interp_indices)
                    interp_indices = []
                    interpolating = False
        
        total_removed.append(num_removed)
        
        # print(f"Removed {num_removed} outliers")
        doneTest = len(total_removed) > 3 and (total_removed[-1] == total_removed[-2] == total_removed[-3] == total_removed[-4])
        if num_removed == 0 or doneTest:
            print(f"Done after {i+1} iterations.")
            break
        sig_r = sig.reshape(-1, 1)

        if i == max_iter-1:
            print(f"Done, i = {i+1} = max_iter.")

    bestFreqs = sig
    if plot:
        plt.plot(x_vals, signal, color = "purple", label = "Best")
        plt.plot(x_vals, bestFreqs, color = "green", label = "Best, with outliers removed and interpolated")
        plt.scatter(x_vals, scaled_outlier_scores, marker='x', color = 'orange', 
                    label=f"NOF: {mean_nof:.3f} SD {sd_nof:.3f}")
        plt.show()
    
    return bestFreqs

def clean_frequencies(
    data : xr.DataTree,
    k_transition = 20.0):

    BzData = data["/Magnetic_Field_Bz/growthRates/positive"].to_dataset()
    ExData = data["/Electric_Field_Ex/growthRates/positive"].to_dataset()

    betterFreqs = combine_lowKBz_highKEx(ExData.frequencyOfMaxPowerInK, BzData.frequencyOfMaxPowerInK, k_transition)       
    wavenumbers = np.array(ExData.frequencyOfMaxPowerInK.coords["wavenumber"])

    # Outlier detection
    bestFreqs = remove_outliers(betterFreqs, wavenumbers)
    return bestFreqs

def clean_frequency_beta(dir : Path):
    sims = [
        "run_4_stats.nc",
        "run_7_stats.nc",
        "run_10_stats.nc",
        "run_17_stats.nc", 
        "run_22_stats.nc", 
        "run_28_stats.nc", 
        "run_31_stats.nc", 
        "run_33_stats.nc",
        "run_40_stats.nc",
        "run_42_stats.nc",
        "run_51_stats.nc",
        "run_55_stats.nc",
        "run_63_stats.nc",
        "run_64_stats.nc",
        "run_69_stats.nc", 
        "run_73_stats.nc", 
        "run_87_stats.nc",
        "run_89_stats.nc", 
        "run_90_stats.nc", 
        "run_93_stats.nc", 
        "run_97_stats.nc",
        "run_98_stats.nc",
        "run_99_stats.nc",]

    for sim in sims:
        datafile = dir / sim

        data = xr.open_datatree(
            datafile,
            engine="netcdf4"
        )

        BzData = data["/Magnetic_Field_Bz/growthRates/positive"].to_dataset()
        ExData = data["/Electric_Field_Ex/growthRates/positive"].to_dataset()

        BzData.frequencyOfMaxPowerInK.plot(color = "red", label="Bz")
        ExData.frequencyOfMaxPowerInK.plot(color = "blue", label="Ex")

        k_cutoff = 20.0
        ExLow = ExData.frequencyOfMaxPowerInK.sel(wavenumber=slice(None,-k_cutoff)).to_numpy()
        BzMid = BzData.frequencyOfMaxPowerInK.sel(wavenumber=slice(-k_cutoff,k_cutoff)).to_numpy()
        ExHigh = ExData.frequencyOfMaxPowerInK.sel(wavenumber=slice(k_cutoff,None)).to_numpy()
        betterFreqs = np.concatenate((ExLow, BzMid, ExHigh))           
        wavenumbers = np.array(ExData.frequencyOfMaxPowerInK.coords["wavenumber"])

        plt.plot(wavenumbers, betterFreqs, color = "purple", label = "Hybrid Bz, Ex")
        plt.plot()

        # Outlier detection
        # bestFreqs = remove_outliers(betterFreqs, wavenumbers)
        # plt.plot(wavenumbers, betterFreqs, color = "purple", label = "Hybrid Bz, Ex")
        # plt.plot(wavenumbers, bestFreqs, color = "orange", label = "Outliers removed")
        plt.legend()
        plt.title(sim)
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--dir",
        action="store",
        help="Directory containing simulation data analysis (netCDF) files for evaluation.",
        required = True,
        type=Path
    )
    parser.add_argument(
        "--outputDir",
        action="store",
        help="Directory in which to save plots.",
        required = True,
        type=Path
    )
    parser.add_argument(
        "--fields",
        action="store",
        help="EPOCH output fields to use for growth rate analysis, e.g. Magnetic_Field_Bz, Electric_Field_Ex.",
        required = True,
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--rSquaredThreshold",
        action="store",
        help="R squared threshold for filtering data.",
        required = False,
        type=float
    )
    parser.add_argument(
        "--rawRSquaredThreshold",
        action="store",
        help="R squared threshold of raw (unsmoothed) data for filtering.",
        required = False,
        type=float
    )
    parser.add_argument(
        "--stdErrThreshold",
        action="store",
        help="Standard error threshold for filtering data.",
        required = False,
        type=float
    )
    parser.add_argument(
        "--rawStdErrThreshold",
        action="store",
        help="Standard error threshold for raw (unsmoothed) data for filtering.",
        required = False,
        type=float
    )
    parser.add_argument(
        "--kTransition",
        action="store",
        help="Transition point in wavenumber between high and low k regions, where the Alfven branch in Ex or Bz is respectively better resolved.",
        required = False,
        type=float
    )
    parser.add_argument(
        "--displayPlots",
        action="store_true",
        help="Display plots.",
        required = False
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug.",
        required = False
    )

    args = parser.parse_args()

    # clean_frequency_beta(args.dir)
    correlate_growth_rate_with_frequency(args.dir, args.outputDir, args.fields, args.rSquaredThreshold, args.rawRSquaredThreshold, args.stdErrThreshold, args.rawStdErrThreshold, args.kTransition, args.displayPlots)