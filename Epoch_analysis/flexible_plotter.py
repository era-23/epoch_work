import argparse
import glob
from pathlib import Path
import epydeck
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sdf_xarray import SDFPreprocess
import numpy as np
import xarray as xr
import xrft
import epoch_utils
import ml_utils
import plasmapy.formulary.frequencies as ppf
import plasmapy.formulary.speeds as pps
import astropy.units as u
import netCDF4 as nc
from scipy.interpolate import make_smoothing_spline

class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):  # Override function that finds format to use.
        self.format = "%.1f"  # Give format here

def all_power_spectra(dataFolder : Path, fields : list):

    # Get all datafiles
    data_files = glob.glob(str(dataFolder / "*.nc"))

    allSpectra = []

    allSpectra = ml_utils.read_data(
        data_files, 
        {f : [] for f in fields},
        with_names = True, 
        with_coords=True
    )
    
    b0 = np.array(ml_utils.read_data(
        data_files, 
        {"B0strength" : []},
        with_names = True
    )["B0strength"])
    
    # Check all corrdinates are aligned
    # coords = [v for k,v in allSpectra.items() if k.endswith("_coords")]
    # for i in range(len(coords)-1):
    #     assert np.allclose(coords[i], coords[i+1])

    for field in fields:
        
        # Raw data
        spectra = np.array([np.array(s) for s in allSpectra[field]])
        print(spectra.shape)

        # For each simulation, take log10 of power^2 / B0^2
        # for i in range(spectra.shape[0]):
        #     spectra[i] = np.log10(spectra[i]**2 / b0[i]**2)
        
        plt.subplots(figsize=(12,8))
        heatmap = plt.imshow(spectra, cmap="plasma", aspect='auto', origin='lower')
        # heatmap = plt.imshow(np.log10(spectra), cmap="plasma", aspect='auto', origin='lower')
        plt.title(f"{epoch_utils.fieldNameToText(field)}")
        plt.xlim(allSpectra[field + "_coords"][0][0], allSpectra[field + "_coords"][0][-1])
        plt.ylim(float(allSpectra["sim_ids"][0]), float(allSpectra["sim_ids"][-1]))
        plt.xlabel(r"Frequency/$\Omega_\alpha$")
        plt.ylabel("Simulation ID")
        cbar = plt.colorbar(heatmap)
        cbar.ax.set_ylabel("power/T")
        plt.tight_layout()
        plt.show()
        
def power_spectrum(dataFile : Path, field : str, doPlot : bool = True):
    
    data = xr.open_datatree(
            dataFile,
            engine="netcdf4"
    )

    B0 = float(data.attrs["B0strength"])
    s = field.split("/")
    group = "/" if len(s) == 1 else '/'.join(s[:-1])
    fieldName = s[-1]
    power_trace = data[group].variables[fieldName]
    fieldData = power_trace.data.astype("float")
    coords = data[group].coords[power_trace.dims[0]]
    fieldName = epoch_utils.fieldNameToText(field)
    
    if doPlot:
        fig, axs = plt.subplots(figsize=(15, 10))
        axs.plot(coords, fieldData)
        axs.set_xticks(ticks=np.arange(np.floor(coords[0]), np.ceil(coords[-1])+1.0, 1.0), minor=True)
        axs.grid(which='both', axis='x')
        axs.set_xlabel(r"Frequency [$\Omega_{ci}$]")
        axs.set_ylabel(f"Sum of power in {fieldName} over all k [T]")
        plt.show()

        fig, axs = plt.subplots(figsize=(15, 10))
        axs.plot(coords, 10.0 * np.log10(fieldData / B0))
        axs.set_xticks(ticks=np.arange(np.floor(coords[0]), np.ceil(coords[-1])+1.0, 1.0), minor=True)
        axs.grid(which='both', axis='x')
        axs.set_xlabel(r"Frequency [$\Omega_{ci}$]")
        axs.set_ylabel(f"Sum of power in {fieldName} over all k [dB]")
        plt.show()

    return fieldData, coords

def compare_spectra(folder : Path, simNumbers : list, maxXcoord : float = 50.0, axesToEquate = list):
    
    # Limit to two simulations for now
    assert len(simNumbers) == 2

    # Finding folder with all angle data...
    allAngle_folders = set(glob.glob(str(folder / "*angles*/")) + glob.glob(str(folder.parent / "*angles*/")))
    assert len(allAngle_folders) == 1
    # Get angle subfolders
    # Get data folders for each angle
    angle_folders = glob.glob(str(Path(allAngle_folders.pop()) / "9[0-9]/data/"))
    assert len(angle_folders) > 0

    # Finding data folder with combined spectra...
    combinedSpectra_folder = set(glob.glob(str(folder / "*combined_spectra*/data/")))
    assert len(combinedSpectra_folder) == 1
    combinedSpectra_folder = Path(combinedSpectra_folder.pop())

    fig, ((axBz1, axBz2), (axEx1, axEx2), (axEy1, axEy2)) = plt.subplots(3, 2, sharex=True, figsize=(16, 20))
    # fig.suptitle(f"{simNumbers[0]} vs {simNumbers[1]}")
    
    f = ScalarFormatterForceFormat(useMathText=True, useOffset=False)
    f.set_scientific(True)
    axBz1.set_title(f"Run {simNumbers[0]} spectral power: {r'$B_z$'}")
    axBz1.set_ylabel(r"$T \cdot \Omega_{c, \alpha}$")
    axBz1.yaxis.set_major_formatter(f)
    f = ScalarFormatterForceFormat(useMathText=True, useOffset=False)
    f.set_scientific(True)
    axBz2.set_title(f"Run {simNumbers[1]} spectral power: {r'$B_z$'}")
    axBz2.yaxis.set_major_formatter(f)
    
    f = ScalarFormatterForceFormat(useMathText=True, useOffset=False)
    f.set_scientific(True)
    axEx1.set_title(f"Run {simNumbers[0]} spectral power: {r'$E_x$'}")
    axEx1.set_ylabel(r"$\frac{V}{m} \cdot \Omega_{c, \alpha}$")
    axEx1.yaxis.set_major_formatter(f)
    f = ScalarFormatterForceFormat(useMathText=True, useOffset=False)
    f.set_scientific(True)
    axEx2.set_title(f"Run {simNumbers[1]} spectral power: {r'$E_x$'}")
    axEx2.yaxis.set_major_formatter(f)
    
    f = ScalarFormatterForceFormat(useMathText=True, useOffset=False)
    f.set_scientific(True)
    axEy1.set_title(f"Run {simNumbers[0]} spectral power: {r'$E_y$'}")
    axEy1.set_ylabel(r"$\frac{V}{m} \cdot \Omega_{c, \alpha}$")
    axEy1.yaxis.set_major_formatter(f)
    f = ScalarFormatterForceFormat(useMathText=True, useOffset=False)
    f.set_scientific(True)
    axEy1.set_xlabel(r"Frequency [$\Omega_{c, \alpha}$]")
    axEy2.set_title(f"Run {simNumbers[1]} spectral power: {r'$E_y$'}")
    axEy2.yaxis.set_major_formatter(f)
    axEy2.set_xlabel(r"Frequency [$\Omega_{c, \alpha}$]")

    axes = {simNumbers[0] : (axBz1, axEx1, axEy1), simNumbers[1] : (axBz2, axEx2, axEy2)}
    allData = {"Bz" : [], "Ex" : [], "Ey": []}

    for simNumber in simNumbers:

        combined_data_xr = xr.open_datatree(combinedSpectra_folder / f"run_{simNumber}_combined_stats.nc", engine="netcdf4")
        print(f"Run {simNumber} parameters:")
        print(f"Background Density: {combined_data_xr.backgroundDensity:.3e}")
        print(f"Beam fraction: {combined_data_xr.beamFraction:.3e}")
        print(f"B0 strength: {combined_data_xr.B0strength:.3e}")
        print(f"Pitch: {combined_data_xr.pitch:.3e}")

        for angle_dataPath in angle_folders:
            angle_dataPath = Path(angle_dataPath)
            print(f"Processing {simNumber} -- {angle_dataPath.absolute()}...")

            # Get this angle simulation's data
            angle_data_xr = xr.open_datatree(angle_dataPath / f"run_{simNumber}_stats.nc", engine="netcdf4")

            # Plots
            axBz = axes[simNumber][0]
            axEx = axes[simNumber][1]
            axEy = axes[simNumber][2]

            # Slice to maximum frequency
            bzData = angle_data_xr["Magnetic_Field_Bz/power/powerByFrequency"].sel(frequency = slice(0.1, maxXcoord))
            exData = angle_data_xr["Electric_Field_Ex/power/powerByFrequency"].sel(frequency = slice(0.1, maxXcoord))
            eyData = angle_data_xr["Electric_Field_Ey/power/powerByFrequency"].sel(frequency = slice(0.1, maxXcoord))
            coords = bzData.coords["frequency"] # Assumes coords are constant for the different spectra
            axBz.plot(
                coords.data, 
                bzData.data, 
                label = f"{angle_data_xr.attrs['B0angle']} degrees"
            )
            axEx.plot(
                coords.data, 
                exData.data,
                label = f"{angle_data_xr.attrs['B0angle']} degrees"
            )
            axEy.plot(
                coords.data, 
                eyData.data,
                label = f"{angle_data_xr.attrs['B0angle']} degrees"
            )

        # Slice to maximum frequency
        bzData = combined_data_xr["Magnetic_Field_Bz/power/powerByFrequency"].sel(frequency = slice(0.1, maxXcoord))
        exData = combined_data_xr["Electric_Field_Ex/power/powerByFrequency"].sel(frequency = slice(0.1, maxXcoord))
        eyData = combined_data_xr["Electric_Field_Ey/power/powerByFrequency"].sel(frequency = slice(0.1, maxXcoord))
        coords = bzData.coords["frequency"] # Assumes coords are constant for the different spectra
        allData["Bz"].extend(bzData)
        allData["Ex"].extend(exData)
        allData["Ey"].extend(eyData)
        axBz.plot(
            coords.data, 
            bzData.data, 
            label = "Sum"
        )
        axEx.plot(
            coords.data, 
            exData.data, 
            label = "Sum"
        )
        axEy.plot(
            coords.data, 
            eyData.data, 
            label = "Sum"
        )
    
    axes = {"Bz" : (axBz1, axBz2), "Ex" : (axEx1, axEx2), "Ey": (axEy1, axEy2)}
    for field, axs in axes.items():
        for ax in axs:
            ax.set_ylim(bottom=0)
            ax.grid()
            if field in axesToEquate:
                ax.set_ylim(top=np.max(allData[field]))
            
    handles, labels = axBz1.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    axBz2.legend(handles, labels)
    axBz1.legend(handles, labels)
    # fig.tight_layout()
    fig.align_labels()
    fig.subplots_adjust(wspace=0.1, hspace=0.25)
    plt.show()

def dispersion_relations_for_papers(
        dataFolder : Path, 
        outFolder : Path, 
        simNumber : int, 
        field : str, 
        maxK : float = 100.0,
        maxW : float = 100.0,
        fastSpecies : str = 'He-4 2+', 
        bkgdSpecies : str = 'D+'
    ):
    
    plt.rcParams.update({'axes.titlesize': 32.0})
    plt.rcParams.update({'axes.labelsize': 36.0})
    plt.rcParams.update({'xtick.labelsize': 28.0})
    plt.rcParams.update({'ytick.labelsize': 28.0})
    plt.rcParams.update({'legend.fontsize': 24.0})

    if "Bz" in field:
        field = "Magnetic_Field_Bz"
    elif "Ex" in field:
        field = "Electric_Field_Ex"
    elif "Ey" in field:
        field = "Electric_Field_Ey"

    print("Reading data....")
    # Open datafiles as xarray
    ds = xr.open_mfdataset(
        str(dataFolder / "*.sdf"),
        data_vars='minimal', 
        coords='minimal', 
        compat='override', 
        preprocess=SDFPreprocess()
    )

    # Drop initial conditions because they may not represent a solution
    ds = ds.sel(time=ds.coords["time"]>ds.coords["time"][0])

    # Read input deck
    inputDeck = {}
    with open(str(dataFolder / "input.deck")) as id:
        inputDeck = epydeck.loads(id.read())

    ion_gyroperiod = 2.0 * np.pi / ppf.gyrofrequency(inputDeck["constant"]["b0_strength"] * u.T, particle = fastSpecies)
    number_density_bkgd = float(inputDeck["constant"]["background_density"]) * (1.0 - inputDeck['constant']['frac_beam'])
    alfven_velocity = pps.Alfven_speed(inputDeck["constant"]["b0_strength"] * u.T, density = number_density_bkgd / u.m**3, ion = bkgdSpecies)

    print("Normalising data....")
    ds = epoch_utils.normalise_data(ds, ion_gyroperiod, alfven_velocity)

    # Take FFT
    print("Taking FFT....")
    original_spec : xr.DataArray = xrft.xrft.fft(ds[field].load(), true_amplitude=True, true_phase=True, window=None)
    original_spec = original_spec.rename(freq_time="frequency", freq_x_space="wavenumber")
    # Remove zero-frequency component
    original_spec = original_spec.where(original_spec.wavenumber!=0.0, None)

    # Dispersion relations
    print("Creating t-k spectrum....")
    tk_spec = epoch_utils.create_t_k_spectrum(original_spec, maxK = maxK, load=True, debug=True)
    dotFftUnits = r'$\cdot\frac{\Omega_{c,\alpha}}{v_A}$'
    tk_unit = f"{epoch_utils.fieldNameToUnit(field)}{dotFftUnits}"
    epoch_utils.create_t_k_plot(tk_spec, field, tk_unit, outFolder, f"run_{simNumber}", maxK, display=False)
    print("Creating w-k spectrum....")
    dummyStats = nc.Dataset(outFolder / "dummy.nc", mode="w")
    dotFftUnits = r'$\cdot\frac{\Omega_{c,\alpha}^2}{v_A}$'
    wk_unit = f"{epoch_utils.fieldNameToUnit(field)}{dotFftUnits}"
    _ = epoch_utils.create_omega_k_plots(original_spec, dummyStats, field, wk_unit, outFolder, f"run_{simNumber}", inputDeck, bkgdSpecies, fastSpecies, maxK=maxK, maxW=maxW, display=False, debug=True)
    print("Done.")

def energy_plots_for_papers(
        folder : Path, 
        maxTime : float = None,
        displayPlots : bool = True,
        doLog : bool = False,
        noTitle : bool = True,
        equateAxes : bool = True,
        compareTracesBeta : bool = False
):
    plt.rcParams.update({'axes.titlesize': 32.0})
    plt.rcParams.update({'axes.labelsize': 32.0})
    plt.rcParams.update({'xtick.labelsize': 28.0})
    plt.rcParams.update({'ytick.labelsize': 28.0})
    plt.rcParams.update({'legend.fontsize': 20.0})

    angles = glob.glob(str(folder / "data"/ "9*"))
    upperBound = float("-inf")
    lowerBound = float("inf")
    energyTraces = {}
    energyFields = [
        "/Energy/backgroundIonMeanEnergyDensity", 
        "/Energy/electronMeanEnergyDensity", 
        "/Energy/magneticFieldMeanEnergyDensity", 
        "/Energy/electricFieldMeanEnergyDensity", 
        "/Energy/fastIonMeanEnergyDensity"
    ]
    timeCoords = None

    # Get all data
    for angle in angles:
        
        # Get simulation stats files
        simFiles = glob.glob(str(Path(angle) / "*_stats.nc"))

        for s in simFiles:
            stats = xr.open_datatree(
                s,
                engine="netcdf4"
            )
            energyTraces[s] = dict.fromkeys(energyFields)

            if timeCoords is None: # Assuming constant
                timeCoords = stats["/Energy"].coords["time"].data
            
            totalDeltaED = np.zeros_like(stats["/Energy/backgroundIonMeanEnergyDensity"].to_numpy())
            for field in energyFields:
                deltaED = (stats[field] - stats[field].isel({"time" : 0})).to_numpy() / float(stats["/Energy/fastIonMeanEnergyDensity"].isel({"time" : 0}).data)
                totalDeltaED += deltaED
                energyTraces[s][field] = 100.0 * deltaED
            energyTraces[s]["totalMeanEnergyDensity"] = 100.0 * totalDeltaED
            if equateAxes:
                for trace in energyTraces[s].values():
                    if np.max(trace) > upperBound:
                        upperBound = np.max(trace)
                    if np.min(trace) < lowerBound:
                        lowerBound = np.min(trace)
                # Increase by 5% for viewability
                upperBound *= 1.05
                lowerBound *= 1.05

    # Plot
    for statsFile, tracesByField in energyTraces.items():
        
        simNumber = Path(statsFile).name.split("_")[1]
        angle = Path(statsFile).parent.name

        if compareTracesBeta:
            sameSim = [s for s in energyTraces.keys() if f"run_{simNumber}" in s and f"data/{angle}/" not in s][0]
            sameAngle = [s for s in energyTraces.keys() if f"data/{angle}/" in s and f"run_{simNumber}" not in s][0]
            sameSimTraces = energyTraces[sameSim]
            sameAngleTraces = energyTraces[sameAngle]

        # Initialise plotting
        fig, ax = plt.subplots(figsize=(12, 8))
        filename = Path(f"run_{simNumber}_angle_{angle}_percentage_energy_change.png")
        totalDeltaED = np.zeros_like(tracesByField["/Energy/backgroundIonMeanEnergyDensity"])
        
        # Iterate deltas
        for variable, percentageED in tracesByField.items(): 
            
            colour = next((epoch_utils.E_TRACE_SPECIES_COLOUR_MAP[c] for c in epoch_utils.E_TRACE_SPECIES_COLOUR_MAP.keys() if c in variable), False)
            ax.plot(timeCoords, percentageED, label=f"{epoch_utils.SPECIES_NAME_MAP[variable]}", color = colour)
            if compareTracesBeta:
                if variable == "totalMeanEnergyDensity": # Fixes legend
                    ax.plot(timeCoords, sameSimTraces[variable], label="Same simulation", linestyle = '--', color = colour, alpha = 0.6)
                    ax.plot(timeCoords, sameAngleTraces[variable], label="Same angle", linestyle = ':', color = colour, alpha = 0.6)
                else:
                    ax.plot(timeCoords, sameSimTraces[variable], linestyle = '--', color = colour, alpha = 0.6)
                    ax.plot(timeCoords, sameAngleTraces[variable], linestyle = ':', color = colour, alpha = 0.6)
        
        ax.legend()
        ax.set_xlabel(r"Time [$\tau_{ci}$]")
        ax.set_ylabel("Change in energy density [%]")
        if equateAxes:
            ax.set_ylim(bottom=lowerBound, top=upperBound)
        if doLog:
            ax.set_yscale("symlog")
        ax.grid()
        if not noTitle:
            ax.set_title(f"Run {simNumber} Angle {angle}: Percentage change in ED relative to fast ion energy")
        fig.tight_layout()
        fig.savefig(folder / filename)
        if displayPlots:
            plt.show()
        plt.close("all")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--dataFile",
        action="store",
        help="Filepath of simulation output data to plot, e.g. /data/run_23_stats.nc",
        required = False,
        type=Path
    )
    parser.add_argument(
        "--powerSpectrum",
        action="store_true",
        help="Calculate single power spectrum from a netcdf data file.",
        required = False
    )
    parser.add_argument(
        "--compareSpectra",
        action="store_true",
        help="Compare two sets of combined spectra. Requires two simulation number, plus a folder containing their original uncombined data and combined spectra (in subfolders).",
        required = False
    )
    parser.add_argument(
        "--dispersion",
        action="store_true",
        help="Plot dispersion relations in paper-friendly way.",
        required = False
    )
    parser.add_argument(
        "--energy",
        action="store_true",
        help="Plot energy traces in paper-friendly way.",
        required = False
    )
    parser.add_argument(
        "--dataFolder",
        action="store",
        help="Filepath of folder of simulation output data to plot, e.g. /data/",
        required = False,
        type=Path
    )
    parser.add_argument(
        "--outputFolder",
        action="store",
        help="Filepath of folder for plot output.",
        required = False,
        type=Path
    )
    parser.add_argument(
        "--fields",
        action="store",
        help="Field(s) to plot, e.g. /Magnetic_Field_Bz/power/powerByFrequency",
        required = True,
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--sims",
        action="store",
        help="Simulation numbers to compare",
        required = False,
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--maxK",
        action="store",
        help="Maximum wavenumber for plotting.",
        required = False,
        type=float
    )
    parser.add_argument(
        "--maxW",
        action="store",
        help="Maximum frequency for plotting.",
        required = False,
        type=float
    )
    parser.add_argument(
        "--equateAxes",
        action="store",
        help="Equate y-axes for the specified fields, e.g. Bz, Ex",
        required = False,
        type=str,
        nargs="*"
    )

    args = parser.parse_args()

    plt.rcParams.update({'axes.titlesize': 26.0})
    plt.rcParams.update({'axes.labelsize': 26.0})
    plt.rcParams.update({'xtick.labelsize': 18.0})
    plt.rcParams.update({'ytick.labelsize': 18.0})
    plt.rcParams.update({'legend.fontsize': 16.0})

    if args.powerSpectrum:
        if args.dataFile:
            for f in args.fields:
                power_spectrum(args.dataFile, f)
        if args.dataFolder:
            all_power_spectra(args.dataFolder, args.fields)
    if args.compareSpectra:
        compare_spectra(args.dataFolder, args.sims, 50, args.equateAxes)
    if args.dispersion:
        dispersion_relations_for_papers(args.dataFolder, args.outputFolder, args.sims[0], field=args.fields[0], maxK=args.maxK, maxW=args.maxW)
    if args.energy:
        energy_plots_for_papers(args.dataFolder)

