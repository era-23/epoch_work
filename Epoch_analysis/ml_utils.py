from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import itertools
import csv
import GPy
from scipy.stats import norm
from scipy.interpolate import griddata
from scipy.signal import find_peaks
from sklearn.pipeline import Pipeline
import xarray as xr
from functools import partial
from matplotlib import animation
from inference.plotting import matrix_plot
from SALib import ProblemSpec
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sktime.clustering.dbscan import TimeSeriesDBSCAN
from sktime.clustering.k_means import TimeSeriesKMeans, TimeSeriesKMeansTslearn
from sktime.clustering.k_medoids import TimeSeriesKMedoids
from sktime.clustering.k_shapes import TimeSeriesKShapes
from sktime.clustering.kernel_k_means import TimeSeriesKernelKMeans
from sktime.regression.deep_learning import CNNRegressor, CNTCRegressor, FCNRegressor, InceptionTimeRegressor, LSTMFCNRegressor, MACNNRegressor, MCDCNNRegressor, MLPRegressor, ResNetRegressor, SimpleRNNRegressor, TapNetRegressor
from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor
from sktime.regression.interval_based import TimeSeriesForestRegressor
from sktime.regression.kernel_based import TimeSeriesSVRTslearn, RocketRegressor
from sktime.regression.compose import ComposableTimeSeriesForestRegressor
from sklearn.preprocessing import StandardScaler
import dataclasses as dc
from dataclasses import dataclass
from numpy.typing import ArrayLike
import SALib.sample as salsamp

@dataclass
class GPModel:
    kernelName : str
    inputNames : list
    normalisedInputs : np.ndarray
    outputName : str
    output : np.ndarray
    regressionModel : GaussianProcessRegressor | GPy.models.GPCoregionalizedRegression | GPy.models.GPMultioutRegression = None
    classificationModel : GaussianProcessClassifier = None
    modelParams : dict = None
    fitSuccess : bool = None

@dataclass
class GPResults:
    kernelNames : list
    inputNames : list
    outputNames : list
    output : np.ndarray
    sobolIndicesO1 : list
    sobolIndicesO2 : list
    sobolIndicesTotal : list
    cvAccuracyMean : float = 0.0
    cvAccuracyVar : float = 0.0
    spectrumName : str = None
    regressionModel : GaussianProcessRegressor | GPy.models.GPCoregionalizedRegression | GPy.models.GPMultioutRegression = None
    fitSuccess : bool = False
    objective : float = 0.0

@dataclass
class SimulationMetadata:
    simId : int
    backgroundDensity : float
    beamFraction : float
    B0 : float
    B0angle : float

@dataclass
class SpectralFeatures1D:
    peaksFound : bool = False
    peakPowers : ArrayLike = None # Powers of all peaks
    peakCoordinates : ArrayLike  = None # Coordinates (e.g. frequencies) of all peaks
    numPeaks : int = 0 # Number of peaks
    maxPeakPower : float = 0.0 # Power of highest peak
    maxPeakCoordinate : float = 0.0 # Coordinate of highest power peak
    meanPeakNPowers : ArrayLike  = None # Means of n highest peaks, ascending in peak count, which are descending in power, i.e. mean of 1 highest, mean of 2 highest etc.
    meanPeak4Powers : float = 0.0 # Power mean of 4 highest peaks, if applicable
    meanPeak3Powers : float = 0.0 # Power mean of 3 highest peaks, if applicable
    meanPeak2Powers : float = 0.0 # Power mean of 2 highest peaks, if applicable
    varPeakNPowers : ArrayLike = 0.0 # Power variance of n highest power peaks, ordered as with means above
    varPeak4Powers : float = 0.0 # Power variance of 4 highest peaks, if applicable
    varPeak3Powers : float = 0.0 # Power variance of 4 highest peaks, if applicable
    varPeak2Powers : float = 0.0 # Power variance of 4 highest peaks, if applicable
    peakProminences: ArrayLike = None # Prominences of all peaks, in coordinate order
    maxPeakProminence : float = 0.0 # Maximum peak prominence found in spectrum
    meanPeakProminence : float = 0.0 # Mean peak prominence found in spectrum
    varPeakProminence : float = 0.0 # Variance in peak prominence found in spectrum
    maxPeakPowerProminence : float = 0.0 # Prominence of highest power peak
    activeRegions : ArrayLike = None # Indices of active spectral regions, i.e regions with any peaks within short distance (configurable) of each other
    numActiveRegions : int = 0 # Number of active regions
    activeRegionMeanPowers : ArrayLike = None # Mean peak power within each active region, in coordinate order 
    activeRegionVarPowers : ArrayLike = None # Variance of peak power within each active region, in coordinate order 
    activeRegionMeanCoordinates : ArrayLike = None # Coordinate centres of each active region
    activeRegionPeakSeparations : ArrayLike = None # Mean peak separation (in coordinate units) within each active region
    activeRegionMeanPeakSeparations : float = 0.0 # Mean of mean peak separations within active regions
    activeRegionVarPeakSeparations : float = 0.0 # Variance of mean peak separations within active regions
    activeRegionCoordWidths : ArrayLike = None # Width in coordinate units of each active region
    activeRegionMeanCoordWidths : float = 0.0 # Mean of active region coordinate widths
    activeRegionVarCoordWidths : float = 0.0 # Variance in active region coordinate widths
    totalActiveCoordinateProportion : float = 0.0 # Total proportion of signal that is in active regions (in coordinate units)
    totalActivePowerProportion : float = 0.0 # Total proportion of signal power that is in active regions (in y-axis/power units)
    spectrumSum : float = 0.0 # Total sum of spectral power
    spectrumMean : float = 0.0 # Mean of spectral power
    spectrumVar : float = 0.0 # Variance in spectral power

fieldNameToText_dict = {
    "Energy/electricFieldEnergyDensity_delta" : "E_deltaE",
    "Energy/magneticFieldEnergyDensity_delta" : "B_deltaE", 
    "Energy/backgroundIonEnergyDensity_delta" : "background\ndelta KE", 
    "Energy/electronEnergyDensity_delta" : "e_deltaKE",
    "Energy/fastIonEnergyDensity_max" : "fast ion\nmax KE", 
    "Energy/fastIonEnergyDensity_timeMax" : "FI_timeMaxKE", 
    "Energy/fastIonEnergyDensity_min" : "fast ion\nmin KE", 
    "Energy/fastIonEnergyDensity_timeMin" : "fast ion\nmin time", 
    "Energy/fastIonEnergyDensity_delta" : "fast ion\ndelta KE",
    "Energy/backgroundIonEnergyDensity_max" : "background\nmax KE",
    "Energy/backgroundIonEnergyDensity_timeMax" : "background\npeak time",
    "Energy/electronEnergyDensity_timeMax" : "electron_timeMaxKE",
    "Energy/electricFieldEnergyDensity_timeMax" : "Ex_timeMaxE",
    "Energy/magneticFieldEnergyDensity_timeMax" : "Bz_timeMaxE",
    "Energy/backgroundIonEnergyDensity_timeMin" : "bkgdIon_timeMinKE",
    "Energy/electronEnergyDensity_timeMin" : "electron_timeMinKE",
    "Energy/electricFieldEnergyDensity_timeMin" : "Ex_timeMinE",
    "Energy/magneticFieldEnergyDensity_timeMin" : "Bz_timeMinE",
    "Magnetic_Field_Bz/totalMagnitude" : "Bz_totalPower", 
    "Magnetic_Field_Bz/meanMagnitude" : "Bz_meanPower", 
    "Magnetic_Field_Bz/totalDelta" : "Bz_deltaTotalPower", 
    "Magnetic_Field_Bz/meanDelta" : "Bz_deltaMeanPower", 
    "Magnetic_Field_Bz/peakTkSpectralPower" : "Bz (peak) [T]", 
    "Magnetic_Field_Bz/meanTkSpectralPower" : "Bz (mean)", 
    "Magnetic_Field_Bz/peakTkSpectralPowerRatio" : "Bz_tkPowerRatio", 
    "Magnetic_Field_Bz/growthRates/max/growthRate" : "Bz_maxGamma",
    "Magnetic_Field_Bz/growthRates/max/peakPower" : "Bz_maxGammaPeakPower",
    "Magnetic_Field_Bz/growthRates/max/residual" : "Bz_maxGammaFitResidual",
    "Magnetic_Field_Bz/growthRates/max/time" : "Bz_maxGammaTime",
    "Magnetic_Field_Bz/growthRates/max/totalPower" : "Bz_maxGammaTotalPower",
    "Magnetic_Field_Bz/growthRates/max/wavenumber" : "Bz_maxGammaK",
    "Magnetic_Field_Bz/growthRates/maxInHighPeakPowerK/growthRate" : "Bz_peakKmaxGamma",
    "Magnetic_Field_Bz/growthRates/maxInHighPeakPowerK/peakPower" : "Bz_peakKmaxGammaPeakPower",
    "Magnetic_Field_Bz/growthRates/maxInHighPeakPowerK/residual" : "Bz_peakKmaxGammaFitResidual",
    "Magnetic_Field_Bz/growthRates/maxInHighPeakPowerK/time" : "Bz_peakKmaxGammaTime",
    "Magnetic_Field_Bz/growthRates/maxInHighPeakPowerK/totalPower" : "Bz_peakKmaxGammaTotalPower",
    "Magnetic_Field_Bz/growthRates/maxInHighPeakPowerK/wavenumber" : "Bz_peakKmaxGammaK",
    "Magnetic_Field_Bz/growthRates/maxInHighTotalPowerK/growthRate" : "Bz_totalKmaxGamma",
    "Magnetic_Field_Bz/growthRates/maxInHighTotalPowerK/peakPower" : "Bz_totalKmaxGammaPeakPower",
    "Magnetic_Field_Bz/growthRates/maxInHighTotalPowerK/residual" : "Bz_totalKmaxGammaFitResidual",
    "Magnetic_Field_Bz/growthRates/maxInHighTotalPowerK/time" : "Bz_totalKmaxGammaTime",
    "Magnetic_Field_Bz/growthRates/maxInHighTotalPowerK/totalPower" : "Bz_totalKmaxGammaTotalPower",
    "Magnetic_Field_Bz/growthRates/maxInHighTotalPowerK/wavenumber" : "Bz_totalKmaxGammaK",
    "Magnetic_Field_Bz/growthRates/positive/bestInHighestPeakPowerK/growthRate" : "Growth rate (peak power k)",
    "Magnetic_Field_Bz/growthRates/positive/bestInHighestPeakPowerK/time" : "Time max growth (peak power k)",
    "Magnetic_Field_Bz/growthRates/positive/bestInHighestPeakPowerK/wavenumber" : "Peak k",
    "Magnetic_Field_Bz/growthRates/positive/bestInHighestTotalPowerK/growthRate" : "Growth rate (total power k)",
    "Magnetic_Field_Bz/growthRates/positive/bestInHighestTotalPowerK/time" : "Time max growth (total power k)",
    "Magnetic_Field_Bz/growthRates/positive/bestInHighestTotalPowerK/wavenumber" : "Total power k",
    "Electric_Field_Ex/totalMagnitude" : "Ex_totalPower", 
    "Electric_Field_Ex/meanMagnitude" : "Ex_meanPower", 
    "Electric_Field_Ex/totalDelta" : "Ex_deltaTotalPower", 
    "Electric_Field_Ex/meanDelta" : "Ex_deltaMeanPower", 
    "Electric_Field_Ex/peakTkSpectralPower" : "Ex (peak)", 
    "Electric_Field_Ex/meanTkSpectralPower" : "Ex (mean)", 
    "Electric_Field_Ex/peakTkSpectralPowerRatio" : "Ex_tkPowerRatio",
    "Electric_Field_Ex/growthRates/max/growthRate" : "Ex_maxGamma",
    "Electric_Field_Ex/growthRates/max/peakPower" : "Ex_maxGammaPeakPower",
    "Electric_Field_Ex/growthRates/max/residual" : "Ex_maxGammaFitResidual",
    "Electric_Field_Ex/growthRates/max/time" : "Ex_maxGammaTime",
    "Electric_Field_Ex/growthRates/max/totalPower" : "Ex_maxGammaTotalPower",
    "Electric_Field_Ex/growthRates/max/wavenumber" : "Ex_maxGammaK",
    "Electric_Field_Ex/growthRates/maxInHighPeakPowerK/growthRate" : "Ex_peakKmaxGamma",
    "Electric_Field_Ex/growthRates/maxInHighPeakPowerK/peakPower" : "Ex_peakKmaxGammaPeakPower",
    "Electric_Field_Ex/growthRates/maxInHighPeakPowerK/residual" : "Ex_peakKmaxGammaFitResidual",
    "Electric_Field_Ex/growthRates/maxInHighPeakPowerK/time" : "Ex_peakKmaxGammaTime",
    "Electric_Field_Ex/growthRates/maxInHighPeakPowerK/totalPower" : "Ex_peakKmaxGammaTotalPower",
    "Electric_Field_Ex/growthRates/maxInHighPeakPowerK/wavenumber" : "Ex_peakKmaxGammaK",
    "Electric_Field_Ex/growthRates/maxInHighTotalPowerK/growthRate" : "Ex_totalKmaxGamma",
    "Electric_Field_Ex/growthRates/maxInHighTotalPowerK/peakPower" : "Ex_totalKmaxGammaPeakPower",
    "Electric_Field_Ex/growthRates/maxInHighTotalPowerK/residual" : "Ex_totalKmaxGammaFitResidual",
    "Electric_Field_Ex/growthRates/maxInHighTotalPowerK/time" : "Ex_totalKmaxGammaTime",
    "Electric_Field_Ex/growthRates/maxInHighTotalPowerK/totalPower" : "Ex_totalKmaxGammaTotalPower",
    "Electric_Field_Ex/growthRates/maxInHighTotalPowerK/wavenumber" : "Ex_totalKmaxGammaK",
    "B0strength" : "B0", 
    "B0angle" : "B0 angle", 
    "backgroundDensity" : "density", 
    "beamFraction" : "beam frac",
}

def get_algorithm(name, **kwargs):
    match name:
        case "TimeSeriesDBSCAN":
            return TimeSeriesDBSCAN(kwargs["distance"])
        case "TimeSeriesKMeans":
            return TimeSeriesKMeans(kwargs["n_clusters"])
        case "TimeSeriesKMeansTslearn":
            return TimeSeriesKMeansTslearn(kwargs["n_clusters"])
        case "TimeSeriesKMedoids": 
            return TimeSeriesKMedoids(kwargs["n_clusters"])
        case "TimeSeriesKShapes":
            return TimeSeriesKShapes(kwargs["n_clusters"])
        case "TimeSeriesKernelKMeans":
            return TimeSeriesKernelKMeans(kwargs["n_clusters"])
        case "CNNRegressor":
            return CNNRegressor(kwargs)
        case "CNTCRegressor": 
            return CNTCRegressor(kwargs)
        case "FCNRegressor":
            return FCNRegressor(kwargs)
        case "InceptionTimeRegressor":
            return InceptionTimeRegressor(kwargs)
        case "KNeighborsTimeSeriesRegressor":
            return KNeighborsTimeSeriesRegressor(n_neighbors=kwargs["n_neighbours"], distance=kwargs["distance"])
        case "LSTMFCNRegressor":
            return LSTMFCNRegressor(kwargs)
        case "MACNNRegressor":
            return MACNNRegressor(kwargs)
        case "MCDCNNRegressor": 
            return MCDCNNRegressor(kwargs)
        case "MLPRegressor":
            return MLPRegressor(kwargs)
        case "ResNetRegressor": 
            return ResNetRegressor(kwargs)
        case "RocketRegressor":
            return RocketRegressor(kwargs)
        case "SimpleRNNRegressor":
            return SimpleRNNRegressor(kwargs)
        case "TapNetRegressor":
            return TapNetRegressor(kwargs)
        case "TimeSeriesSVRTslearn":
            return TimeSeriesSVRTslearn(kwargs)
        case "TimeSeriesForestRegressor":
            return TimeSeriesForestRegressor(kwargs)
        case "ComposableTimeSeriesForestRegressor":
            return ComposableTimeSeriesForestRegressor(kwargs)

def read_data(dataFiles, data_dict : dict, with_names : bool = False, with_coords : bool = False) -> dict:
    
    dataFields = list(data_dict.keys())

    if with_coords:
        for fieldPath in dataFields:
            dimCoordsKey = fieldPath + "_coords"
            data_dict[dimCoordsKey] = []

    if with_names:
        data_dict["sim_ids"] = []
    
    for simulation in dataFiles:

        data = xr.open_datatree(
            simulation,
            engine="netcdf4"
        )

        for fieldPath in dataFields:
            s = fieldPath.split("/")
            group = "/" if len(s) == 1 else '/'.join(s[:-1])
            fieldName = s[-1]
            if fieldName in data[group].attrs.keys():
                data_dict[fieldPath].append(data[group].attrs[fieldName])
            else:
                data_dict[fieldPath].append(data[group].variables[fieldName].values)
                # Get coordinates if requested
                if with_coords:
                    dimCoordsKey = fieldPath + "_coords"
                    dimName = data[group].variables[fieldName].dims[0]
                    dimVals = data[group].coords[dimName].values
                    data_dict[dimCoordsKey].append(dimVals)
        
        if with_names:
            data_dict["sim_ids"].append(simulation.split("/")[-1].split("_")[1])

    return data_dict

def read_spectral_features_from_csv(csvFile : Path) -> list:
    featureSet = []
    with open(csvFile, mode="r") as featureFile:
        allRows = csv.DictReader(featureFile)
        for row in allRows:
            instance = SpectralFeatures1D()
            for k, v in row.items():
                if '[[' in v:
                    # List of lists
                    instance.__dict__[k] = [[int(x.replace('[','').replace(']','')) for x in y.split(', ')] for y in v.split('], [')]
                elif '[' in v:
                    # List
                    v = v.replace('[','').replace(']','')
                    vs = v.split(' ')
                    instance.__dict__[k] = np.array([float(f) for f in vs if f and not f.isspace()])
                elif v == 'nan':
                    # nan
                    instance.__dict__[k] - np.nan
                elif '.' in v:
                    # Float
                    instance.__dict__[k] = float(v)
                elif v == "False" or v == "True":
                    # Bool
                    instance.__dict__[k] = bool(v == "True") 
                elif v:
                    # Int
                    instance.__dict__[k] = int(v) 
            featureSet.append(instance)

    return featureSet

def write_spectral_features_to_csv(csvFile : Path, featureSet : list):
    with open(csvFile, mode="w") as featureFile:
        writer = csv.DictWriter(featureFile, fieldnames=[f.name for f in dc.fields(SpectralFeatures1D)])
        writer.writeheader()
        for instance in featureSet:
            writer.writerow(instance.__dict__)

def anim_init(fig, ax, xx, yy, zz):
    ax.scatter(xx, yy, zz, color="black")
    return fig,

def anim_animate(i, fig, ax):
    ax.view_init(elev=10., azim=i)
    return fig,

def fieldNameToText(name : str) -> str:
    if name in fieldNameToText_dict:
        return fieldNameToText_dict[name]
    else:
        return name

def parse_commandLine_netCDFpaths(paths : list) -> dict:
    
    formattedPaths = dict()
    root_path = "/"

    for path in paths:
        s = path.split("/")
        field = s[-1]

        if len(s) == 1:
            group = root_path
        elif len(s) == 2:
            group = s[0]
        else:
            raise NotImplementedError("netCDF path length is too many groups deep")
        
        if group not in formattedPaths:
            formattedPaths[group] = []
        formattedPaths[group].append(field)

    return formattedPaths

def plot_three_dimensions(input_1_index : int, input_2_index : int, gpModel : GPModel, rawInputData : list = None, rawOutputData : list = None, showModels = True, saveAnimation = False, noTitle = False):
    
    # if showModels:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection='3d')
    #     ax.scatter(gpModel.normalisedInputs[:,input_1_index], gpModel.normalisedInputs[:,input_2_index], gpModel.output)
    #     ax.set_xlabel(fieldNameToText(gpModel.inputNames[input_1_index]))
    #     ax.set_ylabel(fieldNameToText(gpModel.inputNames[input_2_index]))
    #     ax.set_zlabel(fieldNameToText(gpModel.outputName))
    #     if not noTitle:
    #         ax.set_title(f"Training data for output {fieldNameToText(gpModel.outputName)} ({gpModel.kernelName} kernel)")

    #     plt.show()
    plt.close()

    # Sample homogeneously and plot contours with training data
    sp = ProblemSpec({
        'num_vars': gpModel.normalisedInputs.shape[1],
        'names': gpModel.inputNames,
        'bounds': [[np.min(column), np.max(column)] for column in gpModel.normalisedInputs.T]
    })
    gp_test_values = salsamp.sobol.sample(sp, int(2**10))
    for column in range(gp_test_values.shape[1]):
        if column is not input_1_index and column is not input_2_index:
            mean = np.mean(np.array(gp_test_values[:,column]))
            gp_test_values[:,column] = mean

    # Training data
    fig = plt.figure(figsize=(10, 8))
    #ax = fig.add_subplot(projection='3d')
    #ax.set_xlabel('\n' + fieldNameToText(gpModel.inputNames[input_1_index]))
    #ax.set_ylabel('\n' + fieldNameToText(gpModel.inputNames[input_2_index]))
    #ax.set_zlabel('\n' + fieldNameToText(gpModel.outputName))

    # GP predictions
    if gpModel.regressionModel:
        gp_z = gpModel.regressionModel.predict(gp_test_values)
    elif gpModel.classificationModel:
        gp_z = gpModel.classificationModel.predict_proba(gp_test_values)[:,1]
    else:
        raise ValueError("No regression or classification model found.")
    gp_x, gp_y = gp_test_values[:,input_1_index], gp_test_values[:,input_2_index]
    #ax.plot_trisurf(gp_x, gp_y, gp_z, cmap="plasma")
    X, Y = np.meshgrid(np.linspace(min(gp_x), max(gp_x), len(gp_x)), np.linspace(min(gp_y), max(gp_y), len(gp_y)))
    Z = griddata((gp_x, gp_y), gp_z, (X, Y), method='cubic')
    plt.imshow(Z, extent=[gp_x.min(), gp_x.max(), gp_y.min(), gp_y.max()], origin='lower', cmap='plasma', aspect='auto')
    cbar = plt.colorbar(label=fieldNameToText(gpModel.outputName))
    cbarTickLocs = cbar.get_ticks(minor=False)
    # print(cbarTickLocs)
    rawCbarLabs = [np.round(d, 2) for d in np.linspace(np.min(rawOutputData), np.max(rawOutputData), len(cbarTickLocs))]
    # print(rawCbarLabs)
    cbar.set_ticks(ticks=cbarTickLocs[1:-1], labels=rawCbarLabs[1:-1])
    xTickLocs = plt.xticks(minor=False)[0]
    # print(xTickLocs)
    rawXTickLabs = [np.round(d, 1) for d in np.linspace(np.min(rawInputData[0]), np.max(rawInputData[0]), len(xTickLocs))]
    # print(rawXTickLabs)
    plt.xticks(ticks=xTickLocs[1:-1], labels=rawXTickLabs[1:-1])
    yTickLocs = plt.yticks(minor=False)[0]
    # print(yTickLocs)
    rawYTickLabs = [np.round(d, 1) for d in np.linspace(np.min(rawInputData[1]), np.max(rawInputData[1]), len(yTickLocs))]
    # print(rawYTickLabs)
    plt.yticks(ticks=yTickLocs[1:-1], labels=rawYTickLabs[1:-1])
    plt.xlabel(fieldNameToText(gpModel.inputNames[input_1_index]))
    plt.ylabel(fieldNameToText(gpModel.inputNames[input_2_index]))
    if not noTitle:
        #ax.set_title(f"Training data and GP prediction surface ({gpModel.kernelName})")
        plt.title(f"Training data and GP prediction surface ({gpModel.kernelName})")
    if showModels:
        plt.show()
    plt.close()

    # if saveAnimation:
    #     # Animate
    #     anim = animation.FuncAnimation(
    #         fig, 
    #         partial(anim_animate, fig=fig, ax=ax), 
    #         init_func=partial(anim_init, fig=fig, ax=ax, xx=gpModel.normalisedInputs[:,input_1_index], yy=gpModel.normalisedInputs[:,input_2_index], zz=gpModel.output),
    #         frames=360, 
    #         interval=20, 
    #         blit=False)
    #     # Save
    #     anim.save(f'/home/era536/Documents/for_discussion/2025.02.27/{gpModel.inputNames[input_1_index].replace("/", "-")}_and_{gpModel.inputNames[input_2_index].replace("/", "-")}_vs{gpModel.outputName.replace("/", "-")}_{gpModel.kernelName}.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    # plt.close()

def display_matrix_plots(inputs : dict, normalisedInputs : np.ndarray, outputs: dict, normalisedOutputs : dict = {}, noTitle : bool = False):

    for outName, outData in outputs.items():
        data = [outData] + list(inputs.values())
        labels = [fieldNameToText(outName)] + [fieldNameToText(i) for i in inputs.keys()]
        matrix_plot(data, labels, show=False)
        if not noTitle:
            plt.title("Raw data")
        plt.tight_layout()
        plt.show()

        if normalisedOutputs:
            data = list(normalisedInputs.T)
            data.insert(0, list(normalisedOutputs[outName]))
            labels = [fieldNameToText(outName)] + [fieldNameToText(i) for i in inputs.keys()]
            matrix_plot(data, labels, show=False)
            if not noTitle:
                plt.title("Normalised data")
            plt.tight_layout()
            plt.show()

    plt.close()

def plot_models(gpModels : list, rawInputData : dict = None, rawOutputData : dict = None, showModels : bool = True, saveAnimation : bool = False, noTitle : bool = False):
    
    for model in gpModels:
        model : GPModel
        # 3D plot if there are only 2 input dimensions
        allInputCombos = list(itertools.combinations(range(len(model.inputNames)), 2))
        for inputPair in allInputCombos:
            i1 = inputPair[0]
            i2 = inputPair[1]
            plot_three_dimensions(i1, i2, model, [list(rawInputData.values())[i1], list(rawInputData.values())[i2]], list(rawOutputData.values()), showModels, saveAnimation, noTitle)

        # Sample values of each input dim keeping others fixed at their mean value
        zScore = abs(norm.ppf(0.975))
        numSamples = 1000
        for i in range(len(model.inputNames)):
            inputName = fieldNameToText(model.inputNames[i])
            sampleColumn = model.normalisedInputs[:,i]
            samples = np.linspace(sampleColumn.min(), sampleColumn.max(), numSamples)

            allDatapoints = [[] for _ in model.inputNames]
            trainingDistancesForAlphas = np.zeros(len(sampleColumn))
            for i2 in range(len(model.inputNames)):
                if i2 == i:
                    allDatapoints[i2] = samples
                else:
                    mean = model.normalisedInputs[:,i2].mean()
                    allDatapoints[i2] = np.ones(numSamples) * mean
                    # Sum geometric distance from non-x-axis dim means to calculate how relevant each point is for this view
                    trainingDistancesForAlphas += (model.normalisedInputs[:,i2] - mean)**2

            # Columnise
            allDatapoints = np.array(allDatapoints).T

            # Sample
            y_preds = []
            if model.regressionModel:
                y_preds, y_std = model.regressionModel.predict(allDatapoints, return_std=True)
                plt.fill_between(
                    samples,
                    y_preds - (zScore * y_std),
                    y_preds + (zScore * y_std),
                    color="tab:red",
                    alpha=0.3,
                    label=r"95% confidence interval"
                )
            elif model.classificationModel:
                y_preds = model.classificationModel.predict_proba(allDatapoints)[:,1]
            else:
                raise NotImplementedError("model must be Gaussian Process regression or classification")

            # Calculate alphas
            trainingDistancesForAlphas = np.sqrt(trainingDistancesForAlphas)
            normDistance = (trainingDistancesForAlphas - np.min(trainingDistancesForAlphas)) / (np.max(trainingDistancesForAlphas) - np.min(trainingDistancesForAlphas))
            normAlpha = 1.0 - normDistance # Closer is better therefore higher alpha

            # Plot
            plt.plot(samples, y_preds, label="Mean prediction", color="tab:red")
            plt.scatter(sampleColumn, model.output, label="Training data", alpha=normAlpha)
            plt.xlabel(inputName)
            plt.ylabel(fieldNameToText(model.outputName))
            plt.legend()
            #plt.tight_layout()
            if not noTitle:
                plt.title(f"GP predictions for {fieldNameToText(model.outputName)} ({model.kernelName} kernel), all other input dimensions fixed to mean value", wrap=True)
            plt.show()

def convert_input_for_multi_output_GPy_model(x, num_outputs):
    """
    This functions brings test data to the correct shape making it possible to use the `predict()` method of a trained
    `GPy.util.multioutput.ICM` model (in the case that all outputs have the same input data).

    Behind the scenes, this model is using an extended input space with an additional dimension that points at the
    output each data point belongs to. To make use of the prediction function of GPy, this model needs the input array
    to have the extended format, i.e. adding a column indicating which input should be used for which output.

    This function works also for input dimensions > 1.

    :param x: the x data you want to predict on
    :param num_outputs: The number of outputs in your model
    """

    xt2d = np.array([x[:, i] for i in range(x.shape[1])]).T
    xt = [xt2d] * num_outputs
    identity_matrix = np.hstack([np.repeat(j, _x.shape[0]) for _x, j in zip(xt, range(num_outputs))])
    xt = np.vstack(xt)
    xt = np.hstack([xt, identity_matrix[:, None]])

    return xt

def extract_features_from_1D_power_spectrum(
        spectrum : np.ndarray, 
        coordinates : np.ndarray, 
        peak_prominence_factor : float = 0.125, 
        peak_distance_coordUnits : float = 0.5, 
        excited_region_distance_coordUnits : float = 3.0, 
        spectrum_name : list = None, 
        savePath : Path = None,
        xLabel : str = None) -> dict:

    # Find peaks
    max_height = np.max(spectrum)
    peak_prominence = peak_prominence_factor * max_height
    # Set distance between peaks as n * the coordinate unit (e.g. 0.5 gyrofrequencies)
    peak_distance_idx = int(np.round((peak_distance_coordUnits * len(coordinates)) / (np.max(coordinates) - np.min(coordinates))))
    peaks_idx, peaks_properties = find_peaks(spectrum, distance=peak_distance_idx, prominence=peak_prominence)

    # Find excited regions
    active_region_idx = int(np.round((excited_region_distance_coordUnits * len(coordinates)) / (np.max(coordinates) - np.min(coordinates)))) 
    # Active region is to within N * prominence condition (e.g. 2 gyrofrequencies)
    active_regions_idx = []
    if len(peaks_idx) > 0:
        if len(peaks_idx) > 1:
            active_region = []
            for coord_idx in range(0, len(peaks_idx)-1):
                if (peaks_idx[coord_idx+1] - peaks_idx[coord_idx]) < active_region_idx:
                    active_region.append(peaks_idx[coord_idx])
                    active_region.append(peaks_idx[coord_idx+1])
                else:
                    if active_region:
                        active_regions_idx.append([active_region[0], active_region[-1]])
                    else:
                        active_regions_idx.append([peaks_idx[coord_idx]])
                    active_region = []
            if active_region:
                active_regions_idx.append([active_region[0], active_region[-1]])
            else:
                active_regions_idx.append([peaks_idx[-1]])
        else:
            active_regions_idx.append([peaks_idx[0]])

    # Collate features and plot
    peakPowersRanked = np.sort(spectrum[peaks_idx])[::-1]
    meanPeakNPowers = []
    varPeakNPowers = []
    peakPowersCoordsRanked = np.sort(coordinates[peaks_idx])[::-1]
    for i in range(1, len(peakPowersRanked)+1):
        meanPeakNPowers.append(np.mean(peakPowersRanked[:i]))
        varPeakNPowers.append(np.var(peakPowersRanked[:i]))
    active_region_powers = []
    active_region_coords = []
    active_region_coordSeparations = []
    active_region_coordWidths = []
    prominences = np.array(list(peaks_properties["prominences"]))
    plt.plot(coordinates, spectrum, label="spectrum")
    plt.scatter(coordinates[peaks_idx], spectrum[peaks_idx], color="red", marker="o", label="peak")
    activeTotalCoords = 0.0
    activeTotalPower = 0.0
    if active_regions_idx:
        for region_idx in active_regions_idx:    
            region_peaks_idx = np.intersect1d(range(region_idx[0], region_idx[-1]+1), peaks_idx) if len(region_idx) > 1 else region_idx
            active_region_powers.append(np.array(spectrum[region_peaks_idx]))
            region_peak_coords = np.array(coordinates[region_peaks_idx])
            active_region_coords.append(region_peak_coords)
            
            if len(region_idx) > 1:
                separations = []
                for i in range(1, len(region_peak_coords)):
                    separations.append(region_peak_coords[i] - region_peak_coords[i-1])
                active_region_coordSeparations.append(np.mean(separations))
                active_region_coordWidths.append(coordinates[region_idx[-1]] - coordinates[region_idx[0]]) 
                activeTotalCoords += coordinates[region_idx[1]] - coordinates[region_idx[0]]
                activeTotalPower += np.sum(spectrum[region_idx[0]:region_idx[1]+1])
                plt.axvspan(coordinates[region_idx[0]], coordinates[region_idx[1]], color='orange', alpha=0.3, label="active region")
            else:
                active_region_coordSeparations.append(coordinates[1] - coordinates[0])
                active_region_coordWidths.append(coordinates[1] - coordinates[0]) # Append sampling width of one coordinate
                activeTotalCoords += coordinates[region_idx[0]]
                activeTotalPower += spectrum[region_idx[0]]
                plt.axvline(coordinates[region_idx[0]], color='orange', alpha=0.3, label="active region")
    
    activeTotalCoords /= np.sum(coordinates)
    activeTotalPower /= np.sum(spectrum)
    active_region_coordSeparations = np.array(active_region_coordSeparations)

    # Set legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.ylabel("Power")
    plt.title(f"run {spectrum_name}")
    if xLabel:
        plt.xlabel(xLabel)
    if savePath:
        plt.savefig(savePath / f"run_{spectrum_name}_feature_extraction.png")
    #plt.show()
    plt.close("all")
         
    if len(peaks_idx) > 0:
        features = SpectralFeatures1D(
            peaksFound = True,
            peakPowers = spectrum[peaks_idx],
            peakCoordinates = coordinates[peaks_idx],
            numPeaks = len(peaks_idx),
            maxPeakPower = float(peakPowersRanked[0]),
            maxPeakCoordinate = float(peakPowersCoordsRanked[0]),
            meanPeakNPowers = np.array(meanPeakNPowers),
            meanPeak4Powers = float(meanPeakNPowers[3]) if len(meanPeakNPowers) > 3 else 0.0,
            meanPeak3Powers = float(meanPeakNPowers[2]) if len(meanPeakNPowers) > 2 else 0.0,
            meanPeak2Powers = float(meanPeakNPowers[1]) if len(meanPeakNPowers) > 1 else 0.0,
            varPeakNPowers = np.array(varPeakNPowers),
            varPeak4Powers = float(varPeakNPowers[3]) if len(varPeakNPowers) > 3 else 0.0,
            varPeak3Powers = float(varPeakNPowers[2]) if len(varPeakNPowers) > 2 else 0.0,
            varPeak2Powers = float(varPeakNPowers[1]) if len(varPeakNPowers) > 1 else 0.0,
            peakProminences = prominences,
            maxPeakProminence = prominences.max(),
            meanPeakProminence = np.mean(prominences),
            varPeakProminence = np.var(prominences),
            maxPeakPowerProminence = prominences[np.argmax(spectrum[peaks_idx])],
            activeRegions = active_regions_idx,
            numActiveRegions = len(active_regions_idx),
            activeRegionMeanPowers = np.array([np.mean(r) for r in active_region_powers]),
            activeRegionVarPowers = np.array([np.var(r) for r in active_region_powers]),
            activeRegionMeanCoordinates = np.array([np.mean(r) for r in active_region_coords]),
            activeRegionPeakSeparations = active_region_coordSeparations,
            activeRegionMeanPeakSeparations = float(np.mean(active_region_coordSeparations[active_region_coordSeparations != 0.0])),
            activeRegionVarPeakSeparations = float(np.var(active_region_coordSeparations[active_region_coordSeparations != 0.0])),
            activeRegionCoordWidths = np.array(active_region_coordWidths),
            activeRegionMeanCoordWidths = float(np.mean(active_region_coordWidths)),
            activeRegionVarCoordWidths = float(np.var(active_region_coordWidths)),
            totalActiveCoordinateProportion = float(activeTotalCoords),
            totalActivePowerProportion = float(activeTotalPower),
            spectrumSum = float(np.sum(spectrum)),
            spectrumMean = float(np.mean(spectrum)),
            spectrumVar = float(np.var(spectrum))
        )
    else:
        features = SpectralFeatures1D(
            peaksFound = False, 
            spectrumSum = np.sum(spectrum),
            spectrumMean = np.mean(spectrum),
            spectrumVar = np.var(spectrum)
        )

    return features