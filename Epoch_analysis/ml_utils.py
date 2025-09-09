from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import itertools
import csv

from scipy.stats import norm
from scipy.interpolate import griddata
from scipy.signal import find_peaks, resample
import xarray as xr
from inference.plotting import matrix_plot
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
import json
import dataclasses as dc
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from numpy.typing import ArrayLike
from SALib import ProblemSpec
import SALib.sample as salsamp

# Sktime algorithms
import sktime.clustering.dbscan as skt_dbscan
import sktime.clustering.k_means  as skt_kmeans
import sktime.clustering.k_medoids  as skt_kmedoids
import sktime.clustering.k_shapes  as skt_kshapes
import sktime.clustering.kernel_k_means as skt_kernelmeans
import sktime.regression.deep_learning as skt_deep
import sktime.regression.distance_based as skt_dist
import sktime.regression.interval_based as skt_int
import sktime.regression.kernel_based as skt_kernel
import sktime.regression.compose as skt_compose

# Aeon algorithms
import aeon.regression.interval_based as aeon_int
import aeon.regression.distance_based as aeon_dist
import aeon.regression.feature_based as aeon_feature
import aeon.regression.shapelet_based as aeon_shapelet
import aeon.regression.convolution_based as aeon_conv
import aeon.regression.hybrid as aeon_hybrid
import aeon.regression.deep_learning as aeon_deep
import aeon.regression as aeon_reg

@dataclass
class GPModel:
    kernelName : str
    inputNames : list
    normalisedInputs : np.ndarray
    outputName : str
    output : np.ndarray
    regressionModel : GaussianProcessRegressor = None
    classificationModel : GaussianProcessClassifier = None
    modelParams : dict = None
    fitSuccess : bool = None

@dataclass_json
@dataclass
class GPResults:
    directory : str = None
    kernelNames : ArrayLike = None
    inputNames : ArrayLike = None
    outputSpectrumName : str = None
    outputNames : ArrayLike = None
    logFields : ArrayLike = None
    normalised : bool = False
    original_output_means : dict = None
    original_output_stdevs : dict = None
    original_input_means : dict = None
    original_input_stdevs : dict = None
    numObservations : int = 0
    numFeatures : int = 0
    numOutputs : int = 0
    wRank : int = 0
    fixedParams : dict = None
    sobolSamples : int = 0
    sobolIndicesO1 : dict = None
    sobolIndicesO2 : dict = None
    sobolIndicesTotal : dict = None
    cvStrategy : str = None
    cvFolds : int = 0
    cvRepeats : int = 0
    cvR2_mean : float = 0.0
    cvR2_var : float = 0.0
    cvR2_stderr : float = 0.0
    cvRMSE_mean : float = 0.0
    cvRMSE_var : float = 0.0
    cvRMSE_stderr : float = 0.0
    cvMSLL_mean : float = 0.0
    cvMSLL_var : float = 0.0
    cvMSLL_stderr : float = 0.0
    gpPackage : str = None
    model : dict = None
    fitSuccess : bool = False

@dataclass
class TSRBattery:
    directory : str = None
    package : str = None
    inputSpectra : ArrayLike = None
    outputFields : ArrayLike = None
    logFields : ArrayLike = None
    algorithms : ArrayLike = None
    normalised : bool = False
    numObservations : int = 0
    numOutputs : int = 0
    numInputDimensions : int = 0
    numTimepointsIfEqual : int = 0
    equalLengthTimeseries = True
    multivariate : bool = False
    original_output_means : dict = None
    original_output_stdevs : dict = None
    cvStrategy : str = None
    cvFolds : int = 0
    cvRepeats : int = 0
    results : ArrayLike = None
    
@dataclass
class TSRResult:
    algorithm : str = None
    output : str = None
    cvR2_mean : float = 0.0
    cvR2_var : float = 0.0
    cvR2_stderr : float = 0.0
    cvRMSE_mean : float = 0.0
    cvRMSE_var : float = 0.0
    cvRMSE_stderr : float = 0.0

@dataclass
class SimulationMetadata:
    simId : int
    backgroundDensity : float
    beamFraction : float
    B0 : float
    B0angle : float

@dataclass
class SpectralFeatures1D:
    simID : str = None
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
        case "sktime.TimeSeriesDBSCAN":
            return skt_dbscan.TimeSeriesDBSCAN(kwargs["distance"])
        case "sktime.TimeSeriesKMeans":
            return skt_kmeans.TimeSeriesKMeans(kwargs["n_clusters"])
        case "sktime.TimeSeriesKMeansTslearn":
            return skt_kmeans.TimeSeriesKMeansTslearn(kwargs["n_clusters"])
        case "sktime.TimeSeriesKMedoids": 
            return skt_kmedoids.TimeSeriesKMedoids(kwargs["n_clusters"])
        case "sktime.TimeSeriesKShapes":
            return skt_kshapes.TimeSeriesKShapes(kwargs["n_clusters"])
        case "sktime.TimeSeriesKernelKMeans":
            return skt_kernelmeans.TimeSeriesKernelKMeans(kwargs["n_clusters"])
        case "sktime.CNNRegressor":
            return skt_deep.CNNRegressor(kwargs)
        case "sktime.CNTCRegressor": 
            return skt_deep.CNTCRegressor(kwargs)
        case "sktime.FCNRegressor":
            return skt_deep.FCNRegressor(kwargs)
        case "sktime.InceptionTimeRegressor":
            return skt_deep.InceptionTimeRegressor(kwargs)
        case "sktime.KNeighborsTimeSeriesRegressor":
            return skt_dist.KNeighborsTimeSeriesRegressor(n_neighbors=kwargs["n_neighbours"], distance=kwargs["distance"])
        case "sktime.LSTMFCNRegressor":
            return skt_deep.LSTMFCNRegressor(kwargs)
        case "sktime.MACNNRegressor":
            return skt_deep.MACNNRegressor(kwargs)
        case "sktime.MCDCNNRegressor": 
            return skt_deep.MCDCNNRegressor(kwargs)
        case "sktime.MLPRegressor":
            return skt_deep.MLPRegressor(kwargs)
        case "sktime.ResNetRegressor": 
            return skt_deep.ResNetRegressor(kwargs)
        case "sktime.RocketRegressor":
            return skt_kernel.RocketRegressor(kwargs)
        case "sktime.SimpleRNNRegressor":
            return skt_deep.SimpleRNNRegressor(kwargs)
        case "sktime.TapNetRegressor":
            return skt_deep.TapNetRegressor(kwargs)
        case "sktime.TimeSeriesSVRTslearn":
            return skt_kernel.TimeSeriesSVRTslearn(kwargs)
        case "sktime.TimeSeriesForestRegressor":
            return skt_int.TimeSeriesForestRegressor(kwargs)
        case "sktime.ComposableTimeSeriesForestRegressor":
            return skt_compose.ComposableTimeSeriesForestRegressor(kwargs)
    # AEON
    # Unequal length data series
        case "aeon.Catch22Regressor":
            return aeon_feature.Catch22Regressor()
        case "aeon.DummyRegressor":
            return aeon_reg.DummyRegressor()
        case "aeon.KNeighborsTimeSeriesRegressor":
            return aeon_dist.KNeighborsTimeSeriesRegressor()
        case "aeon.RDSTRegressor":
            return aeon_shapelet.RDSTRegressor()
    # Multivariate
        case "aeon.CanonicalIntervalForestRegressor":
            return aeon_int.CanonicalIntervalForestRegressor()
        case "aeon.DrCIFRegressor":
            return aeon_int.DrCIFRegressor()
        case "aeon.FCNRegressor":
            return aeon_deep.FCNRegressor()
        case "aeon.FreshPRINCERegressor":
            return aeon_feature.FreshPRINCERegressor()
        case "aeon.HydraRegressor":
            return aeon_conv.HydraRegressor()
        case "aeon.InceptionTimeRegressor":
            return aeon_deep.InceptionTimeRegressor()
        case "aeon.IndividualInceptionRegressor":
            return aeon_deep.IndividualInceptionRegressor()
        case "aeon.IndividualLITERegressor":
            return aeon_deep.IndividualLITERegressor()
        case "aeon.IntervalForestRegressor":
            return aeon_int.IntervalForestRegressor()
        case "aeon.LITETimeRegressor":
            return aeon_deep.LITETimeRegressor()
        case "aeon.MLPRegressor":
            return aeon_deep.MLPRegressor()
        case "aeon.MiniRocketRegressor":
            return aeon_conv.MiniRocketRegressor()
        case "aeon.MultiRocketHydraRegressor":
            return aeon_conv.MultiRocketHydraRegressor()
        case "aeon.MultiRocketRegressor":
            return aeon_conv.MultiRocketRegressor()
        case "aeon.QUANTRegressor":
            return aeon_int.QUANTRegressor()
        case "aeon.RISTRegressor":
            return aeon_hybrid.RISTRegressor()
        case "aeon.RandomIntervalRegressor":
            return aeon_int.RandomIntervalRegressor()
        case "aeon.RandomIntervalSpectralEnsembleRegressor":
            return aeon_int.RandomIntervalSpectralEnsembleRegressor()
        case "aeon.ResNetRegressor":
            return aeon_deep.ResNetRegressor()
        case "aeon.RocketRegressor":
            return aeon_conv.RocketRegressor()
        case "aeon.SummaryRegressor":
            return aeon_feature.SummaryRegressor()
        case "aeon.TSFreshRegressor":
            return aeon_feature.TSFreshRegressor()
        case "aeon.TimeCNNRegressor":
            return aeon_deep.TimeCNNRegressor()
        case "aeon.TimeSeriesForestRegressor":
            return aeon_int.TimeSeriesForestRegressor()
    # Univariate
        case "aeon.DisjointCNNRegressor":
            return aeon_deep.DisjointCNNRegressor()

def read_data(dataFiles, data_dict : dict, with_names : bool = False, with_coords : bool = False) -> dict:
    
    dataFields = list(data_dict.keys())
    sim_ids = []

    if with_coords:
        for fieldPath in dataFields:
            dimCoordsKey = fieldPath + "_coords"
            data_dict[dimCoordsKey] = []

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
                data_dict[fieldPath].append(data[group].variables[fieldName].values.astype('float64'))
                # Get coordinates if requested
                if with_coords:
                    dimCoordsKey = fieldPath + "_coords"
                    dimName = data[group].variables[fieldName].dims[0]
                    dimVals = data[group].coords[dimName].values
                    data_dict[dimCoordsKey].append(dimVals)

        sim_ids.append(simulation.split("/")[-1].split("_")[1])
        
    # Sort by sim ID to reduce confusion later
    # sorted_idx = list(np.array([int(id) for id in sim_ids]).argsort())
    try:
        int_ids = [int(id) for id in sim_ids]
        sorted_idx = np.array(int_ids).argsort()
    except Exception:
        sorted_idx = np.array(sim_ids).argsort()

    for field, vals in data_dict.items():
        data_dict[field] = [vals[i] for i in sorted_idx]

    if with_names:
        data_dict["sim_ids"] = [sim_ids[i] for i in sorted_idx]

    return data_dict

def downsample_series(series : ArrayLike, coords : ArrayLike, numSamples : int, signalName : str = None, savedir : Path = None):

    print(f"Resampling signal from {len(series)} to {numSamples}....")

    resampled_signal = resample(series, numSamples)
    res_coords = np.linspace(coords[0], coords[-1], len(resampled_signal))
    print(f"Resampled signal length: {len(resampled_signal)}")

    if savedir is not None:
        plt.plot(coords, series, label = "original")
        plt.plot(res_coords, resampled_signal, label = "resampled")
        plt.legend()
        plt.title(signalName)
        plt.savefig(savedir / f"{signalName}.png")
        plt.clf()

    return resampled_signal, res_coords


def truncate_series(series : ArrayLike, seriesCoordinates : ArrayLike, maxCoordinate : float):

    print(f"Truncting signal from maximum coordinate of {seriesCoordinates[-1]} to first {maxCoordinate}....")

    cutoffIndex = np.searchsorted(seriesCoordinates, maxCoordinate)
    truncatedSeries = series[:cutoffIndex]
    truncatedCoords = seriesCoordinates[:cutoffIndex]

    return truncatedSeries, truncatedCoords

# Normalise a 1D signal
def normalise_1D(data: ArrayLike, doLog : bool = False):
    if isinstance(data, list):
        data = np.array(data)
    orig_mean = np.mean(data)
    orig_sd = np.std(data)
    if doLog:
        data = np.log10(data)
        mean = np.mean(data)
        sd = np.std(data)
    else:
        mean = orig_mean
        sd = orig_sd
    return (data - mean) / sd, orig_mean, orig_sd

# Normalises data row-wise (mean and sd taken row-wise)
def normalise_dataset(data: ArrayLike):
    orig_mean = [np.nanmean(row) for row in data]
    orig_sd = [np.nanstd(row) for row in data]

    normed_data = []
    for row_id in range(data.shape[0]):
        normed_data.append((data[row_id] - orig_mean[row_id]) / orig_sd[row_id])

    return np.array(normed_data), orig_mean, orig_sd

# De-normalises data row-wise (mean and sd taken row-wise)
def denormalise_dataset(data: ArrayLike, orig_mean : float, orig_sd : float, unlog : bool):
    denormed_data = []
    
    for row_id in range(data.shape[0]):
        denormed_data.append(denormalise_datapoint(data[row_id], orig_mean[row_id], orig_sd[row_id], unlog))
    
    return denormed_data

# Denormalises a single data point
def denormalise_datapoint(datapoint : float, orig_mean : float, orig_sd : float, unlog : bool):
    denormed = (datapoint * orig_sd) + orig_mean
    if unlog:
        denormed = 10**denormed
    return  denormed

# Denormalises an RMSE value
def denormalise_rmse(rmseValue : float, orig_sd : float):
    denormed = rmseValue * orig_sd
    return  denormed

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

def __encode_ML_result_to_JSON_dict(result : GPResults | TSRBattery) -> str:
    results_dict = dc.asdict(result)
    for name, val in results_dict.items():
        if isinstance(val, np.ndarray):
            results_dict[name] = val.tolist()
        elif isinstance(val, dict):
            for name2, val2 in val.items():
                if isinstance(val2, np.ndarray):
                    val[name2] = val2.tolist()
    return results_dict

def write_ML_result_to_file(result : GPResults | TSRBattery, filepath : Path):
    with open(filepath, "w", encoding='utf-8') as f:
        result_json = __encode_ML_result_to_JSON_dict(result)
        json.dump(result_json, f, ensure_ascii=False, indent=4)

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
    _ = plt.figure(figsize=(10, 8))
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
        peak_prominence_factor : float = 0.1, 
        peak_distance_coordUnits : float = 0.5, 
        excited_region_distance_coordUnits : float = 3.0, 
        spectrum_name : list = None, 
        savePath : Path = None,
        xLabel : str = None,
        yLabel : str = None,
        xUnit : str = None,
        yUnit : str = None) -> dict:

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
        plt.xlabel(f"{xLabel} [{xUnit}]")
    if yLabel:
        plt.ylabel(f"{yLabel} [{yUnit}]")
    if savePath:
        plt.savefig(savePath / f"run_{spectrum_name}_feature_extraction.png")
    #plt.show()
    plt.close("all")
         
    if len(peaks_idx) > 0:
        features = SpectralFeatures1D(
            simID = spectrum_name,
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
            simID = spectrum_name,
            peaksFound = False, 
            spectrumSum = np.sum(spectrum),
            spectrumMean = np.mean(spectrum),
            spectrumVar = np.var(spectrum)
        )

    return features

# Calculate NLL for a single data point
def negative_log_likelihood(prediction : float | np.ndarray, true_value : float | np.ndarray, variance : float | np.ndarray):
    if isinstance(true_value, np.ndarray):
        assert prediction.shape == true_value.shape
        assert prediction.shape == variance.shape
        return 0.5 * (np.add(np.log(2.0 * np.pi * variance), (np.subtract(true_value, prediction)**2/variance)))
    else:
        return 0.5 * (np.log(2.0 * np.pi * variance) + ((true_value - prediction)**2/variance))

# Calculate standardized log loss
def standardized_log_loss(
        prediction : float | np.ndarray, 
        prediction_variance : float | np.ndarray,
        true_value : float | np.ndarray,
        training_mean : float,
        training_variance : float):
    
    nll = negative_log_likelihood(prediction, true_value, prediction_variance)
    if training_mean.size == 1:
        training_mean = float(training_mean)
        training_variance = float(training_variance)
    nll_baseline = negative_log_likelihood(prediction, training_mean, training_variance)
    msll = nll - nll_baseline

    return msll

# Calculate squared error
def squared_error(prediction : float, true_value : float) -> float:
    return (prediction - true_value)**2

def mean_squared_error(predictions : ArrayLike, true_values : ArrayLike) -> tuple:
    assert len(predictions) == len(true_values)
    n = len(predictions)
    preds = np.array(predictions)
    truth = np.array(true_values)
    ses = squared_error(preds, truth)
    ses_var = np.var(ses)
    ses_stdErr = np.std(ses)/np.sqrt(len(ses))
    return np.sum(ses)/n, ses_var, ses_stdErr

def root_mean_squared_error(predictions : ArrayLike, true_values : ArrayLike) -> tuple:
    mses, var, stdErr = mean_squared_error(predictions, true_values)
    return np.sqrt(mses), var, stdErr