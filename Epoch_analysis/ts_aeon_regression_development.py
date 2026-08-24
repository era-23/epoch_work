import argparse
import glob
import os
from pathlib import Path
import warnings

from sklearn.preprocessing import MinMaxScaler
import ml_utils
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import time
from scipy.stats import sem
from sklearn.model_selection import RepeatedKFold, LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, log_loss

from aeon.datasets import load_cardano_sentiment, load_covid_3month
from aeon.transformations.collection import Normalizer

from sklearn.metrics import root_mean_squared_error, r2_score
from scipy.stats import linregress
import logging

import epoch_utils

import xarray as xr

from dataclass_csv import DataclassWriter

# 0 = All logs (default)
# 1 = Filter out INFO logs
# 2 = Filter out INFO and WARNING logs
# 3 = Filter out INFO, WARNING, and ERROR logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Optional: Silence the oneDNN message explicitly
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# plt.rcParams.update({'axes.titlesize': 32.0})
# plt.rcParams.update({'axes.labelsize': 36.0})
# plt.rcParams.update({'xtick.labelsize': 28.0})
# plt.rcParams.update({'ytick.labelsize': 28.0})
# plt.rcParams.update({'legend.fontsize': 24.0})

def regress_from_hd5(
        directory : Path,
        outputFields : list,
        logFields : list,
        algorithms : list,
        logInputs : bool = False,
        resultsFilepath : Path = None,
        doPlot : bool = True,
        noTitle : bool = False,
        nThreads : int = 1,
        numFolds = 10,
        numRepeats = 1
):
    # Initialise results objects
    battery = ml_utils.TSRBattery()
    allPredictionsRecord = []
    battery.package = "Aeon"

    data_dir = directory / "data"
    data_files = glob.glob(str(data_dir / "*.h5")) 
    battery.directory = str(directory.resolve())
    battery.algorithms = algorithms
    battery.normalised = True
    battery.scaledInputs = False

    # JWSC h5 data schema:
    # spectra/F == frequencies
    # spectra/Y == growth rates
    # parameters/X == input values

    outputFields = ["B0", "log(density)", "log(fi_conc)", "pitch", "background_temp"]
    inputData = []
    aeon_only = np.all([str.startswith(a, "aeon") for a in algorithms])

    # Input data
    for f in data_files:
        data : xr.DataTree = xr.open_datatree(f)

        # for n in range(data.num_samples):
        #     plt.subplots(figsize = (10,6))
        #     plt.plot(data["spectra"].F, data["spectra"].Y[:,n])
        #     plt.title(", ".join([f"{f}: {v:.3f}" for f, v in zip(inputFields, data["parameters"].X[:,n].data)]))
        #     plt.vlines(epoch_utils.calculate_gyrofrequencies_in_Hz(data["parameters"].X[0,n].data, max_freq=float(data["spectra"].F.max())), ymin=plt.ylim()[0], ymax=plt.ylim()[1], colors="tab:orange", linestyles="dashed", alpha=0.8)
        #     plt.grid()
        #     plt.show()

        inputData.append(data)
        data.close()

    assert np.allclose(inputData[0]["spectra"].F.data, inputData[1]["spectra"].F.data)
    # inputFreqs = np.array(inputData[0]["spectra"].F)
    inputParams = np.concat((np.array(inputData[0]["parameters"].X.data), np.array(inputData[1]["parameters"].X.data)), axis=1)
    inputSpectra = np.concat((np.array(inputData[0]["spectra"].Y.data), np.array(inputData[1]["spectra"].Y.data)), axis=1).T

    targetFields = {k : v for k, v in zip(outputFields, inputParams)}

    if logInputs:
        for i in range(inputSpectra.shape[0]):
            log_spec = epoch_utils.zero_negative_safe_log(inputSpectra[i])
            inputSpectra[i] = log_spec

    # Reshape into 3D numpy array of shape (n_cases, n_channels, n_timepoints)
    if aeon_only:
        inputSpectra = np.expand_dims(inputSpectra, axis = 1)

    # Output data
    battery.outputFields = np.array(outputFields)
    battery.numOutputs = len(outputFields)
    battery.logFields = np.array(logFields)
    battery.original_output_means = dict.fromkeys(targetFields)
    battery.original_output_stdevs = dict.fromkeys(targetFields)

    battery.equalLengthTimeseries = True
    battery.numObservations = inputSpectra.shape[0]
    battery.numInputDimensions = 1
    battery.numTimepointsIfEqual = inputSpectra.shape[-1]
    battery.multivariate = battery.numInputDimensions > 1

    battery.cvStrategy = "RepeatedKFolds"
    battery.cvFolds = numFolds
    battery.cvRepeats = numRepeats
    battery.results = []

    # Dumb hack
    # best_results = {"backgroundDensity" : 0.452861, "beamFraction" : 0.313241, "B0strength" : 0.024969, "pitch" : 0.587287}

    for output_field, output_values in targetFields.items():
        
        assert len(output_values) == inputSpectra.shape[0]
        case_indices = np.arange(len(output_values))
        output_values = np.array(output_values)
        cv = RepeatedKFold(n_splits=numFolds, n_repeats=numRepeats)
        tt_split = list(enumerate(cv.split(case_indices)))

        # Record denormalisation parameters
        _, scaler = ml_utils.normalise_data(output_values)
        print(f"Original data mean: {np.mean(output_values)}, original data SD: {np.std(output_values)}")
        print(f"Mean (0.0) in normalised RMSE units is {scaler.mean_} in original {output_field} units (or {10**scaler.mean_} in log space).")
        print(f"SD in normalised RMSE units is {np.sqrt(scaler.var_)} in original {output_field} units (or {10**np.sqrt(scaler.var_)} in log space).")
        # print(f"Best RMSE = {best_results[output_field]}, which denormalises to {scaler.inverse_transform([[best_results[output_field]]])[0][0] - scaler.mean_[0]}, or {10**scaler.inverse_transform([[best_results[output_field]]])[0][0]} in log space.")
        # print(f"ALT LOG: Best RMSE high = {10**(scaler.mean_[0] + (np.sqrt(scaler.var_) * best_results[output_field]))}")
        # print(f"ALT LOG: Best RMSE low = {10**(scaler.mean_[0] - (np.sqrt(scaler.var_) * best_results[output_field]))}")
        battery.original_output_means[output_field] = scaler.mean_
        battery.original_output_stdevs[output_field] = np.sqrt(scaler.var_)

        for algorithm in algorithms:

            print(f"Building {algorithm} model for {output_field} from James' linear data....")
            
            # Results
            result = ml_utils.TSRResult()
            result.output = output_field
            result.algorithm = algorithm
            
            tsr = ml_utils.get_algorithm(algorithm, nThreads)

            # CV Folds
            all_test_indices = []
            all_train_R2s = []
            all_test_R2s = []
            all_test_points = []
            all_predictions = []
            all_test_points_denormed = []
            all_predictions_denormed = []

            for fold, (train, test) in tt_split:
                fold_time_start = time.process_time_ns()
                print(f"Fold: {fold}....")

                train_x = [inputSpectra[t] for t in train]
                train_y = output_values[train]
                test_x = [inputSpectra[t] for t in test]
                test_y = output_values[test]

                # Renormalise for each split
                train_y, scaler = ml_utils.normalise_data(train_y)
                test_y, _ = ml_utils.normalise_data(test_y, scaler = scaler)
                print(f"scaler mean: {scaler.mean_}")
                print(f"1.0 in normalised RMSE units is {scaler.inverse_transform([[1.0]])} in original {output_field} units (may be logged).")

                print("    Training model....")
                # Fit
                tsr.fit(train_x, train_y)

                # Predict
                predictions = tsr.predict(test_x)
                preds_denormed = ml_utils.denormalise_data(predictions, scaler)
                test_y_denormed = ml_utils.denormalise_data(test_y, scaler)
                if output_field in logFields:
                    preds_denormed = 10.0**preds_denormed
                    test_y_denormed = 10.0**test_y_denormed
                if len(predictions) < 10:
                    for i in range(len(predictions)):
                        print(f"    Prediction:  {preds_denormed[i]}, Ground truth: {test_y_denormed[i]} (normalised)")
                        # print(f"    Ground truth: {test_y} (normalised), {test_y_denormed} (original)")
                else:
                    print("Fold test size > 10, skipping printing of predictions.")
                if aeon_only:
                    score = tsr.score(test_x, test_y, metric='r2')
                    training_r2 = tsr.score(train_x, train_y, metric='r2')
                else:
                    score = tsr.score(test_x, test_y)
                    training_r2 = tsr.score(train_x, train_y)
                all_test_R2s.append(score)
                skl_rmse = root_mean_squared_error(test_y, predictions)
                all_train_R2s.append(training_r2)
                print("-------- RESULTS ---------")
                print(f"    test r2:      {score}")
                print(f"    training r2:  {training_r2}")
                print(f"    sklearn rmse: {skl_rmse} (actuals S.D.: {np.std(test_y)})")

                all_test_indices.extend(test.tolist())
                all_test_points.extend(test_y.tolist())
                all_test_points_denormed.extend(test_y_denormed)
                all_predictions.extend(list(predictions))
                all_predictions_denormed.extend(preds_denormed)

                # Log predictions
                testLens = [len(predictions), len(test), len(test_y), len(test_y_denormed), len(predictions), len(preds_denormed)]
                assert len(set(testLens)) == 1 # All lists have equal length
                for i in range(len(predictions)):
                    predRecord = ml_utils.TSRPrediction(
                        algorithm=algorithm,
                        inputChannels=["growth_rates"],
                        outputQuantity=output_field,
                        datapoint_ID=test[i],
                        fold_ID=fold,
                        trueValue_normalised=test_y[i],
                        trueValue_denormalised=test_y_denormed[i][0],
                        trueValue_denormalised_log10=np.log10(test_y_denormed[i][0]),
                        predictedValue_normalised=predictions[i],
                        predictedValue_denormalised=preds_denormed[i][0],
                        predictedValue_denormalised_log10=np.log10(preds_denormed[i][0])
                    )
                    allPredictionsRecord.append(predRecord)

                fold_time_end = time.process_time_ns()
                fold_time = fold_time_end - fold_time_start
                fold_time_s = fold_time / 1e9
                print(f"Fold time: {fold_time_s}s.")

            rmse, rmse_var, rmse_se = ml_utils.root_mean_squared_error(all_predictions, all_test_points)
            r2 = np.mean(all_test_R2s)
            r2_sem = sem(all_test_R2s)
            r2_var = np.var(all_test_R2s)

            if math.isnan(r2):
                # Recalculate based on r2 over folds (primarily for LOOCV)
                r2 = r2_score(all_test_points, all_predictions)

            mean_training_r2 = np.mean(all_train_R2s)
            train_r2_sem = sem(all_train_R2s)
            train_r2_var = np.var(all_train_R2s)
            
            summary_str = f"{output_field} -- {algorithm}: Mean test r2 = {r2:.5f}+-{r2_sem:.5f}, mean test RMSE = {rmse:.5f}+-{rmse_se:.5f}, mean train r2 = {mean_training_r2}+-{train_r2_sem}"
            print("--------------------------------------------------------------------------------------------------------------------------")
            print(summary_str)
            logger.info(summary_str)
            print("--------------------------------------------------------------------------------------------------------------------------")
            result.cvR2_mean = r2
            result.cvR2_var = r2_var
            result.cvR2_stderr = r2_sem
            result.cvRMSE_mean = rmse
            result.cvRMSE_var = rmse_var
            result.cvRMSE_stderr = rmse_se
            result.cvMAE_mean = mean_absolute_error(y_true=all_test_points, y_pred=all_predictions, multioutput="uniform_average")
            mae_all = mean_absolute_error(y_true=all_test_points, y_pred=all_predictions, multioutput="raw_values")
            result.cvMAE_var = np.var(mae_all)
            result.cvMAE_stderr = sem(mae_all)
            result.cvMAPE_mean = mean_absolute_percentage_error(y_true=all_test_points, y_pred=all_predictions, multioutput="uniform_average")
            mape_all = mean_absolute_percentage_error(y_true=all_test_points, y_pred=all_predictions, multioutput="raw_values")
            result.cvMAPE_var = np.var(mape_all)
            result.cvMAPE_stderr = sem(mape_all)
            result.trainR2_mean = mean_training_r2
            result.trainR2_stderr = train_r2_sem
            result.trainR2_var = train_r2_var

            battery.results.append(result)

            if doPlot:
                plot_predictions(
                    algorithm_name = algorithm,
                    field = output_field,
                    sim_ids = all_test_indices, 
                    truth = all_test_points_denormed,
                    preds = all_predictions_denormed,
                    r2 = result.cvR2_mean,
                    rmse = result.cvRMSE_mean,
                    saveFolder = resultsFilepath.parent / "predictions" / algorithm, 
                    doLog = output_field in logFields,
                    noTitle = noTitle
                )

    # Write results and all predictions
    ml_utils.write_ML_result_to_file(battery, resultsFilepath)
    if len(allPredictionsRecord) > 0:
        prds_path = resultsFilepath.parent / "predictions"
        prds_path.mkdir(parents = True, exist_ok=True)
        with open(prds_path / f"{resultsFilepath.name.replace('.json', '').replace('.', '')}_predictions.csv", "w") as f:
            w = DataclassWriter(f, allPredictionsRecord, ml_utils.TSRPrediction)
            w.write()

def demo():
    covid_train, covid_train_y = load_covid_3month(split="train")
    covid_test, covid_test_y = load_covid_3month(split="test")
    cardano_train, cardano_train_y = load_cardano_sentiment(split="train")
    cardano_test, cardano_test_y = load_cardano_sentiment(split="test")
    print(f"Covid spectrum shape:   {covid_train.shape} (n_cases: {covid_train.shape[0]}, n_channels: {covid_train.shape[1]}, n_timepoints: {covid_train.shape[2]})")
    print(f"Covid output shape:     {covid_train_y.shape}")
    print(f"Cardano spectrum shape: {cardano_train.shape} (n_cases: {cardano_train.shape[0]}, n_channels: {cardano_train.shape[1]}, n_timepoints: {cardano_train.shape[2]})")
    print(f"Cardano output shape:   {cardano_train_y.shape}")

def correlate_predictions_with_iciness(
        algorithm_name : str,
        field : str,
        truth : list,
        preds : list,
        iceMetrics : dict,
        r2 : float,
        rmse : float,
        saveFolder : Path
):
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    squared_errors = (np.array([float(p) for p in preds]) - np.array([float(t) for t in truth]))**2
    
    correlations = []
    plotTitle = f"{field} -- {algorithm_name} (prediction {r'$r^2$'} = {r2:.3f}, {r'rmse'} = {rmse:.3f})"
    unique_name = f"{algorithm_name}_{field}"
    correlations = epoch_utils.correlate_and_plot_iciness_vs_baseline(
        iceMetrics, 
        "squared prediction error", 
        squared_errors.tolist(), 
        saveFolder, 
        unique_name,
        correlations, 
        plotTitle, 
        doPlot = True
    )

    with open(saveFolder / f"{unique_name}_iciness_correlations.csv", 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, correlations[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(correlations)

    metric_correlations = []
    print(f"{algorithm_name} METRIC CORRELATIONS: ")
    for metric in iceMetrics.keys():
        metric_parts = metric.split('_ICEmetric_')
        metric_short = f"ICEmetric_{metric_parts[1]}" if len(metric_parts) > 1 else metric_parts[0]
        data = [c["r2"] for c in correlations if c["metric"] == metric_short]
        mean = np.mean(data)
        meanAbs = np.mean(np.abs(data))
        print(f"{metric}:   | raw r2 = {mean:.5f}, abs r2 = {meanAbs:.5f}")
        metric_correlations.append({"metric" : metric, "mean_r2" : mean, "mean_abs_r2" : meanAbs})

    with open(saveFolder / f"{unique_name}_mean_iciness_correlations_by_metric.csv", 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, metric_correlations[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(metric_correlations)

def plot_predictions(
        algorithm_name : str,
        field : str,
        sim_ids : list,
        truth : list,
        preds : list,
        r2 : float,
        rmse : float,
        saveFolder : Path,
        doLog : bool,
        noTitle : bool = True
):
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    unit = epoch_utils.fieldNameToUnit_dict[field]

    plt.subplots(figsize=(12, 8))
    plt.scatter(truth, sim_ids, label = "True value", marker = "o", color = "blue")
    plt.scatter(preds, sim_ids, label = "Predicted value", marker = "o", color = "red")
    plt.grid()
    if not noTitle:
        plt.title(f"Predictions from {algorithm_name} ({r'$r^2$'} = {r2:.3f}, {r'rmse'} = {rmse:.3f})")
    plt.ylabel("Simulation ID")
    if doLog:
        plt.xscale("log")
        plt.grid(which="both")
    
    plt.xlabel(f"{field} [{unit}]")

    for i in range(len(truth)):
        plt.plot([truth[i], preds[i]], [sim_ids[i], sim_ids[i]], color = "black", label = "errors")

    # Set legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.savefig(saveFolder / f"{algorithm_name}_{field}_allPredictions.png", bbox_inches="tight")

    plt.subplots(figsize=(12, 8))
    for i in range(len(truth)):
        plt.plot([truth[i], truth[i]], [preds[i], truth[i]], color = "black", label = "errors")
    plt.plot([np.min(truth), np.max(truth)], [np.min(truth), np.max(truth)], color = "blue", linestyle="dashed", label="ideal predictions")
    plt.scatter(truth, preds, marker = "o", color = "red")
    plt.grid()
    if not noTitle:
        plt.title(f"{field} -- {algorithm_name} ({r'$r^2$'} = {r2:.3f}, {r'rmse'} = {rmse:.3f})")
    if doLog:
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(which="both")

    plt.xlabel(f"True values [{unit}]")
    plt.ylabel(f"Predictions [{unit}]")

    # Set legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.savefig(saveFolder / f"{algorithm_name}_{field}_prediction_error.png", bbox_inches="tight")
    
def regress(
        directory : Path,
        inputSpectraNames : list,
        outputFields : list,
        logFields : list,
        algorithms : list,
        cvFolds : int,
        cvRepeats : int,
        cvStrategy : str = "RepeatedKFolds",
        scaleInputs : bool = False,
        logInputs : bool = False,
        doIce : bool = False,
        includeFreqs : bool = False,
        iceMetricsToUse : list = None,
        resultsFilepath : Path = None,
        doPlot : bool = True,
        noTitle : bool = False,
        nThreads : int = 1
):
    logging.basicConfig(filename='aeon_tsc.log', level=logging.INFO)

    # Initialise results objects
    battery = ml_utils.TSRBattery()
    allPredictionsRecord = []
    battery.package = "Aeon"

    if directory.name != "data":
        data_dir = directory / "data"
    else:
        data_dir = directory
    data_files = glob.glob(str(data_dir / "*.nc")) 
    battery.directory = str(directory.resolve())
    battery.algorithms = algorithms
    battery.normalised = True
    battery.scaledInputs = scaleInputs

    # Input data
    inputs = {name : [] for name in inputSpectraNames}
    inputs = ml_utils.read_data(data_files, inputs, with_names = True, with_coords = True, with_iciness = doIce, denorm_coords = True)
    battery.inputSpectra = np.array(inputSpectraNames)

    if doIce:
        iceMetrics = dict()
        if iceMetricsToUse:
            for m in iceMetricsToUse:
                for n in inputs:
                    if m in n:
                        iceMetrics[n] = inputs[n]
        else:
            for n in inputs.keys():
                if "ICEmetric" in n:
                    iceMetrics[n] = inputs[n]

    # Output data
    outputs = {outputField : [] for outputField in outputFields}
    outputs = ml_utils.read_data(data_files, outputs, with_names = False, with_coords = False)
    battery.outputFields = np.array(outputFields)
    battery.numOutputs = len(outputs.keys())

    if "B0angle" in outputs:
        transf = np.array(outputs["B0angle"])
        outputs["B0angle"] = np.abs(transf - 90.0) 

    spec_lengths = []
    for field in inputSpectraNames:
        spec_lengths.extend([len(s) for s in inputs[field]])
    min_l = np.min(spec_lengths)
    print(f"Max spec length: {np.max(spec_lengths)} min spec length: {min_l}")
    max_common_coord = np.max([c[-1] for c in [inputs[f"{inputSpectrumName}_denorm_coords"] for inputSpectrumName in inputSpectraNames]])
    
    scaler_train = MinMaxScaler(feature_range=(0, 1))
    inputData = []
    for field in inputSpectraNames:
        specs = inputs[field]
        coords = inputs[f"{field}_denorm_coords"]
        for i in range(len(specs)):
            if len(specs[i]) > min_l:
                truncd_series, truncd_coords = ml_utils.truncate_series(specs[i], inputs[f"{field}_denorm_coords"][i], max_common_coord)
                resamp_series, resamp_coords = ml_utils.resample_series(truncd_series, truncd_coords, min_l, f"{field.split('/')[0]}_run_{inputs['sim_ids'][i]}", directory / "spectra_homogenisation/")
                specs[i] = resamp_series
                coords[i] = resamp_coords
        spec_data = specs if not scaleInputs else [scaler_train.fit_transform(s.reshape(-1, 1)).flatten()for s in specs]
        if logInputs:
            for i in range(len(spec_data)):
                trace = spec_data[i] 
                spec_data[i] = [np.log10(trace[1]) if trace[1] != 0.0 else 0.0] + np.log10(trace[1:]).tolist()
        inputData.append(spec_data)
    if includeFreqs:
        inputData.append(coords) # Append only the last set of coordinates (they should be the same for all fields)

    # Reshape into 3D numpy array of shape (n_cases, n_channels, n_timepoints)
    inputSpectra = np.swapaxes(np.array(inputData), 0, 1)

    logFields = np.intersect1d(outputFields, logFields)
    battery.logFields = np.array(logFields)
    battery.original_output_means = dict.fromkeys(outputFields)
    battery.original_output_stdevs = dict.fromkeys(outputFields)

    battery.equalLengthTimeseries = True
    battery.numObservations = inputSpectra.shape[0]
    battery.numInputDimensions = inputSpectra.shape[1]
    battery.numTimepointsIfEqual = inputSpectra.shape[2]
    battery.multivariate = battery.numInputDimensions > 1

    battery.cvStrategy = cvStrategy
    battery.cvFolds = cvFolds
    battery.cvRepeats = cvRepeats
    battery.results = []

    trainingTimeTotal_ns = 0
    cvTimeTotal_ns = 0
    fold_training_times_ns = []
    fold_inference_times_cpu_ns = []
    fold_inference_times_clock_ns = []

    # Dumb hack
    best_results = {"backgroundDensity" : 0.452861, "beamFraction" : 0.313241, "B0strength" : 0.024969, "pitch" : 0.587287}

    clock_time_start = time.time()
    cv_time_start = time.process_time_ns()
    for output_field, output_values in outputs.items():
        
        assert len(output_values) == inputSpectra.shape[0]
        case_indices = np.arange(len(output_values))
        output_values = np.array(output_values)
        if cvStrategy == "RepeatedKFolds":
            cv = RepeatedKFold(n_splits=cvFolds, n_repeats=cvRepeats)
        elif cvStrategy == "LeaveOneOut":
            cv = LeaveOneOut()
        else:
            print("CV Strategy not implemented. Defaulting to RepeatedKFolds.")
            cv = RepeatedKFold(n_splits=cvFolds, n_repeats=cvRepeats)
        tt_split = list(enumerate(cv.split(case_indices)))

        if output_field in logFields:
            output_values = np.log10(output_values)

        # Record denormalisation parameters
        _, scaler = ml_utils.normalise_data(output_values)
        print(f"Original data mean: {np.mean(output_values)}, original data SD: {np.std(output_values)}")
        print(f"Mean (0.0) in normalised RMSE units is {scaler.mean_} in original {output_field} units (or {10**scaler.mean_} in log space).")
        print(f"SD in normalised RMSE units is {np.sqrt(scaler.var_)} in original {output_field} units (or {10**np.sqrt(scaler.var_)} in log space).")
        print(f"Best RMSE = {best_results[output_field]}, which denormalises to {scaler.inverse_transform([[best_results[output_field]]])[0][0] - scaler.mean_[0]}, or {10**scaler.inverse_transform([[best_results[output_field]]])[0][0]} in log space.")
        print(f"ALT LOG: Best RMSE high = {10**(scaler.mean_[0] + (np.sqrt(scaler.var_) * best_results[output_field]))}")
        print(f"ALT LOG: Best RMSE low = {10**(scaler.mean_[0] - (np.sqrt(scaler.var_) * best_results[output_field]))}")
        battery.original_output_means[output_field] = scaler.mean_
        battery.original_output_stdevs[output_field] = np.sqrt(scaler.var_)

        for algorithm in algorithms:
            print(f"Building {algorithm} model for {output_field} from {inputSpectraNames}....")
            
            # Results
            result = ml_utils.TSRResult()
            result.output = output_field
            result.algorithm = algorithm
            
            tsr = ml_utils.get_algorithm(algorithm, nThreads)

            # CV Folds
            all_test_indices = []
            all_train_R2s = []
            all_test_R2s = []
            all_test_points = []
            all_predictions = []
            all_test_points_denormed = []
            all_predictions_denormed = []
            
            for fold, (train, test) in tt_split:
                fold_start_training_time = time.process_time_ns()
                print(f"Fold: {fold} (test indices: {test})....")
                # print(f"    Train indices: {train}")
                # print(f"    Test indices:  {test}")

                train_x = [inputSpectra[t] for t in train]
                train_y = output_values[train]
                test_x = [inputSpectra[t] for t in test]
                test_y = output_values[test]

                # Renormalise for each split
                train_y, scaler = ml_utils.normalise_data(train_y)
                test_y, _ = ml_utils.normalise_data(test_y, scaler = scaler)
                print(f"scaler mean: {scaler.mean_}")
                print(f"1.0 in normalised RMSE units is {scaler.inverse_transform([[1.0]])} in original {output_field} units (may be logged).")

                print("    Training model....")
                # Fit
                tsr.fit(train_x, train_y)
                fold_end_training_time = time.process_time_ns()
                
                # Timing
                fold_time_ns = fold_end_training_time - fold_start_training_time
                fold_training_times_ns.append(fold_time_ns)
                trainingTimeTotal_ns += fold_time_ns
                print(f"Fold training time: {fold_time_ns / 1E9} s")

                # Predict
                fold_inf_time_clock_start = time.perf_counter_ns()
                fold_inference_time_start = time.process_time_ns()
                predictions = tsr.predict(test_x)
                fold_inference_time_end = time.process_time_ns()
                fold_inf_time_clock_end = time.perf_counter_ns()
                fold_inference_time_ns = fold_inference_time_end - fold_inference_time_start
                fold_inf_time_clock_ns = fold_inf_time_clock_end - fold_inf_time_clock_start
                fold_inference_times_cpu_ns.append(fold_inference_time_ns)
                fold_inference_times_clock_ns.append(fold_inf_time_clock_ns)
                print(f"Fold inference time: {fold_inference_time_ns} CPU ns or {fold_inf_time_clock_ns} clock ns.")

                preds_denormed = ml_utils.denormalise_data(predictions, scaler)
                test_y_denormed = ml_utils.denormalise_data(test_y, scaler)
                if output_field in logFields:
                    preds_denormed = 10.0**preds_denormed
                    test_y_denormed = 10.0**test_y_denormed
                print(f"    Predictions:  {predictions} (normalised), {preds_denormed} (original)")
                print(f"    Ground truth: {test_y} (normalised), {test_y_denormed} (original)")
                score = tsr.score(test_x, test_y, metric='r2')
                all_test_R2s.append(score)
                skl_rmse = root_mean_squared_error(test_y, predictions)
                training_r2 = tsr.score(train_x, train_y, metric='r2')
                all_train_R2s.append(training_r2)
                print(f"    training r2:  {training_r2}")
                print(f"    knn r2:       {score}")
                print(f"    sklearn rmse: {skl_rmse} (actuals S.D.: {np.std(test_y)})")

                all_test_indices.extend(test.tolist())
                all_test_points.extend(test_y.tolist())
                all_test_points_denormed.extend(test_y_denormed)
                all_predictions.extend(list(predictions))
                all_predictions_denormed.extend(preds_denormed)

                # Log predictions
                testLens = [len(predictions), len(test), len(test_y), len(test_y_denormed), len(predictions), len(preds_denormed)]
                assert len(set(testLens)) == 1 # All lists have equal length
                for i in range(len(predictions)):
                    predRecord = ml_utils.TSRPrediction(
                        algorithm=algorithm,
                        inputChannels=np.array(inputSpectraNames),
                        outputQuantity=output_field,
                        datapoint_ID=test[i],
                        fold_ID=fold,
                        trueValue_normalised=test_y[i],
                        trueValue_denormalised=test_y_denormed[i][0],
                        trueValue_denormalised_log10=np.log10(test_y_denormed[i][0]),
                        predictedValue_normalised=predictions[i],
                        predictedValue_denormalised=preds_denormed[i][0],
                        predictedValue_denormalised_log10=np.log10(preds_denormed[i][0])
                    )
                    allPredictionsRecord.append(predRecord)

            rmse, rmse_var, rmse_se = ml_utils.root_mean_squared_error(all_predictions, all_test_points)
            r2 = np.mean(all_test_R2s)
            r2_sem = sem(all_test_R2s)
            r2_var = np.var(all_test_R2s)

            if math.isnan(r2):
                # Recalculate based on r2 over folds (primarily for LOOCV)
                r2 = r2_score(all_test_points, all_predictions)

            mean_training_r2 = np.mean(all_train_R2s)
            train_r2_sem = sem(all_train_R2s)
            train_r2_var = np.var(all_train_R2s)
            
            summary_str = f"{output_field} -- {algorithm}: Mean test r2 = {r2:.5f}+-{r2_sem:.5f}, mean test RMSE = {rmse:.5f}+-{rmse_se:.5f}, mean train r2 = {mean_training_r2}+-{train_r2_sem}"
            print("--------------------------------------------------------------------------------------------------------------------------")
            print(summary_str)
            logger.info(summary_str)
            print("--------------------------------------------------------------------------------------------------------------------------")
            result.cvR2_mean = r2
            result.cvR2_var = r2_var
            result.cvR2_stderr = r2_sem
            result.cvRMSE_mean = rmse
            result.cvRMSE_var = rmse_var
            result.cvRMSE_stderr = rmse_se
            result.cvMAE_mean = mean_absolute_error(y_true=all_test_points, y_pred=all_predictions, multioutput="uniform_average")
            mae_all = mean_absolute_error(y_true=all_test_points, y_pred=all_predictions, multioutput="raw_values")
            result.cvMAE_var = np.var(mae_all)
            result.cvMAE_stderr = sem(mae_all)
            result.cvMAPE_mean = mean_absolute_percentage_error(y_true=all_test_points, y_pred=all_predictions, multioutput="uniform_average")
            mape_all = mean_absolute_percentage_error(y_true=all_test_points, y_pred=all_predictions, multioutput="raw_values")
            result.cvMAPE_var = np.var(mape_all)
            result.cvMAPE_stderr = sem(mape_all)
            result.trainR2_mean = mean_training_r2
            result.trainR2_stderr = train_r2_sem
            result.trainR2_var = train_r2_var

            battery.results.append(result)

            if doIce:               
                correlate_predictions_with_iciness(
                    algorithm_name = algorithm,
                    field = output_field,
                    truth = all_test_points,
                    preds = all_predictions,
                    iceMetrics = iceMetrics,
                    r2 = result.cvR2_mean,
                    rmse = result.cvRMSE_mean,
                    saveFolder = resultsFilepath.parent / "iciness" / algorithm, 
                )
            if doPlot:
                plot_predictions(
                    algorithm_name = algorithm,
                    field = output_field,
                    sim_ids = all_test_indices, 
                    truth = all_test_points_denormed,
                    preds = all_predictions_denormed,
                    r2 = result.cvR2_mean,
                    rmse = result.cvRMSE_mean,
                    saveFolder = resultsFilepath.parent / "predictions" / algorithm, 
                    doLog = output_field in logFields,
                    noTitle = noTitle
                )
    
    clock_time_end = time.time()
    cv_time_end = time.process_time_ns()
    cvTimeTotal_ns = cv_time_end - cv_time_start
    clock_time = clock_time_end - clock_time_start
    print(f"Clock time: {clock_time / 60.0} min. Process time: {cvTimeTotal_ns / 6E10} min.")

    battery.cvTimeTotal_CPUhours = float(cvTimeTotal_ns) / 3.6E12
    battery.inferenceTimeMinPerFold_CPUns = int(np.rint(np.min(fold_inference_times_cpu_ns)))
    battery.inferenceTimeMinPerFold_CPUms = float(battery.inferenceTimeMinPerFold_CPUns) / 1E6
    battery.inferenceTimeMinPerFold_ClockNs = int(np.rint(np.min(fold_inference_times_clock_ns)))
    battery.inferenceTimeMinPerFold_ClockMs = float(battery.inferenceTimeMinPerFold_ClockNs) / 1E6
    battery.inferenceTimeMeanPerFold_CPUns = int(np.rint(np.mean(fold_inference_times_cpu_ns)))
    battery.inferenceTimeMeanPerFold_CPUms = float(battery.inferenceTimeMeanPerFold_CPUns) / 1E6
    battery.inferenceTimeMeanPerFold_ClockNs = int(np.rint(np.mean(fold_inference_times_clock_ns)))
    battery.inferenceTimeMeanPerFold_ClockMs = float(battery.inferenceTimeMeanPerFold_ClockNs) / 1E6
    battery.trainingTimeMinPerFold_CPUhours = np.min(fold_training_times_ns)  / 3.6E12
    battery.trainingTimeMeanPerFold_CPUhours = np.mean(fold_training_times_ns)  / 3.6E12
    battery.trainingTimeTotal_CPUns = trainingTimeTotal_ns
    battery.trainingTimeTotal_CPUhours = float(trainingTimeTotal_ns) / 3.6E12

    # Write results and all predictions
    ml_utils.write_ML_result_to_file(battery, resultsFilepath)
    if len(allPredictionsRecord) > 0:
        prds_path = resultsFilepath.parent / "predictions"
        prds_path.mkdir(parents = True, exist_ok=True)
        with open(prds_path / f"{resultsFilepath.name.replace('.json', '').replace('.', '')}_predictions.csv", "w") as f:
            w = DataclassWriter(f, allPredictionsRecord, ml_utils.TSRPrediction)
            w.write()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--dir",
        action="store",
        help="Directory containing netCDF files of simulation output.",
        required = True,
        type=Path
    )
    parser.add_argument(
        "--inputSpectra",
        action="store",
        help="Spectra to use for TSR input.",
        required = True,
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--outputFields",
        action="store",
        help="Fields to use for TSR output.",
        required = True,
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--logFields",
        action="store",
        help="Fields to log.",
        required = False,
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--algorithms",
        action="store",
        help="Algorithms to test.",
        required = True,
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--cvFolds",
        action="store",
        help="Number of folds to use in k-folds cross-validation.",
        required = False,
        type=int,
        default=10
    )
    parser.add_argument(
        "--cvRepeats",
        action="store",
        help="Number of repeats to use in k-folds cross-validation.",
        required = False,
        type=int,
        default=10
    )
    parser.add_argument(
        "--cvStrategy",
        action="store",
        help="CV strategy.",
        required = False,
        type=str
    )
    parser.add_argument(
        "--scaleInputs",
        action="store_true",
        help="Scale inputs to between 0-1, removing amplitudes of spectra. Makes prediction worse, so only use when necessary (e.g. comparison to experiment).",
        required = False
    )
    parser.add_argument(
        "--logInputs",
        action="store_true",
        help="Take logarithms of input spectra.",
        required = False
    )
    parser.add_argument(
        "--doPlot",
        action="store_true",
        help="Plot predictions.",
        required = False
    )
    parser.add_argument(
        "--noTitle",
        action="store_true",
        help="Exclude title from prediction plots.",
        required = False
    )
    parser.add_argument(
        "--doIce",
        action="store_true",
        help="Correlate predictions with ICE metrics.",
        required = False
    )
    parser.add_argument(
        "--includeFreqs",
        action="store_true",
        help="Beta: include frequencies in Hz (not gyrofrequency) as a channel.",
        required = False
    )
    parser.add_argument(
        "--james",
        action="store_true",
        help="Run TSER against James' linear data in h5 format.",
        required = False
    )
    parser.add_argument(
        "--resultsFilepath",
        action="store",
        help="Filepath of csv to which to write results.",
        required = False,
        type=Path
    )
    parser.add_argument(
        "--iceMetrics",
        action="store",
        help="ICE metrics to correlate and plot.",
        required = False,
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--nThreads",
        action="store",
        help="Number of threads to use for training and prediction.",
        required = False,
        type=int,
        default=1
    )

    args = parser.parse_args()

    if args.doPlot and (args.cvStrategy != "LeaveOneOut"):
        print("WARNING: Prediction plots will only make sense with a LeaveOneOut cross-validation strategy.")

    if not args.james:
        regress(
            args.dir, 
            args.inputSpectra, 
            args.outputFields, 
            args.logFields, 
            args.algorithms, 
            args.cvFolds, 
            args.cvRepeats, 
            args.cvStrategy,
            scaleInputs=args.scaleInputs,
            logInputs=args.logInputs,
            doIce=args.doIce,
            includeFreqs=args.includeFreqs,
            iceMetricsToUse=args.iceMetrics,
            resultsFilepath=args.resultsFilepath,
            doPlot=args.doPlot,
            noTitle=args.noTitle,
            nThreads =args.nThreads)
    if args.james:
        regress_from_hd5(
            directory=args.dir,
            outputFields=args.outputFields,
            logFields=args.logFields if args.logFields is not None else [],
            algorithms=args.algorithms,
            logInputs=args.logInputs,
            resultsFilepath=args.resultsFilepath,
            doPlot=args.doPlot,
            noTitle=args.noTitle,
            nThreads=args.nThreads,
            numFolds=args.cvFolds,
            numRepeats=args.cvRepeats
        )
    # demo()