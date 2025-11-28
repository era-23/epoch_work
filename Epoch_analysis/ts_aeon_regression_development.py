import argparse
import glob
import os
from pathlib import Path
import warnings
import ml_utils
import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy.stats import sem
from sklearn.model_selection import RepeatedKFold, LeaveOneOut

from aeon.datasets import load_cardano_sentiment, load_covid_3month
from aeon.transformations.collection import Normalizer

from sklearn.metrics import root_mean_squared_error

import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

plt.rcParams.update({'axes.titlesize': 26.0})
plt.rcParams.update({'axes.labelsize': 24.0})
plt.rcParams.update({'xtick.labelsize': 20.0})
plt.rcParams.update({'ytick.labelsize': 20.0})
plt.rcParams.update({'legend.fontsize': 18.0})

def demo():
    covid_train, covid_train_y = load_covid_3month(split="train")
    covid_test, covid_test_y = load_covid_3month(split="test")
    cardano_train, cardano_train_y = load_cardano_sentiment(split="train")
    cardano_test, cardano_test_y = load_cardano_sentiment(split="test")
    print(f"Covid spectrum shape:   {covid_train.shape} (n_cases: {covid_train.shape[0]}, n_channels: {covid_train.shape[1]}, n_timepoints: {covid_train.shape[2]})")
    print(f"Covid output shape:     {covid_train_y.shape}")
    print(f"Cardano spectrum shape: {cardano_train.shape} (n_cases: {cardano_train.shape[0]}, n_channels: {cardano_train.shape[1]}, n_timepoints: {cardano_train.shape[2]})")
    print(f"Cardano output shape:   {cardano_train_y.shape}")

def plot_predictions(
        algorithm_name : str,
        field : str,
        sim_ids : list,
        truth : list,
        preds : list,
        saveFolder : Path,
        log_x : bool
):
    plt.subplots(figsize=(12, 8))
    plt.scatter(truth, sim_ids, label = "True value", marker = "o", color = "blue")
    plt.scatter(preds, sim_ids, label = "Predicted value", marker = "o", color = "red")
    plt.title(f"Predictions from {algorithm_name}")
    plt.ylabel("Simulation ID")
    if log_x:
        plt.xscale("log")
        plt.xlabel(f"{field} (log)")
    else:
        plt.xlabel(field)

    for i in range(len(truth)):
        plt.plot([truth[i], preds[i]], [sim_ids[i], sim_ids[i]], color = "black", label = "errors")

    # Set legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.savefig(saveFolder / f"{algorithm_name}_{field}_predictions.png")
    
def regress(
        directory : Path,
        inputSpectraNames : list,
        outputFields : list,
        logFields : list,
        algorithms : list,
        cvFolds : int,
        cvRepeats : int,
        cvStrategy : str = "RepeatedKFolds",
        normalise : bool = True,
        resultsFilepath : Path = None,
        doPlot : bool = True
):
    logging.basicConfig(filename='aeon_tsc.log', level=logging.INFO)

    # Initialise results object
    battery = ml_utils.TSRBattery()
    battery.package = "Aeon"

    if directory.name != "data":
        data_dir = directory / "data"
    else:
        data_dir = directory
    data_files = glob.glob(str(data_dir / "*.nc")) 
    battery.directory = str(directory.resolve())
    battery.algorithms = algorithms
    battery.normalised = normalise

    # Input data
    inputs = {name : [] for name in inputSpectraNames}
    inputs = ml_utils.read_data(data_files, inputs, with_names = True, with_coords = True)
    battery.inputSpectra = np.array(inputSpectraNames)

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
    max_common_coord = np.max([c[-1] for c in [inputs[f"{inputSpectrumName}_coords"] for inputSpectrumName in inputSpectraNames]])
    if not os.path.exists(directory / "spectra_homogenisation/"):
        os.mkdir(directory / "spectra_homogenisation/")
    
    inputData = []
    for field in inputSpectraNames:
        specs = inputs[field]
        for i in range(len(specs)):
            if len(specs[i]) > min_l:
                truncd_series, truncd_coords = ml_utils.truncate_series(specs[i], inputs[f"{field}_coords"][i], max_common_coord)
                resamp_series, _ = ml_utils.downsample_series(truncd_series, truncd_coords, min_l, f"{field.split('/')[0]}_run_{inputs['sim_ids'][i]}", directory / "spectra_homogenisation/")
                specs[i] = resamp_series
        inputData.append(specs) 

    # Reshape into 3D numpy array of shape (n_cases, n_channels, n_timepoints)
    inputSpectra = np.swapaxes(np.array(inputData), 0, 1)
    # outputs = np.array(list(outputs.values()))

    logFields = np.intersect1d(outputFields, logFields)
    battery.logFields = np.array(logFields)
    denormalisation_parameters = dict.fromkeys(outputFields)
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
        
        for algorithm in algorithms:
            print(f"Building {algorithm} model for {output_field} from {inputSpectraNames}....")
            
            # Results
            result = ml_utils.TSRResult()
            result.output = output_field
            result.algorithm = algorithm
            
            tsr = ml_utils.get_algorithm(algorithm)

            # CV Folds
            all_test_indices = []
            all_R2s = []
            all_test_points = []
            all_predictions = []
            all_test_points_denormed = []
            all_predictions_denormed = []
            
            for fold, (train, test) in tt_split:
                print(f"Fold: {fold} (test indices: {test})....")
                # print(f"    Train indices: {train}")
                # print(f"    Test indices:  {test}")

                train_x = [inputSpectra[t] for t in train]
                train_y = output_values[train]
                test_x = [inputSpectra[t] for t in test]
                test_y = output_values[test]

                # Renormalise for each split
                if normalise:
                    train_y, scaler = ml_utils.normalise_data(train_y)
                    test_y, _ = ml_utils.normalise_data(test_y, scaler = scaler)
                    print(f"scaler mean: {scaler.mean_}")
                    print(f"1.0 in normalised RMSE units is {scaler.inverse_transform([[1.0]])} in original {output_field} units (may be logged).")

                print("    Training model....")
                tsr.fit(train_x, train_y)
                predictions = tsr.predict(test_x)
                preds_denormed = ml_utils.denormalise_data(predictions, scaler)
                test_y_denormed = ml_utils.denormalise_data(test_y, scaler)
                if output_field in logFields:
                    preds_denormed = 10.0**preds_denormed
                    test_y_denormed = 10.0**test_y_denormed
                print(f"    Predictions:  {predictions} (normalised), {preds_denormed} (original)")
                print(f"    Ground truth: {test_y} (normalised), {test_y_denormed} (original)")
                score = tsr.score(test_x, test_y, metric='r2')
                all_R2s.append(score)
                skl_rmse = root_mean_squared_error(test_y, predictions)
                print(f"    knn r2:       {score}")
                print(f"    sklearn rmse: {skl_rmse} (actuals S.D.: {np.std(test_y)})")

                all_test_indices.extend(test.tolist())
                all_test_points.extend(test_y.tolist())
                all_test_points_denormed.extend(test_y_denormed)
                all_predictions.extend(list(predictions))
                all_predictions_denormed.extend(preds_denormed)

            rmse, rmse_var, rmse_se = ml_utils.root_mean_squared_error(all_predictions, all_test_points)
            summary_str = f"{output_field} -- {algorithm}: Mean r2 = {np.mean(all_R2s):.5f}+-{sem(all_R2s):.5f}, mean RMSE: {rmse:.5f}+-{rmse_se:.5f}"
            print("--------------------------------------------------------------------------------------------------------------------------")
            print(summary_str)
            logger.info(summary_str)
            print("--------------------------------------------------------------------------------------------------------------------------")
            result.cvR2_mean = np.mean(all_R2s)
            result.cvR2_var = np.var(all_R2s)
            result.cvR2_stderr = sem(all_R2s)
            result.cvRMSE_mean = rmse
            result.cvRMSE_var = rmse_var
            result.cvRMSE_stderr = rmse_se

            battery.results.append(result)

            if doPlot:
                plot_predictions(algorithm, output_field, all_test_indices, all_test_points_denormed, all_predictions_denormed, resultsFilepath.parent, output_field in logFields)
    
    ml_utils.write_ML_result_to_file(battery, resultsFilepath)

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
        required = True,
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
        required = True,
        type=int
    )
    parser.add_argument(
        "--cvRepeats",
        action="store",
        help="Number of repeats to use in k-folds cross-validation.",
        required = True,
        type=int
    )
    parser.add_argument(
        "--cvStrategy",
        action="store",
        help="CV strategy.",
        required = False,
        type=str
    )
    parser.add_argument(
        "--doPlot",
        action="store_true",
        help="Plot predictions.",
        required = False
    )
    parser.add_argument(
        "--resultsFilepath",
        action="store",
        help="Filepath of csv to which to write results.",
        required = False,
        type=Path
    )

    args = parser.parse_args()

    if args.doPlot and (args.cvStrategy is not "LeaveOneOut"):
        print("WARNING: Prediction plots will only make sense with a LeaveOneOut cross-validation strategy.")

    regress(
        args.dir, 
        args.inputSpectra, 
        args.outputFields, 
        args.logFields, 
        args.algorithms, 
        args.cvFolds, 
        args.cvRepeats, 
        args.cvStrategy,
        resultsFilepath=args.resultsFilepath,
        doPlot=args.doPlot)
    # demo()