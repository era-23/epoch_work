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
from sklearn.model_selection import RepeatedKFold

from aeon.datasets import load_cardano_sentiment, load_covid_3month
from aeon.transformations.collection import Normalizer

from sklearn.metrics import root_mean_squared_error

import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

def demo():
    covid_train, covid_train_y = load_covid_3month(split="train")
    covid_test, covid_test_y = load_covid_3month(split="test")
    cardano_train, cardano_train_y = load_cardano_sentiment(split="train")
    cardano_test, cardano_test_y = load_cardano_sentiment(split="test")
    print(f"Covid spectrum shape:   {covid_train.shape} (n_cases: {covid_train.shape[0]}, n_channels: {covid_train.shape[1]}, n_timepoints: {covid_train.shape[2]})")
    print(f"Covid output shape:     {covid_train_y.shape}")
    print(f"Cardano spectrum shape: {cardano_train.shape} (n_cases: {cardano_train.shape[0]}, n_channels: {cardano_train.shape[1]}, n_timepoints: {cardano_train.shape[2]})")
    print(f"Cardano output shape:   {cardano_train_y.shape}")

def regress(
        directory : Path,
        inputSpectraNames : list,
        outputFields : list,
        logFields : list,
        algorithms : list,
        cvFolds : int,
        cvRepeats : int,
        normalise : bool = True,
        resultsFilepath = None
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
    
    for field in inputSpectraNames:
        specs = inputs[field]
        for i in range(len(specs)):
            if len(specs[i]) > min_l:
                truncd_series, truncd_coords = ml_utils.truncate_series(specs[i], inputs[f"{field}_coords"][i], max_common_coord)
                resamp_series, _ = ml_utils.downsample_series(truncd_series, truncd_coords, min_l, f"{field.split('/')[0]}_run_{inputs['sim_ids'][i]}", directory / "spectra_homogenisation/")
                specs[i] = resamp_series

    # Reshape into 3D numpy array of shape (n_cases, n_channels, n_timepoints)
    inputSpectra = np.array([np.reshape(a, (1,-1)) for a in specs])
    # outputs = np.array(list(outputs.values()))

    logFields = np.intersect1d(outputFields, logFields)
    battery.logFields = np.array(logFields)
    denormalisation_parameters = dict.fromkeys(outputFields)
    battery.original_output_means = dict.fromkeys(outputFields)
    battery.original_output_stdevs = dict.fromkeys(outputFields)
    if normalise:
        for field, vals in outputs.items():
            # print(f"orig mean: {np.mean(vals)}")
            # print(f"orig SD: {np.std(vals)}")
            if field in logFields:
                # print(f"orig log mean: {np.mean(np.log10(vals))}")
                # print(f"orig log SD: {np.std(np.log10(vals))}")
                outputs[field], mean, sd = ml_utils.normalise_1D(vals, doLog = True)
            else:
                outputs[field], mean, sd = ml_utils.normalise_1D(vals)
            battery.original_output_means[field] = mean
            battery.original_output_stdevs[field] = sd
            denormalisation_parameters[field] = (mean, sd)
            print(f"1.0 in normalised RMSE units is {ml_utils.denormalise_rmse(1.0, sd)} in original {field} units.")
        
        print(f"out mean: {np.mean(list(outputs.values()))}, out std: {np.std(list(outputs.values()))}")

        # norm = Normalizer()
        # inputSpectra = norm.fit_transform(inputSpectra)

        # spec_flat = []
        # for c in inputSpectra_norm:
        #     spec_flat.extend(c[0]) 
        # print(f"in mean:  {np.mean(spec_flat)}, in std:  {np.std(spec_flat)}")
    else:
        for field in logFields:
            outputs[field] = np.log10(outputs[field])

    battery.equalLengthTimeseries = True
    battery.numObservations = inputSpectra.shape[0]
    battery.numInputDimensions = inputSpectra.shape[1]
    battery.numTimepointsIfEqual = inputSpectra.shape[2]
    battery.multivariate = battery.numInputDimensions > 1

    battery.cvStrategy = "RepeatedKFolds"
    battery.cvFolds = cvFolds
    battery.cvRepeats = cvRepeats
    battery.results = []

    for output_field, output_values in outputs.items():
        
        assert len(output_values) == inputSpectra.shape[0]
        case_indices = np.arange(len(output_values))
        output_values = np.array(output_values)
        rkf = RepeatedKFold(n_splits=cvFolds, n_repeats=cvRepeats)
        tt_split = list(enumerate(rkf.split(case_indices)))
        
        for algorithm in algorithms:
            print(f"Building {algorithm} model for {output_field} from {inputSpectraNames}....")
            
            # Results
            result = ml_utils.TSRResult()
            result.output = output_field
            result.algorithm = algorithm
            
            tsr = ml_utils.get_algorithm(algorithm)

            # Repeated K Folds
            all_R2s = []
            all_test_points = []
            all_predictions = []
            
            for fold, (train, test) in tt_split:
                print(f"Fold: {fold}....")
                # print(f"    Train indices: {train}")
                # print(f"    Test indices:  {test}")

                train_x = [inputSpectra[t] for t in train]
                train_y = output_values[train]
                test_x = [inputSpectra[t] for t in test]
                test_y = output_values[test]

                print("    Training model....")
                tsr.fit(train_x, train_y)
                predictions = tsr.predict(test_x)
                print(f"    Predictions:  {predictions}")
                print(f"    Ground truth: {test_y}")
                score = tsr.score(test_x, test_y, metric='r2')
                all_R2s.append(score)
                skl_rmse = root_mean_squared_error(test_y, predictions)
                print(f"    knn r2:       {score}")
                print(f"    sklearn rmse: {skl_rmse} (actuals S.D.: {np.std(test_y)})")

                all_test_points.extend(test_y.tolist())
                all_predictions.extend(list(predictions))
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
        help="Spectra to use for TSC input.",
        required = True,
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--outputFields",
        action="store",
        help="Fields to use for TSC output.",
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
        "--resultsFilepath",
        action="store",
        help="Filepath of csv to which to write results.",
        required = False,
        type=Path
    )

    args = parser.parse_args()

    regress(args.dir, args.inputSpectra, args.outputFields, args.logFields, args.algorithms, args.cvFolds, args.cvRepeats, resultsFilepath=args.resultsFilepath)
    # demo()