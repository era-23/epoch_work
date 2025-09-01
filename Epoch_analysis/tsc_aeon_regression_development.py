import argparse
import glob
from pathlib import Path
import warnings
import ml_utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import RepeatedKFold

from aeon.datasets import load_cardano_sentiment, load_covid_3month
from aeon.transformations.collection import Normalizer

from sklearn.metrics import r2_score, root_mean_squared_error

warnings.filterwarnings("ignore")

unequal_length_estimators = ["catch22", "knn", "rdst"]

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
        inputSpectrum : str,
        outputFields : list,
        logFields : list,
        normalise : bool = True
):

    # Initialise results object
    results = ml_utils.GPResults()
    results.gpPackage = "Aeon"

    if directory.name != "data":
        data_dir = directory / "data"
    else:
        data_dir = directory
    data_files = glob.glob(str(data_dir / "*.nc")) 
    results.directory = str(directory.resolve())

    # Input data
    inputs = {inputSpectrum : []}
    inputs = ml_utils.read_data(data_files, inputs, with_names = False, with_coords = False)
    results.inputNames = [inputSpectrum]

    # Output data
    outputs = {outputField : [] for outputField in outputFields}
    outputs = ml_utils.read_data(data_files, outputs, with_names = False, with_coords = False)
    results.outputNames = outputFields

    specs = list(inputs.values())[0]
    inputSpectra = [np.reshape(a, (1,-1)) for a in specs]
    # outputs = np.array(list(outputs.values()))

    logFields = np.intersect1d(outputFields, logFields)
    if normalise:
        for field, vals in outputs.items():
            if field in logFields:
                outputs[field] = ml_utils.normalise_1D(np.log10(vals))
            else:
                outputs[field] = ml_utils.normalise_1D(vals)
        
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

    for output_field, output_values in outputs.items():
        print(f"Building model for {output_field} from {inputSpectrum}....")
        assert len(output_values) == len(inputSpectra)
        case_indices = np.arange(len(output_values))
        output_values = np.array(output_values)

        knn = ml_utils.get_algorithm("aeon.KNeighborsTimeSeriesRegressor")

        # Repeated K Folds
        rkf = RepeatedKFold(n_splits=5, n_repeats=1)
        for fold, (train, test) in enumerate(rkf.split(case_indices)):
            print(f"Fold: {fold}....")
            print(f"    Train indices: {train}")
            print(f"    Test indices:  {test}")

            train_x = [inputSpectra[t] for t in train]
            train_y = output_values[train]
            test_x = [inputSpectra[t] for t in test]
            test_y = output_values[test]

            print("    Training model....")
            knn.fit(train_x, train_y)
            predictions = knn.predict(test_x)
            print(f"    Predictions:  {predictions}")
            print(f"    Ground truth: {test_y}")
            score = knn.score(test_x, test_y)
            skl_rmse = root_mean_squared_error(test_y, predictions)
            print(f"    knn r2:       {score}")
            print(f"    sklearn rmse: {skl_rmse} (actuals S.D.: {np.std(test_y)})")

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
        "--inputSpectrum",
        action="store",
        help="Spectrum to use for TSC input.",
        required = True,
        type=str
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

    args = parser.parse_args()

    regress(args.dir, args.inputSpectrum, args.outputFields, args.logFields)
    # demo()