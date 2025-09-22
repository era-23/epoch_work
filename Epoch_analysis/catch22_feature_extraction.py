import argparse
import glob
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pycatch22
import xarray as xr

import ml_utils

def demo():
    tsData_asList = [1, 2, 3, 4] # (or more interesting data!)
    tsData_as_numpy = np.array([1, 2, 3, 4])

    # for time series data supplied as a list...
    results = pycatch22.catch22_all(tsData_asList)
    # alternatively, for time series data as a numpy array...
    results_numpy = pycatch22.catch22_all(tsData_as_numpy)

    for feature, output in zip(results['names'], results['values']):
        print(f"{feature}: {output}")

def read_extracted_data(directory : Path):
    extract_file = directory / "feature_extraction" / "catch24_features.nc"

    data = xr.open_dataset(extract_file, engine="netcdf4")
    sim_ids = [int(id) for id in data.coords["index"]]

    print(f"Simulations in file: {len(sim_ids)}")
    print(f"Number of features: {len(data.variables) - 1}")

    return data

def plot_extracted_data_against_inputs(inputFeatureNames : list, inputDataFolder : Path, outputFeatures : xr.Dataset):
    
    ##### Get input data
    data_files = glob.glob(str(inputDataFolder / "*.nc"))

    # Input data
    inputs = {name : [] for name in inputFeatureNames}
    inputs = ml_utils.read_data(data_files, inputs, with_names = True, with_coords = False)
    inputs = ml_utils.preprocess_plasma_features(inputs)

    num_cases = len(list(outputFeatures.values())[0])
    num_features = len(list(outputFeatures.keys()))

    # Check number of simulations equal
    assert num_cases == len(inputs["sim_ids"])

    features = dict(outputFeatures.variables.mapping)
    features.pop("index")
    for feature in features.keys():
        # Get indices
        feature_values = [float(outputFeatures[feature].sel(index=i).data) for i in inputs["sim_ids"]]
        ml_utils.plot_4inputs_training_data(
            feature_values, 
            feature, 
            inputs[inputFeatureNames[0]], 
            inputs[inputFeatureNames[1]], 
            inputs[inputFeatureNames[2]], 
            inputs[inputFeatureNames[3]], 
            inputFeatureNames[0],
            inputFeatureNames[1],
            inputFeatureNames[2],
            inputFeatureNames[3],
        )

def extract_all(directory : Path, inputSpectraNames : list, save : bool = True):
    
    if directory.name != "data":
        data_dir = directory / "data"
    else:
        data_dir = directory
        overall_dir = directory.parent
    data_files = glob.glob(str(data_dir / "*.nc"))
    
    # Input data
    inputs = {name : [] for name in inputSpectraNames}
    inputs = ml_utils.read_data(data_files, inputs, with_names = True, with_coords = True)

    features_dict = {}
    for spectrum in inputSpectraNames:
        print(f"Features for {spectrum}:")
        all_spectra = inputs[spectrum]
        for sim_spectrum in all_spectra:
            results = pycatch22.catch22_all(sim_spectrum, catch24 = True)
            for feature, output in zip(results['names'], results['values']):
                if f"{ml_utils.fieldNameToText(spectrum)}__{feature}" not in features_dict:
                    features_dict[f"{ml_utils.fieldNameToText(spectrum)}__{feature}"] = []
                features_dict[f"{ml_utils.fieldNameToText(spectrum)}__{feature}"].append(output)
    features_dict["index"] = np.array(inputs["sim_ids"])
    features_df = pd.DataFrame(features_dict, index = np.array(inputs["sim_ids"]))

    # Save features
    if save:
        save_feature_dir = directory / "feature_extraction"
        if not os.path.exists(save_feature_dir):
            os.mkdir(save_feature_dir)
        xr.Dataset.from_dataframe(features_df).to_netcdf(save_feature_dir / "catch24_features.nc")

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
        "--saveFeatures",
        action="store_true",
        help="Flag to save features after extraction.",
        required = False
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Flag to plot features read from the extraction file.",
        required = False
    )

    args = parser.parse_args()

    extract_all(args.dir, args.inputSpectra, args.saveFeatures)

    # if os.path.exists(args.dir / "feature_extraction" / "catch24_features.nc"):
    #     data = read_extracted_data(args.dir)
    #     if args.plot:
    #         plot_extracted_data_against_inputs(["B0strength", "B0angle", "backgroundDensity", "beamFraction"], args.dir / "data", data)
    # else:
    #     extract_all(args.dir, args.inputSpectra, args.saveFeatures)
    #     if args.plot:
    #         plot_extracted_data_against_inputs(read_extracted_data(args.dir))
    # demo()