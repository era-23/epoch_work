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

    args = parser.parse_args()

    extract_all(args.dir, args.inputSpectra, args.saveFeatures)
    # demo()