import argparse
import glob
import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, load_robot_execution_failures
import matplotlib.pyplot as plt
from tsfresh import extract_features

import ml_utils

def demo():
    download_robot_execution_failures()
    timeseries, y = load_robot_execution_failures()

    timeseries[timeseries['id'] == 3].plot(subplots=True, sharex=True, figsize=(10,10))
    plt.show()

    timeseries[timeseries['id'] == 20].plot(subplots=True, sharex=True, figsize=(10,10))
    plt.show()

    extracted_features = extract_features(timeseries, column_id="id", column_sort="time")

    print(extracted_features)

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

    # tsfresh expects either:
    # # A flat dataframe with columns id, time, and each spectrum as its own column
    # # A stacked dataframe with columns id, time, spectrum label and spectrum value (all spectral values in one long column)
    # # A dictionary with keys for each spectrum and values are dataframes with id, time, value
    spectra_dict = {}
    for spectrum in inputSpectraNames:
        
        numCases = len(inputs[spectrum])
        numTimePoints = len(inputs[spectrum][0])
        
        all_time_points = np.concatenate(inputs[f"{spectrum}_coords"])
        sim_ids = inputs["sim_ids"]
        all_sim_ids_repeated = np.repeat(sim_ids, numTimePoints)
        all_spectrum_data = np.concatenate(inputs[spectrum])

        assert len(all_time_points) == len(all_sim_ids_repeated)
        assert len(all_time_points) == len(all_spectrum_data)

        data_dic = { "id" : all_sim_ids_repeated, "time": all_time_points, "value": all_spectrum_data }
        data_df = pd.DataFrame(data=data_dic)
        spectra_dict[ml_utils.fieldNameToText(spectrum)] = data_df

    # Extract features
    extracted_features = extract_features(spectra_dict, column_id = "id", column_sort = "time", column_kind = None, column_value = "value")

    # Save features
    if save:
        save_feature_dir = directory / "feature_extraction"
        if not os.path.exists(save_feature_dir):
            os.mkdir(save_feature_dir)
        xr.Dataset.from_dataframe(extracted_features).to_netcdf(save_feature_dir / "features.nc")

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