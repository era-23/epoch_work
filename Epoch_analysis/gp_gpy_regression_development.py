import argparse
import glob
import os
import GPy
import ml_utils
from pathlib import Path
from matplotlib import pyplot as plt
from dataclasses import fields

def regress(directory : Path, inputFieldName : str, outputFields : list):
    
    if directory.name != "data":
        data_dir = directory / "data"
    else:
        data_dir = directory
    data_files = glob.glob(str(data_dir / "*.nc")) 

    # Input data
    inputs = {inputFieldName : []}
    inputs = ml_utils.read_data(data_files, inputs, with_names = True, with_coords = True)

    # Output data
    outputs = {outp : [] for outp in outputFields}
    outputs = ml_utils.read_data(data_files, outputs, with_names = True, with_coords=False)
    
    features = create_spectral_features(directory, inputFieldName, inputs)

    # Now need to figure out how outputs to be formatted and used in a GP regression

    print(features)

def create_spectral_features(directory : Path, field : str, spectrum_data : dict):
    # if bigLabels:
    #     plt.rcParams.update({'axes.titlesize': 26.0})
    #     plt.rcParams.update({'axes.labelsize': 24.0})
    #     plt.rcParams.update({'xtick.labelsize': 20.0})
    #     plt.rcParams.update({'ytick.labelsize': 20.0})
    #     plt.rcParams.update({'legend.fontsize': 18.0})
    # else:
    #     plt.rcParams.update({'axes.titlesize': 18.0})
    #     plt.rcParams.update({'axes.labelsize': 16.0})
    #     plt.rcParams.update({'xtick.labelsize': 14.0})
    #     plt.rcParams.update({'ytick.labelsize': 14.0})
    #     plt.rcParams.update({'legend.fontsize': 14.0})

    

    # example_model = GPy.examples.regression.sparse_GP_regression_1D(plot=True, optimize=False)
    # plt.show()
    # print(example_model)

    singleValueFields = [f.name for f in fields(ml_utils.SpectralFeatures1D) if f.type is float or f.type is int]
    features = {}

    save_path = directory / "feature_extraction"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    featureSets = []
    for spectrum_idx in range(len(spectrum_data[field])):
        featureSets.append(ml_utils.extract_features_from_1D_power_spectrum(
            spectrum=spectrum_data[field][spectrum_idx], 
            coordinates=spectrum_data[field + "_coords"][spectrum_idx], 
            spectrum_name=spectrum_data["sim_ids"][spectrum_idx], 
            savePath=save_path,
            xLabel=field))

    for featureField in singleValueFields:
        features[featureField] = [simData.__dict__[featureField] for simData in featureSets]
    
    return features

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--dir",
        action="store",
        help="Directory containing netCDF files of simulation output.",
        required = False,
        type=Path
    )
    parser.add_argument(
        "--inputFields",
        action="store",
        help="Fields to use for GP input.",
        required = True,
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--logFields",
        action="store",
        help="Fields which need log transformation preprocessing.",
        required = False,
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--outputFields",
        action="store",
        help="Fields to use for GP output.",
        required = False,
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--matrixPlot",
        action="store_true",
        help="Matrix plot inputs and outputs",
        required = False
    )
    parser.add_argument(
        "--sobol",
        action="store_true",
        help="Calculate and plot SOBOL indices",
        required = False
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate regression models.",
        required = False
    )
    parser.add_argument(
        "--plotModels",
        action="store_true",
        help="Plot regression models.",
        required = False
    )
    parser.add_argument(
        "--showModels",
        action="store_true",
        help="Display regression models.",
        required = False
    )
    parser.add_argument(
        "--bigLabels",
        action="store_true",
        help="Large labels on plots for posters, presentations etc.",
        required = False
    )
    parser.add_argument(
        "--noTitle",
        action="store_true",
        help="No title on plots for posters, papers etc. which will include captions instead.",
        required = False
    )

    args = parser.parse_args()

    regress(args.dir, args.inputFields[0], args.outputFields)