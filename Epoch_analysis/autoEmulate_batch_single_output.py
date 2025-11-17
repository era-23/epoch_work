# General imports for the notebook
import argparse
import glob
import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import torch
import json
from matplotlib import pyplot as plt
import ml_utils
import csv
warnings.filterwarnings("ignore")
from autoemulate.simulations.projectile import Projectile
from autoemulate import AutoEmulate
from autoemulate.transforms import StandardizeTransform, PCATransform
from autoemulate.core.sensitivity_analysis import SensitivityAnalysis

# SMALL_SIZE = 10
# MEDIUM_SIZE = 20
# BIGGER_SIZE = 24
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def plot_results(
        resultsFile : Path,
        outputFields : list = None,
        models : list = None,
        metric : str = "r2_test",
        metricThreshold : float = 0.75,
        featureSet : str = "catch22"
):
    SMALL_SIZE = 10
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 24
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    results_df = pd.read_csv(resultsFile)

    # Useful columns
    data = results_df[["model_name", "outputField", metric, f"{metric}_std"]]

    # Filter by outputFields
    if outputFields:
        data = data[data["outputField"].isin(outputFields)]
        data = data[data[metric] > metricThreshold]
    else:
        outputFields = data[data[metric] > metricThreshold]["outputField"].to_numpy()
        data = data[data["outputField"].isin(outputFields)]
    
    # Filter by models
    if models:
        data = data[data["model_name"].isin(outputFields)]
    else:
        models = list(set(data["model_name"].to_numpy()))
    
    # Plot grouped bar chart
    barVals = {model : [] for model in models}
    barErrs = {model : [] for model in models}
    for field in outputFields:
        fieldResults = data[data["outputField"] == field]
        for _, res in fieldResults.iterrows():
            barVals[res["model_name"]].append(res[metric])
            barErrs[res["model_name"]].append(res[f"{metric}_std"])

    x = np.arange(len(outputFields))
    
    width = 1.0/(len(models) + 1)  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(figsize=(15, 15))

    for model, value in barVals.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, value, width, label=model, yerr=barErrs[model], edgecolor='black')
        #ax.errorbar(x + offset, value, yerr=barErrs[algorithm], fmt=",", color = "k")
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_xlabel('Output field')
    ax.set_ylabel(metric)
    xLabels = [f"{f.split('__')[0]}__{ml_utils.catch22_short_names[f.split('__')[1]]}" for f in outputFields]
    ax.set_xticks(x + (0.5 * (len(models) -1) * width), xLabels, rotation=90)
    ax.legend(loc='lower left')
    ax.set_ylim((0.5, 1))
    fig.tight_layout()

    plt.show()

def run_single_output_GP(
        input_ptt : torch.Tensor,
        output_ptt : torch.Tensor,
        inputFieldNames : list,
        outputFieldName : str,
        saveFolder : Path = None, 
        models : list = None, 
        name : str = None,
        plot : bool = False,
        doPCA : bool = False,
        pcaComponents : int = 8,
        doSobol : bool = False,
        sobolSamples : int = 2048,
        numFolds : int = 9,
        numRepeats : int = 3):

    print(input_ptt.shape)
    print(output_ptt.shape)

    num_input_vars = input_ptt.shape[1]
    print(f"Num input variables: {num_input_vars}")
    num_cases = input_ptt.shape[0]
    print(f"Num cases: {num_cases}")
    num_output_vars = output_ptt.shape[1]
    print(f"Num output variables: {num_output_vars}")

    # Run AutoEmulate with default settings
    ae = AutoEmulate(
        input_ptt, 
        output_ptt, 
        models = models, 
        y_transforms_list = None if not doPCA else [[StandardizeTransform(), PCATransform(n_components = pcaComponents, cache_size = 1), StandardizeTransform()]],
        only_probabilistic = (models is None), 
        shuffle=False,
        n_splits = numFolds, 
        n_bootstraps = numRepeats, 
        log_level="progress_bar"
    )
    best = ae.best_result()
    print("Model with id: ", best.id, " performed best: ", best.model_name)
    print(ae.summarise())
    
    if plot:
        ae.plot(best, fname=saveFolder / (f"AE_bestModel_{name}.png" if name is not None else "AE_bestModel.png"))
        parameters_range = dict.fromkeys(inputFieldNames)
        max_input_vals = torch.max(input_ptt, dim = 0)[0]
        min_input_vals = torch.min(input_ptt, dim = 0)[0]
        for i in range(len(inputFieldNames)):
            parameters_range[inputFieldNames[i]] = (float(max_input_vals[i]), float(min_input_vals[i]))
        ae.plot_surface(best.model, input_index_pair=(0,2), parameters_range = parameters_range, quantile = 0.5, fname=saveFolder / (f"AE_bestModel_surface_{name}.png" if name is not None else "AE_bestModelSurface.png"))
    # Save best model
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)
    try:
        best_result_filepath = ae.save(best, saveFolder, use_timestamp=True)
        print("Model and metadata saved to: ", best_result_filepath)
    except:
        print(f"Unable to save model to {saveFolder}.")

    # Save results
    # results_json_filename = saveFolder / (f"AE_results_{name}.json" if name is not None else "autoEmulate.json")
    results = ae.summarise()
    results["outputField"] = outputFieldName
    results["inputFieldNames"] = "|".join(inputFieldNames)
    results["pca"] = doPCA
    results["pcaComponents"] = pcaComponents
    results["cvFolds"] = numFolds
    results["cvRepeats"] = numRepeats
    results["rmse_test_std"] = best.rmse_test_std
    return results

def harness_multiple_single_output_GPs(
        inputDir : Path, 
        inputFields : list, 
        outputFile : Path, 
        saveFolder : Path, 
        outputFields : list = None,
        models : list = None, 
        name : str = None,
        plot : bool = False,
        doPCA : bool = False,
        pcaComponents : int = 8,
        doSobol : bool = False,
        sobolSamples : int = 2048,
        numFolds : int = 9,
        numRepeats : int = 3):
    
    ##### Get input data
    data_files = glob.glob(str(inputDir / "*.nc"))
    
    # Input data
    inputs = {name : [] for name in inputFields}
    inputs = ml_utils.read_data(data_files, inputs, with_names = True, with_coords = False)

    # Proprocessing
    if "B0angle" in inputs:
        transf = np.array(inputs["B0angle"])
        inputs["B0angle"] = np.abs(transf - 90.0) 
    if "backgroundDensity" in inputs:
        transf = np.array(inputs["backgroundDensity"])
        inputs["backgroundDensity"] = np.log10(transf)
    if "beamFraction" in inputs:
        transf = np.array(inputs["beamFraction"])
        inputs["beamFraction"] = np.log10(transf) 

    ##### Format
    inputData_arr = np.array([np.array(inputs[f]) for f in inputFields]).T
    input_ptt = torch.from_numpy(inputData_arr)

    ##### Get output data
    output_ds = xr.open_dataset(outputFile, engine="netcdf4")
    outputFields = list(output_ds.data_vars) if outputFields is None or len(outputFields) < 1 else outputFields
    if "index" in outputFields:
        outputFields.remove("index")
    all_results = []
    for outputField in outputFields:
        sim_ids = [int(id) for id in output_ds.coords["index"]]
        sorted_idx = np.array(sim_ids).argsort()
        feature_values = output_ds.variables[outputField].values.astype(float)
        outputData_list = [np.array([float(feature_values[i]) for i in sorted_idx])]
    
        outputData_arr = np.array(outputData_list).T
        output_ptt = torch.from_numpy(outputData_arr)
        all_results.append(
            run_single_output_GP(
                input_ptt,
                output_ptt,
                inputFields,
                outputField,
                saveFolder,
                models,
                f"{name}_{outputField}",
                plot,
                doPCA,
                pcaComponents,
                doSobol,
                sobolSamples,
                numFolds,
                numRepeats
        ))

    # Write all results to CSV
    saveFilepath = saveFolder / "all_single_GP_results.csv"
    if os.path.exists(saveFilepath):
        os.remove(saveFilepath)
    for result in all_results:
        result.to_csv(saveFilepath, mode='a', header = not os.path.exists(saveFilepath))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--inputDir",
        action="store",
        help="Directory containing netCDF files of simulation data from which to source inputs.",
        required = True,
        type=Path
    )
    parser.add_argument(
        "--inputFields",
        action="store",
        help="Fields to use as emulation inputs.",
        required = True,
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--outputFile",
        action="store",
        help="NetCDF file of output features.",
        required = True,
        type=Path
    )
    parser.add_argument(
        "--saveFolder",
        action="store",
        help="Filepath in which to save JSON output.",
        required = True,
        type=Path
    )
    parser.add_argument(
        "--outputFields",
        action="store",
        help="Fields to use as emulation outputs.",
        required = False,
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--models",
        action="store",
        help="Models to use. Defaults to all probabilistic emulators.",
        required = False,
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--modelFile",
        action="store",
        help="Filepath of an existing model to load, if desired.",
        required = False,
        type=Path
    )
    parser.add_argument(
        "--name",
        action="store",
        help="Name of the run.",
        required = False,
        type=str
    )
    parser.add_argument(
        "--pca",
        action="store_true",
        help="Perform PCA for dimensionality reduction.",
        required = False
    )
    parser.add_argument(
        "--pcaComponents",
        action="store",
        help="Number of PCA components for dimensionality reduction of the output.",
        required = False,
        type=int
    )
    parser.add_argument(
        "--sobol",
        action="store_true",
        help="Perform Sobol analysis for sensitivity analysis.",
        required = False
    )
    parser.add_argument(
        "--sobolSamples",
        action="store",
        help="Number of Sobol samples to use.",
        required = False,
        type=int
    )
    parser.add_argument(
        "--numFolds",
        action="store",
        help="Number of CV folds.",
        required = False,
        type=int
    )
    parser.add_argument(
        "--numRepeats",
        action="store",
        help="Number of CV repeats.",
        required = False,
        type=int
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plots the best fitting model and saves to saveFolder.",
        required = False
    )
    parser.add_argument(
        "--resultsFilepath",
        action="store",
        help="Filepath of collate results for plotting.",
        required = False,
        type=Path
    )
    parser.add_argument(
        "--plotResults",
        action="store_true",
        help="Plots the results of a batch run.",
        required = False
    )

    args = parser.parse_args()

    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.max_colwidth', None)

    if args.plotResults:
        saveFilepath = args.saveFolder / "all_single_GP_results.csv" if args.resultsFilepath is None else args.resultsFilepath
        plot_results(saveFilepath)
    else:
        harness_multiple_single_output_GPs(
            args.inputDir, 
            args.inputFields, 
            args.outputFile, 
            args.saveFolder, 
            args.outputFields,
            args.models, 
            args.name, 
            args.plot if args.plot is not None else False,
            args.pca if args.pca is not None else False, 
            args.pcaComponents if args.pcaComponents is not None else 8,
            args.sobol if args.sobol is not None else False,
            args.sobolSamples,
            args.numFolds if args.numFolds is not None else 9,
            args.numRepeats if args.numRepeats is not None else 3)