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
warnings.filterwarnings("ignore")
from autoemulate.simulations.projectile import Projectile
from autoemulate import AutoEmulate
from autoemulate.transforms import StandardizeTransform, PCATransform
from autoemulate.core.sensitivity_analysis import SensitivityAnalysis

def demo():
    projectile = Projectile(log_level="error")
    n_samples = 500
    x = projectile.sample_inputs(n_samples).float()
    y, _ = projectile.forward_batch(x)
    y = y.float()

    print(x.shape)
    print(y.shape)

    print(AutoEmulate.list_emulators())

    # Run AutoEmulate with default settings
    ae = AutoEmulate(x, y, only_probabilistic = True, log_level="progress_bar")

    print(ae.summarise())

    best = ae.best_result()
    print("Model with id: ", best.id, " performed best: ", best.model_name)

    ae.plot(best, fname="best_model_plot.png")

    ae.plot(best, input_ranges={0: (0, 4), 1: (200, 500)}, output_ranges={0: (0, 10)})

    print(best.model.predict(x[:10]))

def autoEmulate(
        inputDir : Path, 
        inputFields : list, 
        outputFile : Path, 
        saveFolder : Path, 
        models : list = None, 
        modelFile : Path = None, 
        name : str = None,
        doPCA : bool = False,
        pcaComponents : int = 8,
        doSobol : bool = False):

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

    ##### Get output data
    output_ds = xr.open_dataset(outputFile, engine="netcdf4")
    sim_ids = [int(id) for id in output_ds.coords["index"]]
    sorted_idx = np.array(sim_ids).argsort()
    outputData_list = []
    outputData_names = []
    for feature in output_ds.variables:
        feature_values = output_ds.variables[feature].values.astype(float)
        if feature != "index" and len(set(feature_values)) > 3 and not np.isnan(feature_values).any():
            outputData_names.append(feature)
            outputData_list.append(np.array([float(feature_values[i]) for i in sorted_idx]))
    
    ##### Format
    inputData_arr = np.array([np.array(inputs[f]) for f in inputFields]).T
    outputData_arr = np.array(outputData_list).T
    input_ptt = torch.from_numpy(inputData_arr)
    output_ptt = torch.from_numpy(outputData_arr)

    print(input_ptt.shape)
    print(output_ptt.shape)

    num_input_vars = inputData_arr.shape[1]
    print(f"Num input variables: {num_input_vars}")
    num_cases = inputData_arr.shape[0]
    print(f"Num cases: {num_cases}")
    num_output_vars = outputData_arr.shape[1]
    print(f"Num output variables: {num_output_vars}")

    # Run AutoEmulate with default settings
    # ae = AutoEmulate(input_ptt, output_ptt, models = ["GaussianProcess", "GaussianProcessCorrelated", "GaussianProcessMatern32", "GaussianProcessMatern52", "GaussianProcessRBF"], log_level="progress_bar")
    if modelFile is not None:
        print("Loading model from: ", modelFile)
        ae = AutoEmulate.load_model(modelFile)
        best = ae
    else:
        ae = AutoEmulate(
            input_ptt, 
            output_ptt, 
            models = models, 
            y_transforms_list = None if not doPCA else [[StandardizeTransform(), PCATransform(n_components = pcaComponents), StandardizeTransform()]],
            only_probabilistic = (models is None), 
            shuffle=False,
            n_splits = 9, 
            n_bootstraps = 3, 
            log_level="progress_bar"
        )
        best = ae.best_result()
        print("Model with id: ", best.id, " performed best: ", best.model_name)
        print(ae.summarise())

    # Save best model
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)
    best_result_filepath = ae.save(best, saveFolder, use_timestamp=True)
    print("Model and metadata saved to: ", best_result_filepath)

    # Save results
    results_json_filename = saveFolder / (f"AE_results_{name}.json" if name is not None else "autoEmulate.json")
    results = ae.summarise()
    results["pca"] = doPCA
    results["pcaComponents"] = pcaComponents
    with open(results_json_filename, "w", encoding='utf-8') as f:
        results.to_json(f, default_handler=str)

    if doSobol:

        model = best.model

        problem = {
            'num_vars': num_input_vars,
            'names': inputFields,
            'bounds': [[np.min(inputs[f]), np.max(inputs[f])] for f in inputFields],
            'output_names': outputData_names
        }

        si = SensitivityAnalysis(model, problem=problem)
        si_df = si.run(method='sobol', n_samples=2048)
        si.plot_sobol(si_df, index='S1', fname=saveFolder / (f"AE_SobolPlot_{name}_S1.png" if name is not None else "autoEmulateSobolS1.png"))
        si.plot_sobol(si_df, index='S2', fname=saveFolder / (f"AE_SobolPlot_{name}_S2.png" if name is not None else "autoEmulateSobolS2.png"))
        si.plot_sobol(si_df, index='ST', fname=saveFolder / (f"AE_SobolPlot_{name}_ST.png" if name is not None else "autoEmulateSobolST.png"))
        si.plot_sa_heatmap(si_df, index='ST', cmap='coolwarm', normalize=False, figsize=(15,15), fname=saveFolder / (f"AE_SobolHeatmap_{name}.png" if name is not None else "autoEmulateSobolHeatmap.png"))
        n_parameters = 3
        top_parameters_sa = si.top_n_sobol_params(si_df, top_n=n_parameters)
        print(top_parameters_sa)

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

    args = parser.parse_args()

    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.max_colwidth', None)

    autoEmulate(
        args.inputDir, 
        args.inputFields, 
        args.outputFile, 
        args.saveFolder, 
        args.models, 
        args.modelFile, 
        args.name, 
        args.pca if args.pca is not None else False, 
        args.pcaComponents if args.pcaComponents is not None else 8,
        args.sobol if args.sobol is not None else False)
    # demo()