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
# from autoemulate.emulators.base import SklearnBackend
from autoemulate.transforms import StandardizeTransform, PCATransform
from autoemulate.core.sensitivity_analysis import SensitivityAnalysis
# from sklearn.decomposition import FactorAnalysis

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

def autoEmulateMultiOutput(
        inputDir : Path, 
        inputFields : list, 
        outputFile : Path, 
        saveFolder : Path, 
        outputFields : list = None,
        models : list = None, 
        modelFile : Path = None, 
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

    ##### Get output data
    # Beta: output filter
    output_ds = xr.open_dataset(outputFile, engine="netcdf4")
    sim_ids = [int(id) for id in output_ds.coords["index"]]
    sorted_idx = np.array(sim_ids).argsort()
    outputData_list = []
    outputData_names = []
    if outputFields:
        for feature in outputFields:
            feature_values = output_ds.variables[feature].values.astype(float)
            outputData_names.append(feature)
            outputData_list.append(np.array([float(feature_values[i]) for i in sorted_idx]))
    else:
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
            y_transforms_list = None if not doPCA else [[StandardizeTransform(), PCATransform(n_components = pcaComponents, cache_size = 1), StandardizeTransform()]],
            # y_transforms_list = None if not doPCA else [[StandardizeTransform(), FactorAnalysis(n_components = pcaComponents), StandardizeTransform()]],
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

    # Save best model
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)
    try:
        best_result_filepath = ae.save(best, saveFolder, use_timestamp=True)
        print("Model and metadata saved to: ", best_result_filepath)
    except:
        print(f"Unable to save model to {saveFolder}.")

    # Save results
    results_json_filename = saveFolder / (f"AE_results_{name}.json" if name is not None else "autoEmulate.json")
    results = ae.summarise()
    results["pca"] = doPCA
    results["pcaComponents"] = pcaComponents
    results["cvFolds"] = numFolds
    results["cvRepeats"] = numRepeats
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
        si_df = si.run(method='sobol', n_samples=sobolSamples if sobolSamples is not None else 2048)
        si_df.to_csv(saveFolder / f"AE_SobolAllData_{name}.csv")
        print(f"Sobol columns: {si_df.columns}")
        metrics = ['S1', 'S2', 'ST']
        by_inField = pd.DataFrame(columns=["index", "parameter", "value"])
        by_outField = pd.DataFrame(columns=["index", "output", "value"])
        for m in metrics:
            metric_rows = si_df.loc[si_df["index"] == m]
            for inF in inputFields:
                rows = metric_rows.loc[metric_rows["parameter"] == inF]
                v = np.max(rows['value'])
                by_inField = pd.concat([pd.DataFrame([[m, inF, v]], columns = by_inField.columns), by_inField], ignore_index=True)
            for outF in outputData_names:
                rows = metric_rows.loc[metric_rows['output'] == outF]
                v = np.max(rows['value'])
                by_outField = pd.concat([pd.DataFrame([[m, outF, v]], columns = by_outField.columns), by_outField], ignore_index=True)
        
        by_inField = by_inField.sort_values(by=["index", "value"], ascending=[True, False])
        by_inField.to_csv(saveFolder / f"AE_SobolTotalInputs_{name}.csv")
        print(by_inField)
        by_outField = by_outField.sort_values(by=["index", "value"], ascending=[True, False])
        print(by_outField)
        by_outField.to_csv(saveFolder / f"AE_SobolTotalOutputs_{name}.csv")
        top_outs = by_outField.loc[by_outField["index"] == "ST"]
        print(f"Top 10 output parameters by total Sobol index: {top_outs['output'].head(10).to_list()}")

        si.plot_sobol(si_df, index='S1', fname=saveFolder / (f"AE_SobolPlot_{name}_S1.png" if name is not None else "AE_SobolS1.png"))
        si.plot_sobol(si_df, index='S2', fname=saveFolder / (f"AE_SobolPlot_{name}_S2.png" if name is not None else "AE_SobolS2.png"))
        si.plot_sobol(si_df, index='ST', fname=saveFolder / (f"AE_SobolPlot_{name}_ST.png" if name is not None else "AE_SobolST.png"))
        si.plot_sa_heatmap(si_df, index='ST', cmap='coolwarm', normalize=False, figsize=(15,15), fname=saveFolder / (f"AE_SobolHeatmap_{name}.png" if name is not None else "autoEmulateSobolHeatmap.png"))
        # si.plot_morris(si_df, fname=saveFolder / (f"AE_MorrisPlot_{name}.png" if name is not None else "AE_Morris.png"))
        n_parameters = 3
        top_parameters_sa = si.top_n_sobol_params(si_df, top_n=n_parameters)
        print(top_parameters_sa)

def autoEmulateSingleOutput(
        inputDir : Path, 
        inputFields : list, 
        outputFile : Path, 
        saveFolder : Path, 
        outputFields : list = None,
        models : list = None, 
        modelFile : Path = None, 
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

    ##### Get output data
    # Beta: output filter
    output_ds = xr.open_dataset(outputFile, engine="netcdf4")
    sim_ids = [int(id) for id in output_ds.coords["index"]]
    sorted_idx = np.array(sim_ids).argsort()
    outputData_list = []
    outputData_names = []
    if outputFields:
        for feature in outputFields:
            feature_values = output_ds.variables[feature].values.astype(float)
            outputData_names.append(feature)
            outputData_list.append(np.array([float(feature_values[i]) for i in sorted_idx]))
    else:
        for feature in output_ds.variables:
            feature_values = output_ds.variables[feature].values.astype(float)
            if feature != "index" and len(set(feature_values)) > 3 and not np.isnan(feature_values).any():
                outputData_names.append(feature)
                outputData_list.append(np.array([float(feature_values[i]) for i in sorted_idx]))
    
    ##### Format
    inputData_arr = np.array([np.array(inputs[f]) for f in inputFields]).T
    outputData_arr = np.array(outputData_list).T
    input_ptt = torch.from_numpy(inputData_arr)

    num_input_vars = inputData_arr.shape[1]
    print(f"Num input variables: {num_input_vars}")
    num_cases = inputData_arr.shape[0]
    print(f"Num cases: {num_cases}")
    num_output_vars = outputData_arr.shape[1]
    print(f"Num output variables: {num_output_vars}")

    run_name = name
    allResults = {f : {} for f in outputData_names}
    # Run AutoEmulate with default settings    
    for i in range(num_output_vars):
        output_fieldName = outputData_names[i]
        name = run_name + "_" + output_fieldName
        output_ptt = torch.from_numpy(outputData_arr[:,i])
        if modelFile is not None:
            print("Loading model from: ", modelFile)
            ae = AutoEmulate.load_model(modelFile)
            best = ae
        else:
            ae = AutoEmulate(
                input_ptt, 
                output_ptt, 
                models = models, 
                y_transforms_list = None if not doPCA else [[StandardizeTransform(), PCATransform(n_components = pcaComponents, cache_size = 1), StandardizeTransform()]],
                # y_transforms_list = None if not doPCA else [[StandardizeTransform(), FactorAnalysis(n_components = pcaComponents), StandardizeTransform()]],
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
                ae.plot(best, fname=saveFolder / (f"AE_bestModel_{name}_.png" if name is not None else "AE_bestModel.png"))

        # Save best model
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)
        try:
            best_result_filepath = ae.save(best, saveFolder, use_timestamp=True)
            print("Model and metadata saved to: ", best_result_filepath)
        except:
            print(f"Unable to save model to {saveFolder}.")

        # Save results
        results_json_filename = saveFolder / (f"AE_results_{name}.json" if name is not None else "autoEmulate.json")
        results = ae.summarise()
        results["pca"] = doPCA
        results["pcaComponents"] = pcaComponents
        results["cvFolds"] = numFolds
        results["cvRepeats"] = numRepeats
        with open(results_json_filename, "w", encoding='utf-8') as f:
            results.to_json(f, default_handler=str)

        if doSobol:

            model = best.model

            problem = {
                'num_vars': num_input_vars,
                'names': inputFields,
                'bounds': [[np.min(inputs[f]), np.max(inputs[f])] for f in inputFields],
                'output_names': [output_fieldName]
            }

            si = SensitivityAnalysis(model, problem=problem)
            si_df = si.run(method='sobol', n_samples=sobolSamples if sobolSamples is not None else 2048)
            si_df.to_csv(saveFolder / f"AE_SobolAllData_{name}.csv")
            print(f"Sobol columns: {si_df.columns}")
            metrics = ['S1', 'S2', 'ST']
            by_inField = pd.DataFrame(columns=["index", "parameter", "value"])
            by_outField = pd.DataFrame(columns=["index", "output", "value"])
            for m in metrics:
                metric_rows = si_df.loc[si_df["index"] == m]
                for inF in inputFields:
                    rows = metric_rows.loc[metric_rows["parameter"] == inF]
                    v = np.max(rows['value'])
                    by_inField = pd.concat([pd.DataFrame([[m, inF, v]], columns = by_inField.columns), by_inField], ignore_index=True)
                v = np.max(rows['value'])
                by_outField = pd.concat([pd.DataFrame([[m, output_fieldName, v]], columns = by_outField.columns), by_outField], ignore_index=True)
            
            by_inField = by_inField.sort_values(by=["index", "value"], ascending=[True, False])
            by_inField.to_csv(saveFolder / f"AE_SobolTotalInputs_{name}.csv")
            print(by_inField)
            by_outField = by_outField.sort_values(by=["index", "value"], ascending=[True, False])
            print(by_outField)
            by_outField.to_csv(saveFolder / f"AE_SobolTotalOutputs_{name}.csv")
            top_outs = by_outField.loc[by_outField["index"] == "ST"]
            print(f"Top 10 output parameters by total Sobol index: {top_outs['output'].head(10).to_list()}")

            si.plot_sobol(si_df, index='S1', fname=saveFolder / (f"AE_SobolPlot_{name}_S1.png" if name is not None else "AE_SobolS1.png"))
            si.plot_sobol(si_df, index='S2', fname=saveFolder / (f"AE_SobolPlot_{name}_S2.png" if name is not None else "AE_SobolS2.png"))
            si.plot_sobol(si_df, index='ST', fname=saveFolder / (f"AE_SobolPlot_{name}_ST.png" if name is not None else "AE_SobolST.png"))
            si.plot_sa_heatmap(si_df, index='ST', cmap='coolwarm', normalize=False, figsize=(15,15), fname=saveFolder / (f"AE_SobolHeatmap_{name}.png" if name is not None else "autoEmulateSobolHeatmap.png"))
            # si.plot_morris(si_df, fname=saveFolder / (f"AE_MorrisPlot_{name}.png" if name is not None else "AE_Morris.png"))
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
        "--collateResults",
        action="store_true",
        help="Collate results files.",
        required = False
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plots the best fitting model and saves to saveFolder.",
        required = False
    )
    parser.add_argument(
        "--resultsFilePattern",
        action="store",
        help="Pattern of results files to collate, e.g. ae_pca_*.json .",
        required = False,
        type=str
    )

    args = parser.parse_args()

    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.max_colwidth', None)

    autoEmulateMultiOutput(
        args.inputDir, 
        args.inputFields, 
        args.outputFile, 
        args.saveFolder, 
        args.outputFields,
        args.models, 
        args.modelFile, 
        args.name, 
        args.plot if args.plot is not None else False,
        args.pca if args.pca is not None else False, 
        args.pcaComponents if args.pcaComponents is not None else 8,
        args.sobol if args.sobol is not None else False,
        args.sobolSamples,
        args.numFolds if args.numFolds is not None else 9,
        args.numRepeats if args.numRepeats is not None else 3)
    # demo()