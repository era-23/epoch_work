import argparse
import glob
import os
import pprint
import random
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path
from inference.plotting import matrix_plot
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic
from SALib import ProblemSpec
from SALib.analyze import sobol
import SALib.sample as salsamp

def parse_commandLine_netCDFpaths(paths : list) -> dict:
    
    formattedPaths = dict()
    root_path = "/"

    for path in paths:
        s = path.split("/")
        field = s[-1]

        if len(s) == 1:
            group = root_path
        elif len(s) == 2:
            group = s[0]
        else:
            raise NotImplementedError("netCDF path length is too many groups deep")
        
        if group not in formattedPaths:
            formattedPaths[group] = []
        formattedPaths[group].append(field)

    return formattedPaths

def preprocess_input_data(inputData : dict, log_fields : list) -> tuple:

    inputData_asColVec = np.array(list(inputData.values())).T

    log_index = []
    for field in log_fields:
        log_index.append(list(inputData.keys()).index(field))

    log_transform = ColumnTransformer(
        transformers=[
            ('log', FunctionTransformer(np.log1p), log_index)
        ],
        verbose_feature_names_out=False,
        remainder='passthrough'
    )

    allIndices = list(range(0, len(inputData.keys())))

    scale = ColumnTransformer(
        transformers=[
            ('scale', StandardScaler(), allIndices)
        ],
        verbose_feature_names_out=False,
        remainder='passthrough'
    )
    preprocess = Pipeline(steps=[('log', log_transform), ('scale', scale)])
    return list(inputData.keys()), preprocess.fit_transform(inputData_asColVec)

def preprocess_output_data(outputData : list):
    outputData_shaped = np.array(outputData).reshape(-1, 1)
    return StandardScaler().fit(outputData_shaped).transform(outputData_shaped)

def untrained_GP(kernel : str) -> GaussianProcessRegressor:
    if kernel == 'RQ':
        k = 1.0 * RationalQuadratic(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    elif kernel == 'RBF':
        k = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    return GaussianProcessRegressor(k)

def regress_data(x_train : np.ndarray, y_train : np.ndarray, kernel : str) -> GaussianProcessRegressor:
    gp = untrained_GP(kernel)
    return gp.fit(x_train, y_train)
    
def display_matrix_plots(inputs : dict, normalisedInputs : np.ndarray, outputs: dict, normalisedOutputs : dict):

    for outName, outData in outputs.items():
        data = [outData] + list(inputs.values())
        labels = [outName] + list(inputs.keys())
        matrix_plot(data, labels, show=False)
        plt.title("Raw data")
        plt.tight_layout()
        plt.show()

        data = list(np.array(normalisedOutputs[outName]).T) + list(normalisedInputs.T)
        labels = [outName] + list(inputs.keys())
        matrix_plot(data, labels, show=False)
        plt.title("Normalised data")
        plt.tight_layout()
        plt.show()

    plt.close()

def sobol_analysis(inputs : dict, outputs : dict, normalisedInputData : np.ndarray = None, normalisedOutput : dict = None, plotGP = True):
    
    normalisedInputData = normalisedInputData if normalisedInputData is not None else preprocess_input_data(inputs)[1]

    for outName, outData in outputs.items():

        normalisedOutputData = normalisedOutput[outName] if normalisedOutput is not None else np.array([n[0] for n in preprocess_output_data(outData)])
        
        fit = regress_data(normalisedInputData, normalisedOutputData, 'RQ')
        print(fit.kernel_)

        # SALib SOBOL indices
        sp = ProblemSpec({
            'num_vars': len(inputs),
            'names': list(inputs.keys()),
            'bounds': [[np.min(column), np.max(column)] for column in normalisedInputData.T]
        })
        test_values = salsamp.sobol.sample(sp, int(2**16))
        y_prediction, y_std = fit.predict(test_values, return_std=True)

        if plotGP:
            if len(inputs.keys()) == 2:
                keys = [k for k in inputs.keys()]
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.scatter(normalisedInputData[:,0], normalisedInputData[:,1], normalisedOutputData)
                ax.set_xlabel(keys[0])
                ax.set_ylabel(keys[1])
                ax.set_zlabel(outName)
                ax.set_title("Training data")
                plt.show()

            zScore = abs(norm.ppf(0.975))
            column = 0
            plotSamples = salsamp.sobol.sample(sp, int(2**7))
            plotPreds, plotStds = fit.predict(plotSamples, return_std=True)
            for inputName in inputs.keys():
                testData_unique = dict(zip(plotSamples[:,column], zip(plotPreds, plotStds)))
                testPoints = sorted(testData_unique.keys())
                testPredictions = np.array([testData_unique[p][0] for p in testPoints])
                testStds = np.array([testData_unique[p][1] for p in testPoints])
                plt.figure(figsize=(12,8))
                plt.scatter(normalisedInputData[:,column], normalisedOutputData, label = inputName, color="tab:blue")
                plt.plot(testPoints, testPredictions, label="Mean prediction", color ="tab:red")
                plt.fill_between(
                    testPoints,
                    testPredictions - (zScore * testStds),
                    testPredictions + (zScore * testStds),
                    color="tab:red",
                    alpha=0.3,
                    label=r"95% confidence interval",
                )
                plt.legend()
                plt.title(f"{inputName} vs {outName}")
                plt.xlabel(f"{inputName} [normalised]")
                plt.ylabel(f"{outName} [normalised]")
                plt.show()
                column += 1

        print(f"gp.predict() for {outName} -- y_std: {y_std}")
        sobol_indices = sobol.analyze(sp, y_prediction, print_to_console=True)
        print(f"Sobol indices for output {outName}:")
        print(sobol_indices)
        print(f"Sum of SOBOL indices: ST = {np.sum(sobol_indices['ST'])}, S1 = {np.sum(sobol_indices['S1'])}, abs(S1) = {np.sum(abs(sobol_indices['S1']))} S2 = {np.nansum(sobol_indices['S2'])}, abs(S2) = {np.nansum(abs(sobol_indices['S2']))}")
        plt.rcParams["figure.figsize"] = (12,6)
        sobol_indices.plot()
        plt.title(f"Output: {outName}")
        plt.tight_layout()
        plt.show()
        plt.close()

def evaluate_model(normalisedInputData : np.ndarray, normalisedOutputs : dict, kernels : list, crossValidate : bool = True, folds : int = 5):
    
    scores = {k : [] for k in kernels}
    stds = {k : [] for k in kernels}
    for outName, outData in normalisedOutputs.items():

        if crossValidate:
            for kernel in kernels:
                model = untrained_GP(kernel)
                all_cv_scores = cross_val_score(model, normalisedInputData, outData, cv=folds)
                print(f"R^2 for GP ({kernel} kernel) regression of {outName} across {folds}-fold cross-validation: {all_cv_scores.mean()} (std: {all_cv_scores.std()})")
                scores[kernel].append(all_cv_scores.mean())
                stds[kernel].append(all_cv_scores.std())
        else:
            x_train, x_test, y_train, y_test = train_test_split(normalisedInputData, outData, test_size=0.25)
            
            # Retrain without test data
            for kernel in kernels:
                fit = regress_data(x_train, y_train, kernel)
                r_squared = fit.score(x_test, y_test)
                print(f"R^2 for GP ({kernel} kernel) regression of {outName}: {r_squared}")
                scores[kernel].append(r_squared)

    outputNames = list(normalisedOutputs.keys())
    x = np.arange(len(outputNames))  # the label locations
    width = 1.0/(len(kernels) + 1)  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    fig.set_figheight(6)
    fig.set_figwidth(12)
    for kernel, r2 in scores.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, r2, width, label=kernel)
        if crossValidate:
            zScore = abs(norm.ppf(0.975))
            ax.errorbar(x + offset, r2, yerr=zScore*np.array(stds[kernel]), fmt=' ', color='r')
        ax.bar_label(rects, padding=3, fmt="%.3f")
        multiplier += 1
    ax.set_ylabel(r'$R^2$')
    ax.set_title('R-squared scores by GP/output and kernel')
    ax.set_xticks(x + ((len(kernels)-1)*0.5*width), outputNames)
    ax.legend(loc='upper left')
    ax.set_ylim(0.0, 1.0)
    plt.tight_layout()
    plt.show()
    plt.close()

def process_simulations(
        directory : Path, 
        inputFields : list, 
        outputFields : list,
        matrixPlot : bool = False,
        sobol : bool = False,
        evaluateModels : bool = False,
        ):
    """
    Processes a batch of simulations:
    - Iterates nc files in 'directory'
    - Gets data from inputFields and regresses a GP against outputFields

    Note: Only works for attributes <= one group deep in the netcdf structure, i.e. a top level attribute works, an attribute in a group works, an attribute in a subgroup in a group is not implemented

    ### Parameters:
        directory: Path -- Directory of simulation outputs, containing .nc files each with simulation data
        inputFields: dict -- List of paths of netcdf group name and field names to use for GP input.
        outputFields: dict -- List of paths of netcdf group name and field names to use for GP  output.
    """

    output_files = glob.glob(str(directory / "*.nc")) 

    # Input data
    inputs = {inp : [] for inp in inputFields}

    # Output data
    outputs = {outp : [] for outp in outputFields}

    for simulation in output_files:

        data = xr.open_datatree(
            simulation,
            engine="netcdf4"
        )

        for fieldPath in inputs.keys():
            s = fieldPath.split("/")
            group = "/" if len(s) == 1 else s[0]
            fieldName = s[-1]
            inputs[fieldPath].append(data[group].attrs[fieldName])

        for fieldPath, fieldData in outputs.items():
            s = fieldPath.split("/")
            group = "/" if len(s) == 1 else s[0]
            fieldName = s[-1]
            fieldData.append(data[group].attrs[fieldName])

    # DEBUGGING
    # for outName, outData in outputs.items():
    #     for inName, inData in inputs.items():
    #         plt.scatter(inData, outData)
    #         plt.xlabel(inName)
    #         plt.ylabel(outName)
    #         if inName == "backgroundDensity": 
    #             plt.xscale("log")
    #         plt.show()

    # Preprocess inputs (multiple returned as column vector)
    inNames, normalisedInputData = preprocess_input_data(inputs, ["backgroundDensity"])
    #normalisedInputData = preprocess_input_data(inputs, [])

    # Preprocess output (singular, for now -- 1 GP each)
    normalisedOutputs = {}
    for outputName, outputData in outputs.items():
        normalisedOutputs[outputName] = np.array([n[0] for n in preprocess_output_data(outputData)])

    # DEBUGGING
    # for outName in outputs.keys():
    #     for i in range(normalisedInputData.shape[1]):
    #         plt.scatter(normalisedInputData[:,i], normalisedOutputs[outName])
    #         plt.xlabel(inNames[i])
    #         plt.ylabel(outName)
    #         plt.show()

    # Display matrix plots
    if matrixPlot:
        display_matrix_plots(inputs, normalisedInputData, outputs, normalisedOutputs)

    # Perform SOBOL analysis
    if sobol:
        sobol_analysis(inputs, outputs, normalisedInputData, normalisedOutputs)

    # Evaluate model
    if evaluateModels:
        kernels = ["RBF", "RQ"]
        evaluate_model(normalisedInputData, normalisedOutputs, kernels)

def classify(irbDir : Path, nullDir : Path):

    ###### IRB

    output_files = glob.glob(str(irbDir / "*.nc")) 

    # Input data
    input_backgroundIonEnergyDensity_delta = []
    input_electricFieldEnergyDensity_delta = []
    input_magneticFieldEnergyDensity_delta = []
    input_E_peakTkSpectralPower = []
    input_E_peakTkSpectralPowerRatio = []
    input_B_peakTkSpectralPower = []
    input_B_peakTkSpectralPowerRatio = []

    for simulation in output_files:

        energyData = xr.open_dataset(
            simulation,
            engine="netcdf4",
            group="Energy"
        )
        input_backgroundIonEnergyDensity_delta.append(energyData.backgroundIonEnergyDensity_delta)
        input_electricFieldEnergyDensity_delta.append(energyData.electricFieldEnergyDensity_delta)
        input_magneticFieldEnergyDensity_delta.append(energyData.magneticFieldEnergyDensity_delta)
        del(energyData)

        eFieldData = xr.open_dataset(
            simulation,
            engine="netcdf4",
            group="Electric_Field_Ex"
        )
        input_E_peakTkSpectralPower.append(eFieldData.peakTkSpectralPower)
        input_E_peakTkSpectralPowerRatio.append(eFieldData.peakTkSpectralPowerRatio)
        del(eFieldData)

        bFieldData = xr.open_dataset(
            simulation,
            engine="netcdf4",
            group="Magnetic_Field_Bz"
        )
        input_B_peakTkSpectralPower.append(bFieldData.peakTkSpectralPower)
        input_B_peakTkSpectralPowerRatio.append(bFieldData.peakTkSpectralPowerRatio)
        del(bFieldData)
    
    inputs = {
        "bkgd delta KE" : input_backgroundIonEnergyDensity_delta,
        "E-field delta E" : input_electricFieldEnergyDensity_delta,
        "B-field delta E" : input_magneticFieldEnergyDensity_delta,
        "Ex peak power" : input_E_peakTkSpectralPower,
        "Ex power ratio" : input_E_peakTkSpectralPowerRatio,
        "Bz peak power" : input_B_peakTkSpectralPower,
        "Bz power ratio" : input_B_peakTkSpectralPowerRatio
    }

    # Preprocess inputs (multiple returned as column vector)
    inNames, irbNormalisedInputData = preprocess_input_data(inputs, [])
    irbOutputData = np.ones((1, irbNormalisedInputData.size[1]))

    ###### null

    output_files = glob.glob(str(nullDir / "*.nc")) 

    # Input data
    input_backgroundIonEnergyDensity_delta = []
    input_electricFieldEnergyDensity_delta = []
    input_magneticFieldEnergyDensity_delta = []
    input_E_peakTkSpectralPower = []
    input_E_peakTkSpectralPowerRatio = []
    input_B_peakTkSpectralPower = []
    input_B_peakTkSpectralPowerRatio = []

    for simulation in output_files:

        energyData = xr.open_dataset(
            simulation,
            engine="netcdf4",
            group="Energy"
        )
        input_backgroundIonEnergyDensity_delta.append(energyData.backgroundIonEnergyDensity_delta)
        input_electricFieldEnergyDensity_delta.append(energyData.electricFieldEnergyDensity_delta)
        input_magneticFieldEnergyDensity_delta.append(energyData.magneticFieldEnergyDensity_delta)
        del(energyData)

        eFieldData = xr.open_dataset(
            simulation,
            engine="netcdf4",
            group="Electric_Field_Ex"
        )
        input_E_peakTkSpectralPower.append(eFieldData.peakTkSpectralPower)
        input_E_peakTkSpectralPowerRatio.append(eFieldData.peakTkSpectralPowerRatio)
        del(eFieldData)

        bFieldData = xr.open_dataset(
            simulation,
            engine="netcdf4",
            group="Magnetic_Field_Bz"
        )
        input_B_peakTkSpectralPower.append(bFieldData.peakTkSpectralPower)
        input_B_peakTkSpectralPowerRatio.append(bFieldData.peakTkSpectralPowerRatio)
        del(bFieldData)
    
    inputs = {
        "bkgd delta KE" : input_backgroundIonEnergyDensity_delta,
        "E-field delta E" : input_electricFieldEnergyDensity_delta,
        "B-field delta E" : input_magneticFieldEnergyDensity_delta,
        "Ex peak power" : input_E_peakTkSpectralPower,
        "Ex power ratio" : input_E_peakTkSpectralPowerRatio,
        "Bz peak power" : input_B_peakTkSpectralPower,
        "Bz power ratio" : input_B_peakTkSpectralPowerRatio
    }

    # Preprocess inputs (multiple returned as column vector)
    inNames, nullNormalisedInputData = preprocess_input_data(inputs, [])
    nullOutputData = np.zeros((1, nullNormalisedInputData.size[1]))

    # Classify
    kernel = 1.0 * RBF([1.0])
    #gpc_rbf_isotropic = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    

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
        "--inputFields",
        action="store",
        help="Directory containing netCDF files of simulation output.",
        required = True,
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--outputFields",
        action="store",
        help="Directory containing netCDF files of simulation output.",
        required = True,
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test regression models.",
        required = False
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
        "--evaluateModel",
        action="store_true",
        help="Evaluate regression models.",
        required = False
    )
    parser.add_argument(
        "--classify",
        action="store_true",
        help="Classify null vs IRB cases.",
        required = False
    )
    parser.add_argument(
        "--irbDir",
        action="store",
        help="Directory containing netCDF files of simulation output with IRB for classification.",
        required = False,
        type=Path
    )
    parser.add_argument(
        "--nullDir",
        action="store",
        help="Directory containing netCDF files of simulation output without IRB (null) for classification.",
        required = False,
        type=Path
    )

    args = parser.parse_args()

    if args.test:
        process_simulations(args.dir, args.inputFields, args.outputFields, args.matrixPlot, args.sobol, args.evaluateModel)
    if args.classify:
        classify(args.irbDir, args.nullDir)
