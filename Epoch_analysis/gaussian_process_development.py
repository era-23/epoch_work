import argparse
import glob
import os
import pprint
import random
import typing
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
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, WhiteKernel, ConstantKernel
from SALib import ProblemSpec
from SALib.analyze import sobol
import SALib.sample as salsamp

fieldNameToText = {
    "Energy/electricFieldEnergyDensity_delta" : "E_deltaEnergy",
    "Energy/magneticFieldEnergyDensity_delta" : "B_deltaEnergy", 
    "Energy/backgroundIonEnergyDensity_delta" : "bkgdIon_deltaEnergy", 
    "Energy/electronEnergyDensity_delta" : "e_deltaEnergy",
    "Energy/fastIonEnergyDensity_max" : "fastIon_maxEnergy", 
    "Energy/fastIonEnergyDensity_timeMax" : "fastIon_timeMaxEnergy", 
    "Energy/fastIonEnergyDensity_min" : "fastIon_minEnergy", 
    "Energy/fastIonEnergyDensity_timeMin" : "fastIon_timeMinEnergy", 
    "Energy/fastIonEnergyDensity_delta" : "fastIon_deltaEnergy",
    "Magnetic_Field_Bz/totalMagnitude" : "Bz_totalPower", 
    "Magnetic_Field_Bz/meanMagnitude" : "Bz_meanPower", 
    "Magnetic_Field_Bz/totalDelta" : "Bz_deltaTotalPower", 
    "Magnetic_Field_Bz/meanDelta" : "Bz_deltaMeanPower", 
    "Magnetic_Field_Bz/peakTkSpectralPower" : "Bz_peakTkPower", 
    "Magnetic_Field_Bz/meanTkSpectralPower" : "Bz_meanTkPower", 
    "Magnetic_Field_Bz/peakTkSpectralPowerRatio" : "Bz_tkPowerRatio", 
    "Electric_Field_Ex/totalMagnitude" : "Ex_totalPower", 
    "Electric_Field_Ex/meanMagnitude" : "Ex_meanPower", 
    "Electric_Field_Ex/totalDelta" : "Ex_deltaTotalPower", 
    "Electric_Field_Ex/meanDelta" : "Ex_deltaMeanPower", 
    "Electric_Field_Ex/peakTkSpectralPower" : "Ex_peakTkPower", 
    "Electric_Field_Ex/meanTkSpectralPower" : "Ex_meanTkPower", 
    "Electric_Field_Ex/peakTkSpectralPowerRatio" : "Ex_tkPowerRatio",
    "B0strength" : "B0strength", 
    "B0angle" : "B0angle", 
    "backgroundDensity" : "backgroundDensity", 
    "beamFraction" : "beamFraction",
}

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

def preprocess_input_data(inputData : dict, log_fields : list = []) -> tuple:

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

def untrained_GP(kernel : str, type : str):
    ls_upper = 1e3
    ls_lower = 1e-3
    if kernel == 'RQ':
        k = ConstantKernel(constant_value=1.0, constant_value_bounds=(ls_lower * 1e-2, ls_upper * 1e2)) * RationalQuadratic(length_scale=1.0, length_scale_bounds=(ls_lower, ls_upper)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(ls_lower, ls_upper))
    elif kernel == 'RBF':
        k = ConstantKernel(constant_value=1.0, constant_value_bounds=(ls_lower * 1e-2, ls_upper * 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(ls_lower, ls_upper)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(ls_lower, ls_upper))
    
    if type == "regress":
        return GaussianProcessRegressor(k, n_restarts_optimizer=10)
    elif type == "classify":
        return GaussianProcessClassifier(k, n_restarts_optimizer=0)
    else:
        raise NotImplementedError("type must be one of regress or classify")

def train_data(x_train : np.ndarray, y_train : np.ndarray, kernel : str, problemType : str):
    gp = untrained_GP(kernel, problemType)
    return gp.fit(x_train, y_train)

def read_data(dataFiles, data_dict : dict) -> dict:
    
    for simulation in dataFiles:

        data = xr.open_datatree(
            simulation,
            engine="netcdf4"
        )

        for fieldPath in data_dict.keys():
            s = fieldPath.split("/")
            group = "/" if len(s) == 1 else s[0]
            fieldName = s[-1]
            data_dict[fieldPath].append(data[group].attrs[fieldName])

    return data_dict
    
def display_matrix_plots(inputs : dict, normalisedInputs : np.ndarray, outputs: dict, normalisedOutputs : dict = {}):

    for outName, outData in outputs.items():
        data = [outData] + list(inputs.values())
        labels = [fieldNameToText[outName]] + [fieldNameToText[i] for i in inputs.keys()]
        matrix_plot(data, labels, show=False)
        plt.title("Raw data")
        plt.tight_layout()
        plt.show()

        if normalisedOutputs:
            data = list(normalisedInputs.T)
            data.insert(0, list(normalisedOutputs[outName]))
            labels = [fieldNameToText[outName]] + [fieldNameToText[i] for i in inputs.keys()]
            matrix_plot(data, labels, show=False)
            plt.title("Normalised data")
            plt.tight_layout()
            plt.show()

    plt.close()

def plot_models(models : dict, inputNames : list, normalisedInputData : np.ndarray, normalisedOutputs : dict, problemType : str):
    
    for outputName, model in models.items():

        # 3D plot if there are only 2 input dimensions
        if len(inputNames) == 2:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(normalisedInputData[:,0], normalisedInputData[:,1], normalisedOutputs[outputName])
            ax.set_xlabel(inputNames[0])
            ax.set_ylabel(inputNames[1])
            ax.set_zlabel(outputName)
            ax.set_title(f"Training data for output {outputName}")
            plt.show()

        # Sample values of each input dim keeping others fixed at their mean value
        zScore = abs(norm.ppf(0.975))
        numSamples = 1000
        for i in range(len(inputNames)):
            inputName = fieldNameToText[inputNames[i]]
            sampleColumn = normalisedInputData[:,i]
            samples = np.linspace(sampleColumn.min(), sampleColumn.max(), numSamples)

            allDatapoints = [[] for _ in inputNames]
            trainingDistancesForAlphas = np.zeros(len(sampleColumn))
            for i2 in range(len(inputNames)):
                if i2 == i:
                    allDatapoints[i2] = samples
                else:
                    mean = normalisedInputData[:,i2].mean()
                    allDatapoints[i2] = np.ones(numSamples) * mean
                    # Sum geometric distance from non-x-axis dim means to calculate how relevant each point is for this view
                    trainingDistancesForAlphas += (normalisedInputData[:,i2] - mean)**2

            # Columnise
            allDatapoints = np.array(allDatapoints).T

            # Sample
            y_preds = []
            if problemType == "regress":
                y_preds, y_std = model.predict(allDatapoints, return_std=True)
                plt.fill_between(
                    samples,
                    y_preds - (zScore * y_std),
                    y_preds + (zScore * y_std),
                    color="tab:red",
                    alpha=0.3,
                    label=r"95% confidence interval"
                )
            elif problemType == "classify":
                y_preds = model.predict_proba(allDatapoints)[:,1]
            else:
                raise NotImplementedError("problemType must be regress or classify")

            # Calculate alphas
            trainingDistancesForAlphas = np.sqrt(trainingDistancesForAlphas)
            normDistance = (trainingDistancesForAlphas - np.min(trainingDistancesForAlphas)) / (np.max(trainingDistancesForAlphas) - np.min(trainingDistancesForAlphas))
            normAlpha = 1.0 - normDistance # Closer is better therefore higher alpha

            # Plot
            plt.plot(samples, y_preds, label="Mean prediction", color="tab:red")
            plt.scatter(sampleColumn, normalisedOutputs[outputName], label="Training data", alpha=normAlpha)
            plt.xlabel(inputName)
            plt.ylabel(outputName)
            plt.legend()
            #plt.tight_layout()
            plt.title(f"GP predictions for {outputName}, all other input dimensions fixed to mean value", wrap=True)
            plt.show()

def sobol_analysis(models : dict, inputNames : list, normalisedInputData : np.ndarray, problemType : str):

    formattedNames = [fieldNameToText[i] for i in inputNames]
    # SALib SOBOL indices
    sp = ProblemSpec({
        'num_vars': len(formattedNames),
        'names': formattedNames,
        'bounds': [[np.min(column), np.max(column)] for column in normalisedInputData.T]
    })
    test_values = salsamp.sobol.sample(sp, int(2**16))

    for outName, model in models.items():
        print(f"SOBOL analysing model {model.kernel_} for {outName}....")

        if problemType == "regress":
            y_prediction, y_std = model.predict(test_values, return_std=True)
            print(f"gp.predict() for {outName} -- y_std: {y_std}")
        elif problemType == "classify":
            y_prediction = model.predict_proba(test_values)[:,1]
            print(f"gp.predict() for {outName}")

        
        sobol_indices = sobol.analyze(sp, y_prediction, print_to_console=True)
        print(f"Sobol indices for output {outName}:")
        print(sobol_indices)
        print(f"Sum of SOBOL indices: ST = {np.sum(sobol_indices['ST'])}, S1 = {np.sum(sobol_indices['S1'])}, abs(S1) = {np.sum(abs(sobol_indices['S1']))} S2 = {np.nansum(sobol_indices['S2'])}, abs(S2) = {np.nansum(abs(sobol_indices['S2']))}")
        plt.rcParams["figure.figsize"] = (12,10)
        sobol_indices.plot()
        plt.title(f"Output: {outName}")
        #plt.tight_layout()
        plt.show()
        plt.close()

def evaluate_model(normalisedInputData : np.ndarray, normalisedOutputs : dict, kernels : list, problemType : str, crossValidate : bool = True, folds : int = 5):
    
    scores = {k : [] for k in kernels}
    stds = {k : [] for k in kernels}
    for outName, outData in normalisedOutputs.items():

        if crossValidate:
            for kernel in kernels:
                model = untrained_GP(kernel, problemType)
                all_cv_scores = cross_val_score(model, normalisedInputData, outData, cv=folds)
                print(f"R^2 for GP ({kernel} kernel) regression of {outName} across {folds}-fold cross-validation: {all_cv_scores.mean()} (std: {all_cv_scores.std()})")
                scores[kernel].append(all_cv_scores.mean())
                stds[kernel].append(all_cv_scores.std())
        else:
            x_train, x_test, y_train, y_test = train_test_split(normalisedInputData, outData, test_size=0.25)
            
            # Retrain without test data
            for kernel in kernels:
                fit = train_data(x_train, y_train, kernel, problemType)
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

def regress_simulations(
        directory : Path, 
        inputFields : list, 
        outputFields : list,
        logFields : list = [],
        matrixPlot : bool = False,
        sobol : bool = False,
        evaluateModels : bool = False,
        plotModels : bool = False
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
    inputs = read_data(output_files, inputs)

    # Output data
    outputs = {outp : [] for outp in outputFields}
    outputs = read_data(output_files, outputs)

    # Preprocess inputs (multiple returned as column vector)
    inNames, normalisedInputData = preprocess_input_data(inputs, list(set(logFields).intersection(inputFields)))

    # Preprocess output (singular, for now -- 1 GP each)
    normalisedOutputs = {}
    for outputName, outputData in outputs.items():
        normalisedOutputs[outputName] = np.array([n[0] for n in preprocess_output_data(outputData)])

    # Display matrix plots
    if matrixPlot:
        display_matrix_plots(inputs, normalisedInputData, outputs, normalisedOutputs)

    # Train individual models for each output
    models = dict.fromkeys(normalisedOutputs.keys())
    for outputName in models.keys():
        models[outputName] = train_data(normalisedInputData, normalisedOutputs[outputName], "RQ", "regress")

    if plotModels:
        plot_models(models, inNames, normalisedInputData, normalisedOutputs, "regress")

    # Perform SOBOL analysis
    if sobol:
        sobol_analysis(models, inNames, normalisedInputData, "regress")

    # Evaluate model
    if evaluateModels:
        kernels = ["RBF", "RQ"]
        evaluate_model(normalisedInputData, normalisedOutputs, kernels, "regress")

def classify_simulations(
        irbDir : Path, 
        nullDir : Path, 
        fields : list, 
        logFields : list = [],
        matrixPlot : bool = False,
        sobol : bool = False,
        evaluateModels : bool = False,
        plotModels : bool = False):

    ###### IRB
    data_files = glob.glob(str(irbDir / "*.nc")) 
    # Input data
    inputs = {f : [] for f in fields}
    inputs = read_data(data_files, inputs)

    # Initialise outputs
    # IRB == 1
    normalisedOutputs = {"fastIonSimulation" : np.ones(len(data_files))}

    ###### Null
    data_files = glob.glob(str(nullDir / "*.nc")) 
    # Input data
    inputs = read_data(data_files, inputs)

    # Append output class
    # null == 0
    normalisedOutputs["fastIonSimulation"] = np.concatenate((normalisedOutputs["fastIonSimulation"], np.zeros(len(data_files))), axis=0)

    # Preprocess inputs (multiple returned as column vector)
    inNames, normalisedInputData = preprocess_input_data(inputs, list(set(logFields).intersection(fields)))

    # Display matrix plots
    if matrixPlot:
        display_matrix_plots(inputs, normalisedInputData, normalisedOutputs)

    model = {}
    model["fastIonSimulation"] = train_data(normalisedInputData, normalisedOutputs["fastIonSimulation"], "RQ", "classify")

    if plotModels:
        plot_models(model, inNames, normalisedInputData, normalisedOutputs, "classify")

    # Perform SOBOL analysis
    if sobol:
        sobol_analysis(model, inNames, normalisedInputData, "classify")

    # Evaluate model
    if evaluateModels:
        kernels = ["RBF", "RQ"]
        evaluate_model(normalisedInputData, normalisedOutputs, kernels, "classify")    

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
        "--regress",
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

    if args.regress:
        regress_simulations(args.dir, args.inputFields, args.outputFields, args.logFields, args.matrixPlot, args.sobol, args.evaluate, args.plotModels)
    if args.classify:
        classify_simulations(args.irbDir, args.nullDir, args.inputFields, args.logFields, args.matrixPlot, args.sobol, args.evaluate, args.plotModels)
