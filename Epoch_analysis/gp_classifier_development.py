import argparse
import glob
import copy
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors 
import numpy as np
import xarray as xr
from pathlib import Path
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, WhiteKernel, ConstantKernel, Matern
from sklearn.exceptions import ConvergenceWarning
from SALib import ProblemSpec
from SALib.analyze import sobol
import SALib.sample as salsamp
import epoch_utils as e_utils
import gp_utils
import warnings
warnings.filterwarnings("error", category=ConvergenceWarning)

def preprocess_input_data(inputData : dict, log_fields : list = []) -> tuple:

    for field in log_fields:
        inputData[field] = np.log10(inputData[field])
    
    inputData_asColVec = np.array(list(inputData.values())).T

    allIndices = list(range(0, len(inputData.keys())))

    scale = ColumnTransformer(
        transformers=[
            ('scale', StandardScaler(), allIndices)
        ],
        verbose_feature_names_out=False,
        remainder='passthrough'
    )
    # preprocess = Pipeline(steps=[('log', log_transform), ('scale', scale)])
    preprocess = Pipeline(steps=[('scale', scale)])
    return list(inputData.keys()), preprocess.fit_transform(inputData_asColVec)

def preprocess_output_data(outputData : list, arcsinh = False):
    if arcsinh:
        outputData = np.arcsinh(outputData)
    outputData_shaped = np.array(outputData).reshape(-1, 1)
    return StandardScaler().fit(outputData_shaped).transform(outputData_shaped)

def untrained_GP(kernel : str, cv_lower = 1e-5, cv_upper = 1e5, ls_lower = 1e-5, ls_upper=1e5, a_lower = 1e-5, a_upper = 1e5, nl_lower = 1e-5, nl_upper = 1e5):
    if kernel == 'RQ':
        k = (ConstantKernel(constant_value=1.0, constant_value_bounds=(cv_lower, cv_upper)) * 
             RationalQuadratic(length_scale=1.0, length_scale_bounds=(ls_lower, ls_upper), alpha=1.0, alpha_bounds=(a_lower, a_upper)) +
             WhiteKernel(noise_level=1e-3, noise_level_bounds=(nl_lower, nl_upper)))
    elif kernel == 'RBF':
        k = (ConstantKernel(constant_value=1.0, constant_value_bounds=(cv_lower, cv_upper)) * 
             RBF(length_scale=1.0, length_scale_bounds=(ls_lower, ls_upper)) +
             WhiteKernel(noise_level=1e-3, noise_level_bounds=(nl_lower, nl_upper)))
    elif kernel == "Matern":
        k = (ConstantKernel(constant_value=1.0, constant_value_bounds=(cv_lower, cv_upper)) * 
             Matern(length_scale=1.0, length_scale_bounds=(ls_lower, ls_upper), nu=2.5) +
             WhiteKernel(noise_level=1e-3, noise_level_bounds=(nl_lower, nl_upper)))

    return GaussianProcessClassifier(k, n_restarts_optimizer=0)

def train_data(x_train : np.ndarray, y_train : np.ndarray, kernel : str):

    # Promote convergence warnings to errors
    warnings.filterwarnings("error", category=ConvergenceWarning)

    cv_lower = 1e-5
    cv_upper = 1e5
    ls_lower = 1e-5
    ls_upper = 1e5
    a_lower = 1e-5
    a_upper = 1e5
    nl_lower = 1e-4
    nl_upper = 1e1
    maxIter = 10
    gp = GaussianProcessClassifier()
    fit = GaussianProcessClassifier()
    prevGp = GaussianProcessClassifier()
    prevFit = GaussianProcessClassifier()
    usePreviousModel = True
    for i in range(maxIter):
        try:
            gp = untrained_GP(kernel, cv_lower, cv_upper, ls_lower, ls_upper, a_lower, a_upper, nl_lower, nl_upper)
            fit = gp.fit(x_train, y_train,)
        except ConvergenceWarning as e:
            print(repr(e))
            print(f"ConvergenceWarning encountered for {gp.kernel_}. Attempting to fit again with different bounds (iter_no = {i})....")
            if "constant_value" in repr(e):
                if "lower bound" in repr(e):
                    cv_lower *= 1e-1
                if "upper bound" in repr(e):
                    cv_upper *= 1e1
            elif "length_scale" in repr(e):
                if "lower bound" in repr(e):
                    ls_lower *= 1e-1
                if "upper bound" in repr(e):
                    ls_upper *= 1e1
            elif "alpha" in repr(e):
                if "lower bound" in repr(e):
                    a_lower *= 1e-1
                if "upper bound" in repr(e):
                    a_upper *= 1e1
            elif "noise_level" in repr(e):
                if "lower bound" in repr(e):
                    nl_lower *= 1e-1
                if "upper bound" in repr(e):
                    nl_upper *= 1e1

            if "ABNORMAL_TERMINATION_IN_LNSRCH" not in repr(e):
                prevGp = copy.deepcopy(gp)
                prevFit = copy.deepcopy(fit)
        else:
            usePreviousModel = False
            break
    if usePreviousModel:
        gp = prevGp
        fit = prevFit
    
    try:
        print(fit.kernel_)
    except AttributeError:
        print("Fitting failed.")
        return None, None

    bestParams = {
        "cv_lower" : cv_lower,
        "cv_upper" : cv_upper,
        "ls_lower" : ls_lower,
        "ls_upper" : ls_upper,
        "a_lower" : a_lower,
        "a_upper" : a_upper,
        "nl_lower" : nl_lower,
        "nl_upper" : nl_upper
    }
    # Restore default warning behaviour
    warnings.filterwarnings("default", category=ConvergenceWarning)
    return fit, bestParams

def read_data(dataFiles, data_dict : dict) -> dict:
    
    for simulation in dataFiles:

        data = xr.open_datatree(
            simulation,
            engine="netcdf4"
        )

        for fieldPath in data_dict.keys():
            s = fieldPath.split("/")
            group = "/" if len(s) == 1 else '/'.join(s[:-1])
            fieldName = s[-1]
            data_dict[fieldPath].append(data[group].attrs[fieldName])

    return data_dict
    
def sobol_analysis(gpModels : list):

    formattedNames = [gp_utils.fieldNameToText(i) for i in gpModels[0].inputNames]
    # SALib SOBOL indices
    sp = ProblemSpec({
        'num_vars': len(formattedNames),
        'names': formattedNames,
        'bounds': [[np.min(column), np.max(column)] for column in gpModels[0].normalisedInputs.T]
    })
    test_values = salsamp.sobol.sample(sp, int(2**17))

    for model in gpModels:
        model : gp_utils.GPModel
        print(f"SOBOL analysing {model.kernelName} model for {model.outputName}....")

        if model.regressionModel:
            y_prediction, y_std = model.regressionModel.predict(test_values, return_std=True)
            print(f"{model.regressionModel.kernel_} predictions for {model.outputName} -- y_std: {y_std}")
        elif model.classificationModel:
            y_prediction = model.classificationModel.predict_proba(test_values)[:,1]
            print(f"{model.classificationModel.kernel_} predictions for {model.outputName}")

        
        sobol_indices = sobol.analyze(sp, y_prediction, print_to_console=True)
        print(f"Sobol indices for output {model.outputName}:")
        print(sobol_indices)
        print(f"Sum of SOBOL indices: ST = {np.sum(sobol_indices['ST'])}, S1 = {np.sum(sobol_indices['S1'])}, abs(S1) = {np.sum(abs(sobol_indices['S1']))} S2 = {np.nansum(sobol_indices['S2'])}, abs(S2) = {np.nansum(abs(sobol_indices['S2']))}")
        plt.rcParams["figure.figsize"] = (14,10)
        #fig, ax = plt.subplots()
        sobol_indices.plot()
        plt.subplots_adjust(bottom=0.3)
        plt.title(f"Output: {gp_utils.fieldNameToText(model.outputName)}")
        #plt.tight_layout()
        plt.show()
        plt.close()

def evaluate_model(gpModels : list, crossValidate : bool = True, folds : int = 5):
    
    uniqueKernels = set([m.kernelName for m in gpModels])
    uniqueOutputNames = set([m.outputName for m in gpModels])
    evaluationData = {o : {k : {"score" : None, "std" : None} for k in uniqueKernels} for o in uniqueOutputNames}
    for model in gpModels:
        model : gp_utils.GPModel
        if crossValidate:
            if model.modelParams:
                retrainedModel = untrained_GP(
                    model.kernelName, 
                    model.modelParams["cv_lower"], 
                    model.modelParams["cv_upper"],
                    model.modelParams["ls_lower"],
                    model.modelParams["ls_upper"],
                    model.modelParams["a_lower"],
                    model.modelParams["a_upper"],
                    model.modelParams["nl_lower"],
                    model.modelParams["nl_upper"]
                )
            all_cv_scores = cross_val_score(retrainedModel, model.normalisedInputs, model.output, cv=folds)
            print(f"R^2 for GP ({model.kernelName} kernel) regression of {model.outputName} across {folds}-fold cross-validation: {all_cv_scores.mean()} (std: {all_cv_scores.std()})")
            evaluationData[model.outputName][model.kernelName]["score"] = all_cv_scores.mean()
            evaluationData[model.outputName][model.kernelName]["std"] = all_cv_scores.std()
        else:
            x_train, x_test, y_train, y_test = train_test_split(model.normalisedInputs, model.output, test_size=0.2)
            
            # Retrain without test data
            fit = train_data(x_train, y_train, model.kernelName)
            r_squared = fit.score(x_test, y_test)
            print(f"R^2 for GP ({model.kernelName} kernel) regression of {model.outputName}: {r_squared}")
            evaluationData[model.outputName][model.kernelName]["score"] = r_squared

    x = 0  # the label locations
    width = 1.0/(len(uniqueKernels) + 1)  # the width of the bars
    fig, ax = plt.subplots(layout='constrained')
    fig.set_figheight(6)
    fig.set_figwidth(12)
    zScore = abs(norm.ppf(0.975))
    colors = list(mcolors.TABLEAU_COLORS.keys())
    for _, data in evaluationData.items():
        multiplier = 0
        for kernel, kernelData in data.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, kernelData["score"], width, color=colors[multiplier])
            if x==0:
                rects.set_label(kernel)
            if crossValidate:
                ax.errorbar(x + offset, kernelData["score"], yerr=zScore*np.array(kernelData["std"]), fmt=' ', color='r')
            ax.bar_label(rects, padding=3, fmt="%.3f")
            multiplier += 1
        x += 1
    ax.set_ylabel(r'$R^2$')
    ax.set_title('R-squared scores by GP/output and kernel')
    nameLocs = np.arange(len(uniqueOutputNames))
    ax.set_xticks(nameLocs + ((len(uniqueKernels)-1)*0.5*width), [gp_utils.fieldNameToText(n) for n in evaluationData.keys()])
    ax.legend(loc='upper left')
    ax.set_ylim(0.0, 1.0)
    plt.tight_layout()
    plt.show()
    plt.close()

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
        gp_utils.display_matrix_plots(inputs, normalisedInputData, normalisedOutputs)

    model = {}
    model["fastIonSimulation"] = train_data(normalisedInputData, normalisedOutputs["fastIonSimulation"], "RQ", "classify")

    if plotModels:
        gp_utils.plot_models(model, showModels=plotModels, saveAnimation=False)

    # Perform SOBOL analysis
    if sobol:
        sobol_analysis(model, inNames, normalisedInputData, "classify")

    # Evaluate model
    if evaluateModels:
        kernels = ["RBF", "RQ", "Matern"]
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
        "--saveAnimation",
        action="store_true",
        help="Save animation of model data.",
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

    classify_simulations(args.irbDir, args.nullDir, args.inputFields, args.logFields, args.matrixPlot, args.sobol, args.evaluate, args.plotModels)
