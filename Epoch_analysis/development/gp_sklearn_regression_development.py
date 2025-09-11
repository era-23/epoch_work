import argparse
import glob
import copy
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, WhiteKernel, ConstantKernel, Matern
from sklearn.exceptions import ConvergenceWarning
from SALib import ProblemSpec
from SALib.analyze import sobol
import SALib.sample as salsamp
import ml_utils
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
    return list(inputData.keys()), preprocess.fit_transform(inputData_asColVec), preprocess

def preprocess_output_data(outputData : list, arcsinh = False) -> tuple:
    if arcsinh:
        outputData = np.arcsinh(outputData)
    outputData_shaped = np.array(outputData).reshape(-1, 1)
    scaler = StandardScaler()
    return scaler.fit(outputData_shaped).transform(outputData_shaped), scaler

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

    return GaussianProcessRegressor(k, n_restarts_optimizer=50)

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
    gp = GaussianProcessRegressor()
    fit = GaussianProcessRegressor()
    prevGp = GaussianProcessRegressor()
    prevFit = GaussianProcessRegressor()
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

def sobol_analysis(gpModels : list, noTitle : bool = False):

    formattedNames = [ml_utils.fieldNameToText(i) for i in gpModels[0].inputNames]
    # SALib SOBOL indices
    sp = ProblemSpec({
        'num_vars': len(formattedNames),
        'names': formattedNames,
        'bounds': [[np.min(column), np.max(column)] for column in gpModels[0].normalisedInputs.T]
    })
    test_values = salsamp.sobol.sample(sp, int(2**17))

    for model in gpModels:
        model : ml_utils.GPModel
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
        Si_df = sobol_indices.to_df()
        _, ax = plt.subplots(1, len(Si_df), sharey=True)
        CONF_COLUMN = "_conf"
        for idx, f in enumerate(Si_df):
            conf_cols = f.columns.str.contains(CONF_COLUMN)

            confs = f.loc[:, conf_cols]
            confs.columns = [c.replace(CONF_COLUMN, "") for c in confs.columns]

            Sis = f.loc[:, ~conf_cols]

            ax[idx] = Sis.plot(kind="bar", yerr=confs, ax=ax[idx])
        print(plt.ylim())
        plt.subplots_adjust(bottom=0.3)
        if not noTitle:
            plt.title(f"{model.kernelName} kernel: {ml_utils.fieldNameToText(model.outputName)}")
        plt.tight_layout()
        plt.show()
        plt.close()

def evaluate_model(gpModels : list, crossValidate : bool = True, folds : int = 5, noTitle : bool = False):
    
    uniqueKernels = set([m.kernelName for m in gpModels])
    uniqueOutputNames = set([m.outputName for m in gpModels])
    evaluationData = {o : {k : {"score" : None, "std" : None} for k in uniqueKernels} for o in uniqueOutputNames}
    for model in gpModels:
        model : ml_utils.GPModel
        if crossValidate:
            if model.modelParams:
                untrainedModel = untrained_GP(
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
            all_cv_scores = cross_val_score(untrainedModel, model.normalisedInputs, model.output, cv=KFold(shuffle=True))
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

    alphabeticalEvalData = dict(sorted(evaluationData.items()))
    x = 0  # the label locations
    width = 1.0/(len(uniqueKernels) + 1)  # the width of the bars
    fig, ax = plt.subplots(layout='constrained')
    fig.set_figheight(6)
    fig.set_figwidth(12)
    zScore = abs(norm.ppf(0.975))
    colors = ['tab:blue','tab:purple','tab:green','tab:orange','tab:yellow']
    for _, data in alphabeticalEvalData.items():
        multiplier = 0
        for kernel, kernelData in data.items():
            kernelScore = 0.0 if kernelData["score"] is None else kernelData["score"]
            kernelStd = 0.0 if kernelData["std"] is None else kernelData["std"]
            offset = width * multiplier
            rects = ax.bar(x + offset, kernelScore, width, color=colors[multiplier])
            if x==0:
                rects.set_label(kernel)
            if crossValidate:
                ax.errorbar(x + offset, kernelScore, yerr=zScore*np.array(kernelStd), fmt=' ', color='r')
            #ax.bar_label(rects, padding=3, fmt="%.3f")
            ax.bar_label(rects, label_type='center', fmt="%.3f", fontsize=16)
            multiplier += 1
        x += 1
    ax.set_ylabel(r'$R^2$')
    if not noTitle:
        ax.set_title('R-squared scores by GP/output and kernel')
    nameLocs = np.arange(len(uniqueOutputNames))
    ax.set_xticks(nameLocs + ((len(uniqueKernels)-1)*0.5*width), [ml_utils.fieldNameToText(n) for n in alphabeticalEvalData.keys()])
    ax.legend()
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
        plotModels : bool = False,
        showModels : bool = False,
        bigLabels : bool = False,
        noTitle : bool = False,
        saveAnimation : bool = False
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

    if bigLabels:
        plt.rcParams.update({'axes.titlesize': 26.0})
        plt.rcParams.update({'axes.labelsize': 24.0})
        plt.rcParams.update({'xtick.labelsize': 20.0})
        plt.rcParams.update({'ytick.labelsize': 20.0})
        plt.rcParams.update({'legend.fontsize': 18.0})
    else:
        plt.rcParams.update({'axes.titlesize': 18.0})
        plt.rcParams.update({'axes.labelsize': 16.0})
        plt.rcParams.update({'xtick.labelsize': 14.0})
        plt.rcParams.update({'ytick.labelsize': 14.0})
        plt.rcParams.update({'legend.fontsize': 14.0})

    data_files = glob.glob(str(directory / "*.nc")) 

    # Input data
    inputs = {inp : [] for inp in inputFields}
    inputs = ml_utils.read_data(data_files, inputs)

    # Output data
    outputs = {outp : [] for outp in outputFields}
    outputs = ml_utils.read_data(data_files, outputs)

    # Preprocess inputs (multiple returned as column vector)
    inNames, normalisedInputData, preprocessingPipeline = preprocess_input_data(inputs, list(set(logFields).intersection(inputFields)))

    # Preprocess output (singular, for now -- 1 GP each)
    normalisedOutputs = {}
    outputScaler = None
    for outputName, outputData in outputs.items():
        data, outputScaler = preprocess_output_data(outputData, arcsinh=(outputName in logFields))
        normalisedOutputs[outputName] = np.array([n[0] for n in data])

    # Display matrix plots
    if matrixPlot:
        ml_utils.display_matrix_plots(inputs, normalisedInputData, outputs, normalisedOutputs)

    # Train individual models for each output
    # kernels = ["RBF", "RQ"]
    kernels = ["RQ"]
    models = []
    for k in kernels:
        for outputName, outputData in normalisedOutputs.items():
            regressionModel, modelParams = train_data(normalisedInputData, outputData, k)
            models.append(ml_utils.GPModel(
                regressionModel=regressionModel,
                modelParams=modelParams,
                kernelName=k,
                inputNames=inNames,
                normalisedInputs=normalisedInputData,
                outputName=outputName,
                output=outputData,
                fitSuccess = (regressionModel is not None)
            ))

    successfulModels = [model for model in models if model.fitSuccess]
    if plotModels:
        ml_utils.plot_models(gpModels=successfulModels, rawInputData=inputs, rawOutputData=outputs, showModels=showModels, saveAnimation=saveAnimation, noTitle=noTitle)

    # Evaluate model
    if evaluateModels:
        evaluate_model(successfulModels, noTitle=noTitle)

    # Perform SOBOL analysis
    if sobol:
        sobol_analysis(successfulModels, noTitle=noTitle)

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
    parser.add_argument(
        "--saveAnimation",
        action="store_true",
        help="Save animation of model data.",
        required = False
    )

    args = parser.parse_args()

    regress_simulations(
        args.dir, 
        args.inputFields, 
        args.outputFields, 
        args.logFields, 
        args.matrixPlot, 
        args.sobol, 
        args.evaluate, 
        args.plotModels, 
        args.showModels, 
        args.bigLabels,
        args.noTitle,
        args.saveAnimation
    )
