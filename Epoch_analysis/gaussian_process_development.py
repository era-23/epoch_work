import argparse
import glob
import os
import pprint
import random
import typing
import copy
import itertools
from functools import partial
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib.colors as mcolors 
from mpl_toolkits.mplot3d import Axes3D
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
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, WhiteKernel, ConstantKernel, Matern
from sklearn.exceptions import ConvergenceWarning
from SALib import ProblemSpec
from SALib.analyze import sobol
import SALib.sample as salsamp
import epoch_utils as utils
import warnings
warnings.filterwarnings("error", category=ConvergenceWarning)

fieldNameToText_dict = {
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
    "Magnetic_Field_Bz/growthRates/max/growthRate" : "Bz_maxGamma",
    "Magnetic_Field_Bz/growthRates/max/peakPower" : "Bz_maxGammaPeakPower",
    "Magnetic_Field_Bz/growthRates/max/residual" : "Bz_maxGammaFitResidual",
    "Magnetic_Field_Bz/growthRates/max/time" : "Bz_maxGammaTime",
    "Magnetic_Field_Bz/growthRates/max/totalPower" : "Bz_maxGammaTotalPower",
    "Magnetic_Field_Bz/growthRates/max/wavenumber" : "Bz_maxGammaK",
    "Magnetic_Field_Bz/growthRates/maxInHighPeakPowerK/growthRate" : "Bz_peakKmaxGamma",
    "Magnetic_Field_Bz/growthRates/maxInHighPeakPowerK/peakPower" : "Bz_peakKmaxGammaPeakPower",
    "Magnetic_Field_Bz/growthRates/maxInHighPeakPowerK/residual" : "Bz_peakKmaxGammaFitResidual",
    "Magnetic_Field_Bz/growthRates/maxInHighPeakPowerK/time" : "Bz_peakKmaxGammaTime",
    "Magnetic_Field_Bz/growthRates/maxInHighPeakPowerK/totalPower" : "Bz_peakKmaxGammaTotalPower",
    "Magnetic_Field_Bz/growthRates/maxInHighPeakPowerK/wavenumber" : "Bz_peakKmaxGammaK",
    "Magnetic_Field_Bz/growthRates/maxInHighTotalPowerK/growthRate" : "Bz_totalKmaxGamma",
    "Magnetic_Field_Bz/growthRates/maxInHighTotalPowerK/peakPower" : "Bz_totalKmaxGammaPeakPower",
    "Magnetic_Field_Bz/growthRates/maxInHighTotalPowerK/residual" : "Bz_totalKmaxGammaFitResidual",
    "Magnetic_Field_Bz/growthRates/maxInHighTotalPowerK/time" : "Bz_totalKmaxGammaTime",
    "Magnetic_Field_Bz/growthRates/maxInHighTotalPowerK/totalPower" : "Bz_totalKmaxGammaTotalPower",
    "Magnetic_Field_Bz/growthRates/maxInHighTotalPowerK/wavenumber" : "Bz_totalKmaxGammaK",
    "Electric_Field_Ex/totalMagnitude" : "Ex_totalPower", 
    "Electric_Field_Ex/meanMagnitude" : "Ex_meanPower", 
    "Electric_Field_Ex/totalDelta" : "Ex_deltaTotalPower", 
    "Electric_Field_Ex/meanDelta" : "Ex_deltaMeanPower", 
    "Electric_Field_Ex/peakTkSpectralPower" : "Ex_peakTkPower", 
    "Electric_Field_Ex/meanTkSpectralPower" : "Ex_meanTkPower", 
    "Electric_Field_Ex/peakTkSpectralPowerRatio" : "Ex_tkPowerRatio",
    "Electric_Field_Ex/growthRates/max/growthRate" : "Ex_maxGamma",
    "Electric_Field_Ex/growthRates/max/peakPower" : "Ex_maxGammaPeakPower",
    "Electric_Field_Ex/growthRates/max/residual" : "Ex_maxGammaFitResidual",
    "Electric_Field_Ex/growthRates/max/time" : "Ex_maxGammaTime",
    "Electric_Field_Ex/growthRates/max/totalPower" : "Ex_maxGammaTotalPower",
    "Electric_Field_Ex/growthRates/max/wavenumber" : "Ex_maxGammaK",
    "Electric_Field_Ex/growthRates/maxInHighPeakPowerK/growthRate" : "Ex_peakKmaxGamma",
    "Electric_Field_Ex/growthRates/maxInHighPeakPowerK/peakPower" : "Ex_peakKmaxGammaPeakPower",
    "Electric_Field_Ex/growthRates/maxInHighPeakPowerK/residual" : "Ex_peakKmaxGammaFitResidual",
    "Electric_Field_Ex/growthRates/maxInHighPeakPowerK/time" : "Ex_peakKmaxGammaTime",
    "Electric_Field_Ex/growthRates/maxInHighPeakPowerK/totalPower" : "Ex_peakKmaxGammaTotalPower",
    "Electric_Field_Ex/growthRates/maxInHighPeakPowerK/wavenumber" : "Ex_peakKmaxGammaK",
    "Electric_Field_Ex/growthRates/maxInHighTotalPowerK/growthRate" : "Ex_totalKmaxGamma",
    "Electric_Field_Ex/growthRates/maxInHighTotalPowerK/peakPower" : "Ex_totalKmaxGammaPeakPower",
    "Electric_Field_Ex/growthRates/maxInHighTotalPowerK/residual" : "Ex_totalKmaxGammaFitResidual",
    "Electric_Field_Ex/growthRates/maxInHighTotalPowerK/time" : "Ex_totalKmaxGammaTime",
    "Electric_Field_Ex/growthRates/maxInHighTotalPowerK/totalPower" : "Ex_totalKmaxGammaTotalPower",
    "Electric_Field_Ex/growthRates/maxInHighTotalPowerK/wavenumber" : "Ex_totalKmaxGammaK",
    "B0strength" : "B0strength", 
    "B0angle" : "B0angle", 
    "backgroundDensity" : "backgroundDensity", 
    "beamFraction" : "beamFraction",
}

def anim_init(fig, ax, xx, yy, zz):
    ax.scatter(xx, yy, zz, color="black")
    return fig,

def anim_animate(i, fig, ax):
    ax.view_init(elev=10., azim=i)
    return fig,

def fieldNameToText(name : str) -> str:
    if name in fieldNameToText_dict:
        return fieldNameToText_dict[name]
    else:
        return name

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

def untrained_GP(kernel : str, type : str, cv_lower = 1e-5, cv_upper = 1e5, ls_lower = 1e-5, ls_upper=1e5, a_lower = 1e-5, a_upper = 1e5, nl_lower = 1e-5, nl_upper = 1e5):
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

    if type == "regress":
        return GaussianProcessRegressor(k, n_restarts_optimizer=50)
    elif type == "classify":
        return GaussianProcessClassifier(k, n_restarts_optimizer=0)
    else:
        raise NotImplementedError("type must be one of regress or classify")

def train_data(x_train : np.ndarray, y_train : np.ndarray, kernel : str, problemType : str):

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
            gp = untrained_GP(kernel, problemType, cv_lower, cv_upper, ls_lower, ls_upper, a_lower, a_upper, nl_lower, nl_upper)
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
    
def display_matrix_plots(inputs : dict, normalisedInputs : np.ndarray, outputs: dict, normalisedOutputs : dict = {}):

    for outName, outData in outputs.items():
        data = [outData] + list(inputs.values())
        labels = [fieldNameToText(outName)] + [fieldNameToText(i) for i in inputs.keys()]
        matrix_plot(data, labels, show=False)
        plt.title("Raw data")
        plt.tight_layout()
        plt.show()

        if normalisedOutputs:
            data = list(normalisedInputs.T)
            data.insert(0, list(normalisedOutputs[outName]))
            labels = [fieldNameToText(outName)] + [fieldNameToText(i) for i in inputs.keys()]
            matrix_plot(data, labels, show=False)
            plt.title("Normalised data")
            plt.tight_layout()
            plt.show()

    plt.close()

def plot_three_dimensions(input_1_index : int, input_2_index : int, gpModel : utils.GPModel, showModels = True, saveAnimation = False):
    
    if showModels:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(gpModel.normalisedInputs[:,input_1_index], gpModel.normalisedInputs[:,input_2_index], gpModel.output)
        ax.set_xlabel(fieldNameToText(gpModel.inputNames[input_1_index]))
        ax.set_ylabel(fieldNameToText(gpModel.inputNames[input_2_index]))
        ax.set_zlabel(fieldNameToText(gpModel.outputName))
        ax.set_title(f"Training data for output {fieldNameToText(gpModel.outputName)} ({gpModel.kernelName} kernel)")

        plt.show()
    plt.close()

    # Sample homogeneously and plot contours with training data
    sp = ProblemSpec({
        'num_vars': gpModel.normalisedInputs.shape[1],
        'names': gpModel.inputNames,
        'bounds': [[np.min(column), np.max(column)] for column in gpModel.normalisedInputs.T]
    })
    gp_test_values = salsamp.sobol.sample(sp, int(2**9))
    for column in range(gp_test_values.shape[1]):
        if column is not input_1_index and column is not input_2_index:
            mean = np.mean(np.array(gp_test_values[:,column]))
            gp_test_values[:,column] = mean

    # Training data
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')
    #ax = Axes3D(fig)
    #ax.scatter(allNormalisedInputs[:,input_1_index], allNormalisedInputs[:,input_2_index], output, color="black")
    ax.set_xlabel(fieldNameToText(gpModel.inputNames[input_1_index]))
    ax.set_ylabel(fieldNameToText(gpModel.inputNames[input_2_index]))
    ax.set_zlabel(gpModel.outputName)

    # GP predictions
    if gpModel.regressionModel:
        gp_z = gpModel.regressionModel.predict(gp_test_values)
    elif gpModel.classificationModel:
        gp_z = gpModel.classificationModel.predict_proba(gp_test_values)[:,1]
    else:
        raise NotImplementedError("type must be one of regress or classify")
    gp_x, gp_y = gp_test_values[:,input_1_index], gp_test_values[:,input_2_index]
    ax.plot_trisurf(gp_x, gp_y, gp_z, cmap="plasma")
    #plt.colorbar(surf, label=output_name)
    ax.set_title(f"Training data and GP prediction surface ({gpModel.kernelName})")
    #plt.show()

    if saveAnimation:
        # Animate
        anim = animation.FuncAnimation(
            fig, 
            partial(anim_animate, fig=fig, ax=ax), 
            init_func=partial(anim_init, fig=fig, ax=ax, xx=gpModel.normalisedInputs[:,input_1_index], yy=gpModel.normalisedInputs[:,input_2_index], zz=gpModel.output),
            frames=360, 
            interval=20, 
            blit=False)
        # Save
        anim.save(f'/home/era536/Documents/for_discussion/2025.02.27/{gpModel.inputNames[input_1_index].replace("/", "-")}_and_{gpModel.inputNames[input_2_index].replace("/", "-")}_vs{gpModel.outputName.replace("/", "-")}_{gpModel.kernelName}.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    
    plt.close()

def plot_models(gpModels : list, showModels : bool = True, saveAnimation : bool = False):
    
    for model in gpModels:
        model : utils.GPModel
        # 3D plot if there are only 2 input dimensions
        allInputCombos = list(itertools.combinations(range(len(model.inputNames)), 2))
        for inputPair in allInputCombos:
            i1 = inputPair[0]
            i2 = inputPair[1]
            plot_three_dimensions(i1, i2, model, showModels, saveAnimation)

        # Sample values of each input dim keeping others fixed at their mean value
        zScore = abs(norm.ppf(0.975))
        numSamples = 1000
        for i in range(len(model.inputNames)):
            inputName = fieldNameToText(model.inputNames[i])
            sampleColumn = model.normalisedInputs[:,i]
            samples = np.linspace(sampleColumn.min(), sampleColumn.max(), numSamples)

            allDatapoints = [[] for _ in model.inputNames]
            trainingDistancesForAlphas = np.zeros(len(sampleColumn))
            for i2 in range(len(model.inputNames)):
                if i2 == i:
                    allDatapoints[i2] = samples
                else:
                    mean = model.normalisedInputs[:,i2].mean()
                    allDatapoints[i2] = np.ones(numSamples) * mean
                    # Sum geometric distance from non-x-axis dim means to calculate how relevant each point is for this view
                    trainingDistancesForAlphas += (model.normalisedInputs[:,i2] - mean)**2

            # Columnise
            allDatapoints = np.array(allDatapoints).T

            # Sample
            y_preds = []
            if model.regressionModel:
                y_preds, y_std = model.regressionModel.predict(allDatapoints, return_std=True)
                plt.fill_between(
                    samples,
                    y_preds - (zScore * y_std),
                    y_preds + (zScore * y_std),
                    color="tab:red",
                    alpha=0.3,
                    label=r"95% confidence interval"
                )
            elif model.classificationModel:
                y_preds = model.classificationModel.predict_proba(allDatapoints)[:,1]
            else:
                raise NotImplementedError("model must be Gaussian Process regression or classification")

            # Calculate alphas
            trainingDistancesForAlphas = np.sqrt(trainingDistancesForAlphas)
            normDistance = (trainingDistancesForAlphas - np.min(trainingDistancesForAlphas)) / (np.max(trainingDistancesForAlphas) - np.min(trainingDistancesForAlphas))
            normAlpha = 1.0 - normDistance # Closer is better therefore higher alpha

            # Plot
            plt.plot(samples, y_preds, label="Mean prediction", color="tab:red")
            plt.scatter(sampleColumn, model.output, label="Training data", alpha=normAlpha)
            plt.xlabel(inputName)
            plt.ylabel(fieldNameToText(model.outputName))
            plt.legend()
            #plt.tight_layout()
            plt.title(f"GP predictions for {fieldNameToText(model.outputName)} ({model.kernelName} kernel), all other input dimensions fixed to mean value", wrap=True)
            plt.show()

def sobol_analysis(gpModels : list):

    formattedNames = [fieldNameToText(i) for i in gpModels[0].inputNames]
    # SALib SOBOL indices
    sp = ProblemSpec({
        'num_vars': len(formattedNames),
        'names': formattedNames,
        'bounds': [[np.min(column), np.max(column)] for column in gpModels[0].normalisedInputs.T]
    })
    test_values = salsamp.sobol.sample(sp, int(2**17))

    for model in gpModels:
        model : utils.GPModel
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
        plt.title(f"Output: {fieldNameToText(model.outputName)}")
        #plt.tight_layout()
        plt.show()
        plt.close()

def evaluate_model(gpModels : list, crossValidate : bool = True, folds : int = 5):
    
    uniqueKernels = set([m.kernelName for m in gpModels])
    uniqueOutputNames = set([m.outputName for m in gpModels])
    evaluationData = {o : {k : {"score" : None, "std" : None} for k in uniqueKernels} for o in uniqueOutputNames}
    for model in gpModels:
        model : utils.GPModel
        if crossValidate:
            problemType = "regress" if model.regressionModel else "classify"
            if model.modelParams:
                retrainedModel = untrained_GP(
                    model.kernelName, 
                    problemType, 
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
            fit = train_data(x_train, y_train, model.kernelName, problemType)
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
    ax.set_xticks(nameLocs + ((len(uniqueKernels)-1)*0.5*width), [fieldNameToText(n) for n in evaluationData.keys()])
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
        plotModels : bool = False,
        showModels : bool = False,
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
        normalisedOutputs[outputName] = np.array([n[0] for n in preprocess_output_data(outputData, arcsinh=(outputName in logFields))])

    # Display matrix plots
    if matrixPlot:
        display_matrix_plots(inputs, normalisedInputData, outputs, normalisedOutputs)

    # Train individual models for each output
    kernels = ["RBF", "RQ"]
    models = []
    for k in kernels:
        for outputName, outputData in normalisedOutputs.items():
            regressionModel, modelParams = train_data(normalisedInputData, outputData, k, "regress")
            models.append(utils.GPModel(
                regressionModel=regressionModel,
                modelParams=modelParams,
                kernelName=k,
                inputNames=inNames,
                normalisedInputs=normalisedInputData,
                outputName=outputName,
                output=outputData
            ))

    if plotModels:
        plot_models(models, showModels, saveAnimation)

    # Perform SOBOL analysis
    if sobol:
        sobol_analysis(models)

    # Evaluate model
    if evaluateModels:
        evaluate_model(models)

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
        regress_simulations(args.dir, args.inputFields, args.outputFields, args.logFields, args.matrixPlot, args.sobol, args.evaluate, args.plotModels, args.showModels, args.saveAnimation)
    if args.classify:
        classify_simulations(args.irbDir, args.nullDir, args.inputFields, args.logFields, args.matrixPlot, args.sobol, args.evaluate, args.plotModels)
