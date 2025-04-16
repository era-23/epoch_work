import matplotlib.pyplot as plt
import epoch_utils as e_utils
import numpy as np
import itertools
from scipy.stats import norm
from scipy.interpolate import griddata
from functools import partial
from matplotlib import animation
from inference.plotting import matrix_plot
from SALib import ProblemSpec
import SALib.sample as salsamp

fieldNameToText_dict = {
    "Energy/electricFieldEnergyDensity_delta" : "E_deltaE",
    "Energy/magneticFieldEnergyDensity_delta" : "B_deltaE", 
    "Energy/backgroundIonEnergyDensity_delta" : "background\ndelta KE", 
    "Energy/electronEnergyDensity_delta" : "e_deltaKE",
    "Energy/fastIonEnergyDensity_max" : "fast ion\nmax KE", 
    "Energy/fastIonEnergyDensity_timeMax" : "FI_timeMaxKE", 
    "Energy/fastIonEnergyDensity_min" : "fast ion\nmin KE", 
    "Energy/fastIonEnergyDensity_timeMin" : "fast ion\nmin time", 
    "Energy/fastIonEnergyDensity_delta" : "fast ion\ndelta KE",
    "Energy/backgroundIonEnergyDensity_max" : "background\nmax KE",
    "Energy/backgroundIonEnergyDensity_timeMax" : "background\npeak time",
    "Energy/electronEnergyDensity_timeMax" : "electron_timeMaxKE",
    "Energy/electricFieldEnergyDensity_timeMax" : "Ex_timeMaxE",
    "Energy/magneticFieldEnergyDensity_timeMax" : "Bz_timeMaxE",
    "Energy/backgroundIonEnergyDensity_timeMin" : "bkgdIon_timeMinKE",
    "Energy/electronEnergyDensity_timeMin" : "electron_timeMinKE",
    "Energy/electricFieldEnergyDensity_timeMin" : "Ex_timeMinE",
    "Energy/magneticFieldEnergyDensity_timeMin" : "Bz_timeMinE",
    "Magnetic_Field_Bz/totalMagnitude" : "Bz_totalPower", 
    "Magnetic_Field_Bz/meanMagnitude" : "Bz_meanPower", 
    "Magnetic_Field_Bz/totalDelta" : "Bz_deltaTotalPower", 
    "Magnetic_Field_Bz/meanDelta" : "Bz_deltaMeanPower", 
    "Magnetic_Field_Bz/peakTkSpectralPower" : "Bz (peak)", 
    "Magnetic_Field_Bz/meanTkSpectralPower" : "Bz (mean)", 
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
    "Electric_Field_Ex/peakTkSpectralPower" : "Ex (peak)", 
    "Electric_Field_Ex/meanTkSpectralPower" : "Ex (mean)", 
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
    "B0strength" : "B0", 
    "B0angle" : "B0 angle", 
    "backgroundDensity" : "density", 
    "beamFraction" : "beam frac",
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

def plot_three_dimensions(input_1_index : int, input_2_index : int, gpModel : e_utils.GPModel, showModels = True, saveAnimation = False, noTitle = False):
    
    # if showModels:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection='3d')
    #     ax.scatter(gpModel.normalisedInputs[:,input_1_index], gpModel.normalisedInputs[:,input_2_index], gpModel.output)
    #     ax.set_xlabel(fieldNameToText(gpModel.inputNames[input_1_index]))
    #     ax.set_ylabel(fieldNameToText(gpModel.inputNames[input_2_index]))
    #     ax.set_zlabel(fieldNameToText(gpModel.outputName))
    #     if not noTitle:
    #         ax.set_title(f"Training data for output {fieldNameToText(gpModel.outputName)} ({gpModel.kernelName} kernel)")

    #     plt.show()
    plt.close()

    # Sample homogeneously and plot contours with training data
    sp = ProblemSpec({
        'num_vars': gpModel.normalisedInputs.shape[1],
        'names': gpModel.inputNames,
        'bounds': [[np.min(column), np.max(column)] for column in gpModel.normalisedInputs.T]
    })
    gp_test_values = salsamp.sobol.sample(sp, int(2**10))
    for column in range(gp_test_values.shape[1]):
        if column is not input_1_index and column is not input_2_index:
            mean = np.mean(np.array(gp_test_values[:,column]))
            gp_test_values[:,column] = mean

    # Training data
    fig = plt.figure(figsize=(10, 8))
    #ax = fig.add_subplot(projection='3d')
    #ax.set_xlabel('\n' + fieldNameToText(gpModel.inputNames[input_1_index]))
    #ax.set_ylabel('\n' + fieldNameToText(gpModel.inputNames[input_2_index]))
    #ax.set_zlabel('\n' + fieldNameToText(gpModel.outputName))

    # GP predictions
    if gpModel.regressionModel:
        gp_z = gpModel.regressionModel.predict(gp_test_values)
    elif gpModel.classificationModel:
        gp_z = gpModel.classificationModel.predict_proba(gp_test_values)[:,1]
    else:
        raise NotImplementedError("type must be one of regress or classify")
    gp_x, gp_y = gp_test_values[:,input_1_index], gp_test_values[:,input_2_index]
    #ax.plot_trisurf(gp_x, gp_y, gp_z, cmap="plasma")
    X, Y = np.meshgrid(np.linspace(min(gp_x), max(gp_x), len(gp_x)), np.linspace(min(gp_y), max(gp_y), len(gp_y)))
    Z = griddata((gp_x, gp_y), gp_z, (X, Y), method='cubic')
    plt.imshow(Z, extent=[gp_x.min(), gp_x.max(), gp_y.min(), gp_y.max()], origin='lower', cmap='plasma', aspect='auto')
    plt.colorbar(label=fieldNameToText(gpModel.outputName))
    plt.xlabel(fieldNameToText(gpModel.inputNames[input_1_index]))
    plt.ylabel(fieldNameToText(gpModel.inputNames[input_2_index]))
    if not noTitle:
        #ax.set_title(f"Training data and GP prediction surface ({gpModel.kernelName})")
        plt.title(f"Training data and GP prediction surface ({gpModel.kernelName})")
    if showModels:
        plt.show()
    plt.close()

    # if saveAnimation:
    #     # Animate
    #     anim = animation.FuncAnimation(
    #         fig, 
    #         partial(anim_animate, fig=fig, ax=ax), 
    #         init_func=partial(anim_init, fig=fig, ax=ax, xx=gpModel.normalisedInputs[:,input_1_index], yy=gpModel.normalisedInputs[:,input_2_index], zz=gpModel.output),
    #         frames=360, 
    #         interval=20, 
    #         blit=False)
    #     # Save
    #     anim.save(f'/home/era536/Documents/for_discussion/2025.02.27/{gpModel.inputNames[input_1_index].replace("/", "-")}_and_{gpModel.inputNames[input_2_index].replace("/", "-")}_vs{gpModel.outputName.replace("/", "-")}_{gpModel.kernelName}.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    # plt.close()

def display_matrix_plots(inputs : dict, normalisedInputs : np.ndarray, outputs: dict, normalisedOutputs : dict = {}, noTitle : bool = False):

    for outName, outData in outputs.items():
        data = [outData] + list(inputs.values())
        labels = [fieldNameToText(outName)] + [fieldNameToText(i) for i in inputs.keys()]
        matrix_plot(data, labels, show=False)
        if not noTitle:
            plt.title("Raw data")
        plt.tight_layout()
        plt.show()

        if normalisedOutputs:
            data = list(normalisedInputs.T)
            data.insert(0, list(normalisedOutputs[outName]))
            labels = [fieldNameToText(outName)] + [fieldNameToText(i) for i in inputs.keys()]
            matrix_plot(data, labels, show=False)
            if not noTitle:
                plt.title("Normalised data")
            plt.tight_layout()
            plt.show()

    plt.close()

def plot_models(gpModels : list, showModels : bool = True, saveAnimation : bool = False, noTitle : bool = False):
    
    for model in gpModels:
        model : e_utils.GPModel
        # 3D plot if there are only 2 input dimensions
        allInputCombos = list(itertools.combinations(range(len(model.inputNames)), 2))
        for inputPair in allInputCombos:
            i1 = inputPair[0]
            i2 = inputPair[1]
            plot_three_dimensions(i1, i2, model, showModels, saveAnimation, noTitle)

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
            if not noTitle:
                plt.title(f"GP predictions for {fieldNameToText(model.outputName)} ({model.kernelName} kernel), all other input dimensions fixed to mean value", wrap=True)
            plt.show()