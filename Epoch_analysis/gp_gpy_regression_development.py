import argparse
import glob
import os
import GPy
import SALib.sample as salsamp
import ml_utils
import epoch_utils
import pylab as pb
import numpy as np
import xarray as xr
from scipy.stats import linregress
from SALib import ProblemSpec
from SALib.analyze import sobol
from sklearn.model_selection import RepeatedKFold, LeaveOneOut
from sklearn.metrics import r2_score, root_mean_squared_error
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.collections import PathCollection
from dataclasses import fields

def plot_4inputs_training_data(train, yName, xIn_1, xIn_2, xIn_3, xIn_4, name_1, name_2, name_3, name_4):
    fig = plt.figure(figsize=(12,12))
    # fig.subplots_adjust(bottom=0.2)
    plt.suptitle(f'Output: {yName}')
    # Input 1
    ax1 = fig.add_subplot(411)
    ax1.set_xlabel(f'{name_1}')
    ax1.plot(xIn_1,train,'kx',mew=1.5)
    # Input 2
    ax2 = fig.add_subplot(412)
    ax2.set_xlabel(f'{name_2}')
    ax2.plot(xIn_2,train,'kx',mew=1.5)
    # Input 3
    ax3 = fig.add_subplot(413)
    ax3.set_xlabel(f'{name_3}')
    ax3.plot(xIn_3,train,'kx',mew=1.5)
    # Input 4
    ax4 = fig.add_subplot(414)
    ax4.set_xlabel(f'{name_4}')
    ax4.plot(xIn_4,train,'kx',mew=1.5)
    plt.tight_layout()
    plt.show()

def plot_outputs(m, input_names, output_feature_names, xlim):
    
    num_datapoints = m.Y.shape[0]//len(output_feature_names)
    exponential_alpha_scaling_factor = 20
    plt.close("all")

    for i in range(len(output_feature_names)):

        fig = plt.figure(figsize=(18,8))
        output_name = output_feature_names[i]
        plt.suptitle(f'Output: {output_name}')
        fixed_inputs = [(len(input_names),i)]
        data_slice = slice(i*num_datapoints, (i+1)*num_datapoints)
        training_data = np.delete(m.X[data_slice,:], -1, axis=1)

        #Input 1
        input_index = 0
        x_name = input_names[input_index]
        ax1 = fig.add_subplot(221)
        ax1.set_xlim(xlim)
        ax1.set_xlabel(x_name)
        p : PathCollection = m.plot(
            plot_limits = xlim,
            fixed_inputs = fixed_inputs, 
            which_data_rows = data_slice, 
            ax = ax1, 
            plot_data = True, 
            visible_dims = [input_index]
        )['dataplot'][0]
        other_axes = np.delete(training_data, input_index, axis=1)
        mean_of_other_points = [np.mean(r) for r in other_axes]
        g = np.abs(mean_of_other_points) / np.max(mean_of_other_points)
        proximity_score = np.clip(1.0 - g, a_min = 0.1, a_max = None)
        proximity_score = exponential_alpha_scaling_factor**proximity_score / np.max(exponential_alpha_scaling_factor**proximity_score)
        p.set_color("red")
        p.set_alpha(proximity_score)
        
        #Input 2
        input_index = 1
        x_name = input_names[input_index]
        ax2 = fig.add_subplot(222)
        ax2.set_xlim(xlim)
        ax2.set_xlabel(x_name)
        p : PathCollection = m.plot(
            plot_limits = xlim,
            fixed_inputs = fixed_inputs, 
            which_data_rows = data_slice, 
            ax = ax2, 
            plot_data = True, 
            visible_dims = [input_index]
        )['dataplot'][0]
        other_axes = np.delete(training_data, input_index, axis=1)
        mean_of_other_points = [np.mean(r) for r in other_axes]
        g = np.abs(mean_of_other_points) / np.max(mean_of_other_points)
        proximity_score = np.clip(1.0 - g, a_min = 0.1, a_max = None)
        proximity_score = exponential_alpha_scaling_factor**proximity_score / np.max(exponential_alpha_scaling_factor**proximity_score)
        p.set_color("red")
        p.set_alpha(proximity_score)

        #Input 3
        input_index = 2
        x_name = input_names[input_index]
        ax3 = fig.add_subplot(223)
        ax3.set_xlim(xlim)
        ax3.set_xlabel(x_name)
        p : PathCollection = m.plot(
            plot_limits = xlim,
            fixed_inputs = fixed_inputs, 
            which_data_rows = data_slice, 
            ax = ax3, 
            plot_data = True, 
            visible_dims = [input_index]
        )['dataplot'][0]
        other_axes = np.delete(training_data, input_index, axis=1)
        mean_of_other_points = [np.mean(r) for r in other_axes]
        g = np.abs(mean_of_other_points) / np.max(mean_of_other_points)
        proximity_score = np.clip(1.0 - g, a_min = 0.1, a_max = None)
        proximity_score = exponential_alpha_scaling_factor**proximity_score / np.max(exponential_alpha_scaling_factor**proximity_score)
        p.set_color("red")
        p.set_alpha(proximity_score)

        #Input 4
        input_index = 3
        x_name = input_names[input_index]
        ax4 = fig.add_subplot(224)
        ax4.set_xlim(xlim)
        ax4.set_xlabel(x_name)
        p : PathCollection = m.plot(
            plot_limits = xlim,
            fixed_inputs = fixed_inputs, 
            which_data_rows = data_slice, 
            ax = ax4, 
            plot_data = True, 
            visible_dims = [input_index]
        )['dataplot'][0]
        other_axes = np.delete(training_data, input_index, axis=1)
        mean_of_other_points = [np.mean(r) for r in other_axes]
        g = np.abs(mean_of_other_points) / np.max(mean_of_other_points)
        proximity_score = np.clip(1.0 - g, a_min = 0.1, a_max = None)
        proximity_score = exponential_alpha_scaling_factor**proximity_score / np.max(exponential_alpha_scaling_factor**proximity_score)
        p.set_color("red")
        p.set_alpha(proximity_score)
        # Display
        plt.show()

def demo_plot_2outputs(m, Xt1, Yt1, Xt2, Yt2, xlim):
    fig = pb.figure(figsize=(12,8))
    #Output 1
    ax1 = fig.add_subplot(211)
    ax1.set_xlim(xlim)
    ax1.set_title('Output 1')
    m.plot(plot_limits=xlim,fixed_inputs=[(1,0)],which_data_rows=slice(0,100),ax=ax1)
    ax1.plot(Xt1[:,:1],Yt1,'rx',mew=1.5)
    #Output 2
    ax2 = fig.add_subplot(212)
    ax2.set_xlim(xlim)
    ax2.set_title('Output 2')
    m.plot(plot_limits=xlim,fixed_inputs=[(1,1)],which_data_rows=slice(100,200),ax=ax2)
    ax2.plot(Xt2[:,:1],Yt2,'rx',mew=1.5)
    plt.show()

def demo():
    # pb.ion()
    
    # These functions generate data corresponding to two outputs
    def f_output1(x):
        return 4. * np.cos(x/5.) - .4*x - 35. + np.random.rand(x.size)[:,None] * 2.
    def f_output2(x):
        return 6. * np.cos(x/5.) + .2*x + 35. + np.random.rand(x.size)[:,None] * 8.

    #{X,Y} training set for each output
    X1 = np.random.rand(100)[:,None]
    X1=X1*75
    X2 = np.random.rand(100)[:,None]
    X2=X2*70 + 30
    Y1 = f_output1(X1)
    Y2 = f_output2(X2)
    #{X,Y} test set for each output
    Xt1 = np.random.rand(100)[:,None]*100
    Xt2 = np.random.rand(100)[:,None]*100
    Yt1 = f_output1(Xt1)
    Yt2 = f_output2(Xt2)

    # Plot datasets
    xlim = (0,100)
    ylim = (0,50)
    fig = pb.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    ax1.set_xlim(xlim)
    ax1.set_title('Output 1')
    ax1.plot(X1[:,:1],Y1,'kx',mew=1.5,label='Train set')
    ax1.plot(Xt1[:,:1],Yt1,'rx',mew=1.5,label='Test set')
    ax1.legend()
    ax2 = fig.add_subplot(212)
    ax2.set_xlim(xlim)
    ax2.set_title('Output 2')
    ax2.plot(X2[:,:1],Y2,'kx',mew=1.5,label='Train set')
    ax2.plot(Xt2[:,:1],Yt2,'rx',mew=1.5,label='Test set')
    ax2.legend()
    plt.show()

    # Multiple output kernel function and coregionalization matrix
    K = GPy.kern.RBF(1)
    B = GPy.kern.Coregionalize(input_dim=1,output_dim=2) 
    multkernel = K.prod(B,name='B.K')
    print(multkernel)

    #Components of B
    print('W matrix\n',B.W)
    print('\nkappa vector\n',B.kappa)
    print('\nB matrix\n',B.B)

    # ICM handles this as above
    icm = GPy.util.multioutput.ICM(input_dim=1,num_outputs=2,kernel=GPy.kern.RBF(1))
    print(icm)

    # Now try a Matern kernel:
    K = GPy.kern.Matern32(1)
    icm = GPy.util.multioutput.ICM(input_dim=1,num_outputs=2,kernel=K)

    m = GPy.models.GPCoregionalizedRegression([X1,X2],[Y1,Y2],kernel=icm)
    m['.*Mat32.var'].constrain_fixed(1.) #For this kernel, B.kappa encodes the variance now.
    m.optimize()
    print(m)
    demo_plot_2outputs(m, Xt1, Yt1, Xt2, Yt2, xlim=(0,100),ylim=(-20,60))

    # Kernel selection with LCM and bias kernel since outputs have different means
    K1 = GPy.kern.Bias(1)
    K2 = GPy.kern.Linear(1)
    K3 = GPy.kern.Matern32(1)
    lcm = GPy.util.multioutput.LCM(input_dim=1,num_outputs=2,kernels_list=[K1,K2,K3])

    m = GPy.models.GPCoregionalizedRegression([X1,X2],[Y1,Y2],kernel=lcm)
    m['.*ICM.*var'].unconstrain()
    m['.*ICM0.*var'].constrain_fixed(1.)
    m['.*ICM0.*W'].constrain_fixed(0)
    m['.*ICM1.*var'].constrain_fixed(1.)
    m['.*ICM1.*W'].constrain_fixed(0)
    m.optimize()
    demo_plot_2outputs(m, Xt1, Yt1, Xt2, Yt2, xlim=(0,100),ylim=(-20,60))

def get_model(kernels, X_in, Y_out, results : ml_utils.GPResults, W_rank = 3, bound_length = True):
    
    # Dimensions
    input_dim = X_in.shape[1]
    output_dim = len(Y_out)
    
    # Kernels
    k_list = []
    for k in kernels:
        match k:
            case "white":
                k_list.append(GPy.kern.White(input_dim=input_dim))
            case "linear":
                k_list.append(GPy.kern.Linear(input_dim=input_dim))
            case "ratQuad":
                k_list.append(GPy.kern.RatQuad(input_dim=input_dim, ARD = True))
            case "rbf":
                k_list.append(GPy.kern.RBF(input_dim=input_dim, ARD = True))
            case "matern32":
                k_list.append(GPy.kern.Matern32(input_dim=input_dim, ARD = True))
            case "matern52":
                k_list.append(GPy.kern.Matern52(input_dim=input_dim, ARD = True))
    
    kerns = GPy.util.multioutput.LCM(input_dim=input_dim, num_outputs=output_dim, kernels_list=k_list, W_rank=W_rank)
    model = GPy.models.GPCoregionalizedRegression([X_in for _ in Y_out],Y_out,kernel=kerns)
    if bound_length:
        param = '.*lengthscale'
        lower = 0.0
        upper = 0.2
        model[param].constrain_bounded(lower=lower, upper=upper)
        results.fixedParams = {param: [lower, upper]}
    # print(f"Default gradients: {m.gradient}")

    return model, results

def regress(
        directory : Path, 
        inputFieldNames : list, 
        outputField : str, 
        spectralFeatures : list = None, 
        kernels : list = None,
        wRank : int = 3,
        normalise : bool = True, 
        peaksOnly : bool = True,
        plotModels : bool = True,
        sobol : bool = True,
        evaluate : bool = True,
        cvFolds : int = 5,
        cvRepeats : int = 2,
        printLatex : bool = False,
        writeToFile : bool = True,
        spec_xLabel : str = None,
        spec_yLabel : str = None,
        spec_xUnit : str = None,
        spec_yUnit : str = None):
    
    # Initialise results object
    results = ml_utils.GPResults()
    results.gpPackage = "GPy"

    if directory.name != "data":
        data_dir = directory / "data"
    else:
        data_dir = directory
    data_files = glob.glob(str(data_dir / "*.nc")) 
    results.directory = str(directory.resolve())

    # Input data
    inputs = {inp : [] for inp in inputFieldNames}
    inputs = ml_utils.read_data(data_files, inputs, with_names = False, with_coords = False)
    results.inputNames = inputFieldNames

    # Output data
    outputs = {outputField : []}
    outputs = ml_utils.read_data(data_files, outputs, with_names = True, with_coords=True)
    results.outputSpectrumName = outputField

    # Transformation of B0angle
    if "B0angle" in inputs:
        transf = np.array(inputs["B0angle"])
        inputs["B0angle"] = np.abs(transf - 90.0)
    
    logFields = [
        "maxPeakPower", 
        "maxPeakCoordinate", 
        "meanPeak4Powers", 
        "meanPeak3Powers", 
        "meanPeak2Powers",
        "maxPeakProminence",
        "meanPeakProminence",
        # "varPeakProminence",
        "maxPeakPowerProminence",
        "activeRegionVarPeakSeparations",
        "activeRegionMeanCoordWidths",
        "activeRegionVarCoordWidths",
        "totalActiveCoordinateProportion",
        "totalActivePowerProportion",
        "spectrumVar",
        "spectrumSum",
        "spectrumMean",
        "backgroundDensity",
        # "B0strength",
        # "B0angle",
        "beamFraction"
    ]

    inputs, features, indices, unextracted_indices = create_spectral_features(directory, outputField, inputs, outputs, spectralFeatures, logFields, peaksOnly, spec_xLabel, spec_yLabel, spec_xUnit, spec_yUnit)

    input_names = list(inputs.keys())
    input_array = np.array([a[indices] for a in np.array(list(inputs.values()))])

    output_feature_names = np.array(list(features.keys()))
    output_feature_values = np.array(list(features.values()))
    output_features_array = np.array(output_feature_values)

    results.logFields = np.intersect1d(np.union1d(input_names, output_feature_names), logFields)
    results.outputNames = output_feature_names

    # Normalise
    if normalise:
        results.normalised = True
        input_array, orig_in_mean, orig_in_sd = ml_utils.normalise_dataset(input_array)
        results.original_input_means = {n : m for n, m in zip(input_names, orig_in_mean)}
        results.original_input_stdevs = {n : s for n, s in zip(input_names, orig_in_sd)}
        output_features_array, orig_out_mean, orig_out_sd = ml_utils.normalise_dataset(output_features_array)
        results.original_output_means = {n : m for n, m in zip(input_names, orig_out_mean)}
        results.original_output_stdevs = {n : s for n, s in zip(input_names, orig_out_sd)}
    input_columns = input_array.T
    output_features_columns = output_features_array.T

    # Plot training data
    # input = input_columns
    # output = output_features_columns
    # for f in range(len(output_feature_names)):
    #     x1 = input[:,0]
    #     x2 = input[:,1]
    #     x3 = input[:,2]
    #     x4 = input[:,3]
    #     x1_name = input_names[0]
    #     x2_name = input_names[1]
    #     x3_name = input_names[2]
    #     x4_name = input_names[3]
    #     plot_4inputs_training_data(output[:,f], output_feature_names[f], x1, x2, x3, x4, x1_name, x2_name, x3_name, x4_name)
    #     plt.close("all")

    ##### Attempt regression
    # Format data
    # inputs_formatted = ml_utils.convert_input_for_multi_output_GPy_model(features_peaksOnly_normValues_columns, num_outputs=4)
    # Set up kernels
    input_dim = input_columns.shape[1]
    output_dim = output_features_columns.shape[1]
    results.numFeatures = input_dim
    results.numObservations = input_columns.shape[0]
    results.numOutputs = output_dim
    print("----------------------------------------------------------------------------------------------")
    print(f"Regressing {input_dim} inputs ({input_names}) against {output_dim} outputs ({output_feature_names})")
    print("----------------------------------------------------------------------------------------------")
    X_in = input_columns
    Y_out = output_features_columns
    all_Y = []
    for i in range(output_dim):
        all_Y.append(Y_out[:,i].reshape(-1,1))

    if wRank > output_dim:
        print("wRank is > output dimensions, fixing to the number of outputs")
        wRank = output_dim
    kernel_names = kernels if kernels is not None else ["white", "linear", "ratQuad"]
    results.kernelNames = kernel_names
    results.wRank = wRank
    m, results = get_model(kernels=kernel_names, X_in=X_in, Y_out=all_Y, results=results, W_rank=wRank, bound_length=True)
    m.optimize(messages=True)
    print(m)
    results.model = m.to_dict(save_data=True)
    results.fitSuccess = True
    # for part in m.kern.parts:
    #     sigs = part.get_most_significant_input_dimensions()
    #     if None not in sigs:
    #         print(f'{part.name}: 3 most significant input dims (descending): {feature_names[sigs[0]-1]}, {feature_names[sigs[1]-1]}, {feature_names[sigs[2]-1]}')

    # Visualise model
    if plotModels:
        plot_outputs(m, input_names, output_feature_names, (-3,3))

    # Analyse model (sobol)
    if sobol:
        results = sobol_analysis(m, input_columns, input_names, output_features_columns, output_feature_names, results, printLatex=printLatex)

    # Evaluate model (CV)
    if evaluate:
        results = evaluate_model_k_folds(m, kernel_names, results, output_feature_names, orig_out_mean, orig_out_sd, logFields, k_folds=cvFolds, n_repeats=cvRepeats, printLatex=printLatex)
        # good_test_idx, bad_test_idx = evaluate_model_loo(m, kernel_names, output_names)
        # print(f"Simulations predicted better than baseline by GP: {good_test_idx}")
        # print(f"Simulations predicted worse than baseline by GP: {bad_test_idx}")
        # good_test_idx = np.array(good_test_idx)
        # bad_test_idx = np.array(bad_test_idx)
        # good_points = [np.array(arr)[good_test_idx] for arr in output_array]
        # bad_points = [np.array(arr)[bad_test_idx] for arr in output_array]
        # epoch_utils.my_matrix_plot(data_series=[good_points, bad_points], series_labels=["Simulations perdicted better by GP", "Simulations predicted worse by GP"], parameter_labels=output_names, plot_style="hdi", equalise_pdf_heights=False, filename="/home/era536/Documents/for_discussion/2025.08.07/R2_matrix.png")

    if writeToFile:
        savepath = directory / f"gp_results__package_GPy__outputs_{'-'.join(output_feature_names)}__kernels_{'-'.join(kernel_names)}__Wrank_{wRank}.json"
        ml_utils.write_GP_result_to_file(results, savepath)
        

def evaluate_model_loo(model : GPy.Model, kernel_names, output_names):
    num_features = model.X.shape[1]-1
    num_outputs = len(output_names)
    num_samples = int(model.Y.shape[0]/num_outputs)
    print(num_features)
    print(num_samples)
    x_dummy = np.arange(num_samples)
    x_data = model.X[:num_samples,:]
    y_data = model.Y.reshape(num_outputs, num_samples).T

    # Repeated K Folds
    loo = LeaveOneOut()
    fold_R2s = []
    fold_RMSEs = []
    fold_individualField_RMSEs = []
    test_indices_yielding_positive_R2 = []
    test_indices_yielding_negative_R2 = []
    for fold, (train, test) in enumerate(loo.split(x_dummy)):

        print(f'FOLD {fold}:')
        print(f'TEST IDX: {test}')

        print(f"Fold {fold} -- Preparing data....")
        x_train = x_data[train,:num_features]
        y_train = y_data[train,:]

        x_test = x_data[test,:num_features]
        y_test = y_data[test,:]

        y_test_formatted = y_test.flatten('F')

        num_test_samples = len(test)

        fold_Y = []
        for i in range(num_outputs):
            fold_Y.append(y_train[:,i].reshape(-1,1))

        # Rebuild model
        print(f"Fold {fold} -- Training data....")
        m = get_model(kernels=kernel_names, X_in=x_train, Y_out=fold_Y, W_rank=3, bound_length=True)
        m.optimize(messages=True)
        # Having to rebuild model above is not ideal. It seems set_XY is not properly implemented for coregionalized models where X and Y are lists.

        print(f"Fold {fold} -- Predicting data....")
        output_num = np.zeros((num_test_samples))
        x_test_formatted = x_test
        for i in range(1, num_outputs):
            output_num = np.concatenate((output_num, np.ones(num_test_samples)*i))
            x_test_formatted = np.vstack([x_test_formatted, x_test])
        output_num = output_num[:,None]
        x_test_formatted = np.hstack([x_test_formatted, output_num])

        # Tell GPy which indices correspond to which output noise model
        noise_dict = {'output_index' : x_test_formatted[:,-1].astype(int)}

        y_preds, y_var_diag = m.predict(x_test_formatted, Y_metadata = noise_dict)

        overallR2 = r2_score(y_test_formatted, y_preds)
        overallRMSE = root_mean_squared_error(y_test_formatted, y_preds)
        print(f'Fold {fold} -- Overall R^2: {overallR2}')
        print(f'Fold {fold} -- Overall RMSE: {overallRMSE}')

        assert len(y_preds)//num_test_samples == num_outputs
        fold_individualField_RMSEs.append([])
        for i in range(num_outputs):
            field = output_names[i]
            field_y_test = y_test[:,i]
            field_y_preds = y_preds[int(i*num_test_samples):int((i+1)*num_test_samples)]
            field_RMSE = root_mean_squared_error(field_y_test, field_y_preds)
            fold_individualField_RMSEs[-1].append(field_RMSE)
            print(f'Fold {fold} -- {field} RMSE: {field_RMSE} ()')

        if overallR2 > 0.0:
            test_indices_yielding_positive_R2.append(test[0])
        else:
            test_indices_yielding_negative_R2.append(test[0])

        if overallR2 <= -10.0:
            y_preds_flat = [p[0] for p in y_preds]
            plt.scatter(output_names, y_preds_flat, label="predictions")
            plt.scatter(output_names, y_test_formatted, label="true values")
            plt.title(f'Fold: {fold}')
            plt.legend()
            plt.show()

            res = linregress(y_test_formatted, y_preds_flat)
            plt.scatter(y_test_formatted, y_preds_flat)
            xvals = np.linspace(np.min([np.min(y_test_formatted), np.min(y_preds_flat)]), np.max([np.max(y_test_formatted), np.max(y_preds_flat)]))
            plt.plot(xvals, res.intercept + res.slope*xvals, "r", label="fit")
            plt.title(f'Fold: {fold}')
            plt.legend()
            plt.xlabel("true values")
            plt.ylabel("predictions")
            plt.show()

        fold_R2s.append(overallR2)
        fold_RMSEs.append(overallRMSE)

    print(f"Mean overall R^2 across leave-one-out CV ({len(fold_R2s)} tests): {np.mean(fold_R2s):.5f}+-{np.std(fold_R2s)/np.sqrt(len(fold_R2s)):.5f}")
    print(f"Mean overall RMSE across leave-one-out CV ({len(fold_R2s)} tests): {np.mean(fold_RMSEs):.5f}+-{np.std(fold_RMSEs)/np.sqrt(len(fold_RMSEs)):.5f}")
    fold_individualField_RMSEs = np.array(fold_individualField_RMSEs).T
    for i in range(num_outputs):
        field = output_names[i]
        rmse = fold_individualField_RMSEs[i]
        print(f"Mean {field} RMSE across leave-one-out CV ({len(fold_R2s)} tests): {np.mean(rmse):.5f}+-{np.std(rmse)/np.sqrt(len(rmse)):.5f}")

    return test_indices_yielding_positive_R2, test_indices_yielding_negative_R2

def evaluate_model_k_folds(
        model : GPy.Model, 
        kernel_names, 
        results : ml_utils.GPResults,
        output_names, 
        orig_out_means, 
        orig_out_sds, 
        log_fields, 
        k_folds = 5, 
        n_repeats = 2, 
        printLatex = False) -> ml_utils.GPResults:
    
    results.cvStrategy = "k_folds"
    results.cvFolds = k_folds
    results.cvRepeats = n_repeats

    num_features = model.X.shape[1]-1
    num_outputs = len(output_names)
    num_samples = int(model.Y.shape[0]/num_outputs)
    print(num_features)
    print(num_samples)
    x_dummy = np.arange(num_samples)
    x_data = model.X[:num_samples,:]
    y_data = model.Y.reshape(num_outputs, num_samples).T

    # Repeated K Folds
    rkf = RepeatedKFold(n_splits=k_folds, n_repeats=n_repeats)
    fold_wise_R2s = []
    fold_and_field_wise_R2s = []
    fold_wise_RMSEs = []
    fold_and_field_wise_SEs = []
    fold_and_field_wise_SLLs = []
    all_test_data_and_predictions = {"true_vals" : [], "predictions" : [], "prediction_vars" : []} # Dict of all test data and predictions
    for fold, (train, test) in enumerate(rkf.split(x_dummy)):
        print(f'FOLD {fold}:')
        # print(f'     TRAIN IDX: {train},\n     TEST IDX: {test}')

        print(f"Fold {fold} -- Preparing data....")

        fold_and_field_wise_R2s.append([])
        fold_and_field_wise_SEs.append([])
        fold_and_field_wise_SLLs.append([])

        x_train = x_data[train,:num_features]
        y_train = y_data[train,:]

        x_test = x_data[test,:num_features]
        y_test = y_data[test,:]

        y_test_flat = y_test.flatten('F')

        num_test_samples = len(test)

        fold_Y = []
        for i in range(num_outputs):
            fold_Y.append(y_train[:,i].reshape(-1,1))

        # Rebuild model
        print(f"Fold {fold} -- Training data....")
        m, results = get_model(kernels=kernel_names, X_in=x_train, Y_out=fold_Y, results=results, W_rank=results.wRank, bound_length=True)
        m.optimize(messages=True)
        # Having to rebuild model above is not ideal. It seems set_XY is not properly implemented for coregionalized models where X and Y are lists.

        print(f"Fold {fold} -- Predicting data....")
        output_num = np.zeros((num_test_samples))
        x_test_formatted = x_test
        for i in range(1, num_outputs):
            output_num = np.concatenate((output_num, np.ones(num_test_samples)*i))
            x_test_formatted = np.vstack([x_test_formatted, x_test])
        output_num = output_num[:,None]
        x_test_formatted = np.hstack([x_test_formatted, output_num])

        # Tell GPy which indices correspond to which output noise model
        noise_dict = {'output_index' : x_test_formatted[:,-1].astype(int)}

        y_preds, y_var_diag = m.predict(x_test_formatted, Y_metadata = noise_dict)

        # R2, RMSE, MSLL
        assert y_preds.shape[1] == 1
        y_preds_flat = y_preds.flatten()
        assert y_var_diag.shape[1] == 1
        y_vars_flat = y_var_diag.flatten()
        all_test_data_and_predictions["true_vals"].extend(y_test_flat)
        all_test_data_and_predictions["predictions"].extend(y_preds_flat)
        all_test_data_and_predictions["prediction_vars"].extend(y_vars_flat)
        foldR2 = r2_score(y_test_flat, y_preds_flat)
        fold_wise_R2s.append(foldR2)
        foldRMSE = root_mean_squared_error(y_test_flat, y_preds_flat)
        # foldSLL = ml_utils.standardized_log_loss(y_preds_flat, y_vars_flat, y_test_flat, np.mean(y_train), np.var(y_train))
        print(f'Fold {fold} -- Overall R^2: {foldR2}, Overall RMSE: {foldRMSE}')
        
        assert len(y_preds)//num_test_samples == num_outputs
        for i in range(num_outputs):
            
            field = output_names[i]
            
            field_y_test = y_test[:,i]
            field_y_preds = y_preds_flat[int(i*num_test_samples):int((i+1)*num_test_samples)]
            field_y_vars = y_vars_flat[int(i*num_test_samples):int((i+1)*num_test_samples)]
            
            field_R2 = r2_score(field_y_test, field_y_preds)
            field_RMSE = root_mean_squared_error(field_y_test, field_y_preds)
            field_SE = ml_utils.squared_error(field_y_preds, field_y_test)
            field_SLL = ml_utils.standardized_log_loss(field_y_preds, field_y_vars, field_y_test, np.mean(y_train[:,i]), np.var(y_train[:,i]))
            fold_and_field_wise_R2s[-1].append(field_R2)
            fold_and_field_wise_SEs[-1].append(field_SE)
            fold_and_field_wise_SLLs[-1].append(field_SLL)
            
            print(f'Fold {fold} -- {field} R^2: {field_R2:.5f}, RMSE: {field_RMSE:.5f} ({ml_utils.denormalise_datapoint(field_RMSE, orig_out_means[i], orig_out_sds[i], field in log_fields):.5f}), MSLL: {np.mean(field_SLL):.5f}')

    # Write results
    results.cvR2_mean = np.mean(fold_wise_R2s)
    results.cvR2_var = np.var(fold_wise_R2s)
    results.cvR2_stderr = np.std(fold_wise_R2s)/np.sqrt(len(fold_wise_R2s))
    rmses, rmses_var, rmses_stdErr = ml_utils.root_mean_squared_error(all_test_data_and_predictions["predictions"], all_test_data_and_predictions["true_vals"])
    results.cvRMSE_mean = rmses
    results.cvRMSE_var = rmses_var
    results.cvRMSE_stderr = rmses_stdErr
    all_SLLs = []
    for fold_data in fold_and_field_wise_SLLs:
        for field_data in fold_data:
            all_SLLs.extend(field_data)
    results.cvMSLL_mean = np.mean(all_SLLs)
    results.cvMSLL_var = np.var(all_SLLs)
    results.cvMSLL_stderr = np.std(all_SLLs)/np.sqrt(len(all_SLLs))
    
    # Debug prints
    print(f"Mean overall R^2 across {k_folds} folds and {n_repeats} repeats: {results.cvR2_mean:.5f}+-{results.cvR2_stderr:.5f}  (want >0.0 and close to 1.0)")
    print(f"Mean overall RMSE across {k_folds} folds and {n_repeats} repeats: {results.cvRMSE_mean:.5f}+-{results.cvRMSE_stderr:.5f}  (want <S.D. and close to 0.0)")
    denorms = []
    for i in range(len(output_names)):
        denorms.append(f"{ml_utils.denormalise_datapoint(results.cvRMSE_mean, orig_out_means[i], orig_out_sds[i], output_names[i] in log_fields):.5f} {output_names[i]}")
    print(f"This RMSE is equivalent to {', '.join(denorms)}.")
    print(f"Mean overall SLL across {k_folds} folds and {n_repeats} repeats: {results.cvMSLL_mean:.5f}+-{results.cvMSLL_stderr:.5f}  (want <0.0)")
    if printLatex:
        print("---------------------------------------------- LaTeX ----------------------------------------------------")
        print(f"overall & ${results.cvR2_mean:.5f}\pm{results.cvR2_stderr:.5f}$ & ${results.cvRMSE_mean:.5f}\pm{results.cvRMSE_stderr:.5f}$ & ${results.cvMSLL_mean:.5f}\pm{results.cvMSLL_stderr:.5f}$\\\\")
    
    
    field_wise_R2s = np.array(fold_and_field_wise_R2s).T
    field_wise_SEs = [[] for _ in range(num_outputs)]
    field_wise_SLLs = [[] for _ in range(num_outputs)]
    for fold_data in fold_and_field_wise_SLLs:
        for f in range(num_outputs):
            field_wise_SLLs[f].extend(fold_data[f])
    for fold_data in fold_and_field_wise_SEs:
        for f in range(num_outputs):
            field_wise_SEs[f].extend(fold_data[f])
    
    for i in range(num_outputs):
        field = output_names[i]
        r2s = field_wise_R2s[i]
        rmse_mean = np.sqrt(np.mean(field_wise_SEs[i]))
        rmse_se = np.std(field_wise_SEs[i])/np.sqrt(len(field_wise_SEs[i]))
        slls = field_wise_SLLs[i]
        if printLatex:
            print(f"{field} & ${np.mean(r2s):.5f}\pm{np.std(r2s)/np.sqrt(len(r2s)):.5f}$ & ${rmse_mean:.5f}\pm{rmse_se:.5f} & ${np.mean(slls):.5f}\pm{np.std(slls)/np.sqrt(len(slls)):.5f}$\\\\")
        else:
            print(f"Mean {field} R^2 across {k_folds} folds and {n_repeats} repeats: {np.mean(r2s):.5f}+-{np.std(r2s)/np.sqrt(len(r2s)):.5f}")
            print(f"Mean {field} RMSE across {k_folds} folds and {n_repeats} repeats: {rmse_mean:.5f}+-{rmse_se:.5f}")
            print(f"Mean {field} SLL across {k_folds} folds and {n_repeats} repeats: {np.mean(slls):.5f}+-{np.std(slls)/np.sqrt(len(slls)):.5f}")
    if printLatex:
        print("---------------------------------------------------------------------------------------------------------")

    return results

def sobol_analysis(
        model : GPy.Model, 
        features : np.ndarray, 
        features_names : list, 
        outputs : np.ndarray, 
        output_names : list, 
        results : ml_utils.GPResults,
        printLatex : bool = True,
        noTitle : bool = False) -> ml_utils.GPResults:

    num_inputs = len(features_names)
    num_outputs = len(output_names)

    # SALib SOBOL indices
    sp = ProblemSpec({
        'num_vars': num_inputs,
        'names': list(features_names),
        'bounds': [[np.min(column), np.max(column)] for column in features.T]
    })
    num_samples = int(2**12)
    results.sobolSamples = num_samples
    test_values = salsamp.sobol.sample(sp, num_samples, calc_second_order = (num_inputs > 1))

    # Format test values for each output
    num_sample_points = test_values.shape[0]
    output_num = np.zeros((num_sample_points))
    test_values_formatted = test_values
    for i in range(1, num_outputs):
        output_num = np.concatenate((output_num, np.ones(num_sample_points)*i))
        test_values_formatted = np.vstack([test_values_formatted, test_values])
    output_num = output_num[:,None]
    test_values_formatted = np.hstack([test_values_formatted, output_num])

    # Tell GPy which indices correspond to which output noise model
    noise_dict = {'output_index' : test_values_formatted[:,-1].astype(int)}
    
    # Sample predictions here
    y_prediction, y_var_diag = model.predict(test_values_formatted, Y_metadata = noise_dict) # y_prediction is 1-D for all of output 0, then all of output 1 etc.
    print(f"{model.name} predictions for {output_names} -- y_std: {np.sqrt(y_var_diag)}")
    results.sobolIndicesO1 = dict.fromkeys(output_names)
    results.sobolIndicesO2 = dict.fromkeys(output_names)
    results.sobolIndicesTotal = dict.fromkeys(output_names)
    
    for n in range(num_outputs):
        output_name = output_names[n]
        print(f"SOBOL analysing {model.name} model of {features_names} against {output_name}....")

        names_and_confs = features_names + [f"{name}_conf" for name in features_names]
        results.sobolIndicesO1[output_name] = dict.fromkeys(names_and_confs)
        results.sobolIndicesO2[output_name] = dict.fromkeys(names_and_confs)
        results.sobolIndicesTotal[output_name] = dict.fromkeys(names_and_confs)

        predictions = y_prediction[n*num_sample_points:(n+1)*num_sample_points][:,0]
        sobol_indices = sobol.analyze(sp, predictions, print_to_console=True, calc_second_order = (num_inputs > 1))

        print(f"Sobol indices for output {output_name}:")
        print(sobol_indices)
        # print(f"Sum of SOBOL indices: ST = {np.sum(sobol_indices['ST'])}, S1 = {np.sum(sobol_indices['S1'])}, abs(S1) = {np.sum(abs(sobol_indices['S1']))} S2 = {np.nansum(sobol_indices['S2'])}, abs(S2) = {np.nansum(abs(sobol_indices['S2']))}")
        plt.rcParams["figure.figsize"] = (14,10)
        #fig, ax = plt.subplots()
        Si_df = sobol_indices.to_df()
        _, ax = plt.subplots(1, len(Si_df), sharey=True)
        CONF_COLUMN = "_conf"
        st_df = Si_df[0].get("ST")
        s1_df = Si_df[1].get("S1")
        s2_df = Si_df[2].get("S2")

        # Print and record Sobol
        if printLatex:
            print("---------------------------------------------- LaTeX ----------------------------------------------------")
            print(output_name)
        for idx, st in enumerate(st_df):
            print(st_df.axes)
            print(st_df.index)
            feature_name = str(st_df.index[idx])
            results.sobolIndicesTotal[output_name][feature_name] = st
            results.sobolIndicesTotal[output_name][f"{feature_name}_conf"] = Si_df[0].get('ST_conf')[idx]
            if printLatex:
                print(f"& {feature_name} & ${st:.3f}\pm{Si_df[0].get('ST_conf')[idx]:.3f}$\\\\")
            else:
                print(f"Input: {feature_name} Sobol: {st} Conf: {Si_df[0].get('ST_conf')[idx]}")
        if printLatex:
            print("---------------------------------------------------------------------------------------------------------")
        for idx, s1 in enumerate(s1_df):
            feature_name = str(s1_df.index[idx])
            results.sobolIndicesO1[output_name][feature_name] = s1
            results.sobolIndicesO1[output_name][f"{feature_name}_conf"] = Si_df[1].get('S1_conf')[idx]
        for idx, s2 in enumerate(s2_df):
            feature_name = str(s2_df.index[idx])
            results.sobolIndicesO2[output_name][feature_name] = s2
            results.sobolIndicesO2[output_name][f"{feature_name}_conf"] = Si_df[2].get('S2_conf')[idx]

        # Plot Sobol
        for idx, f in enumerate(Si_df):
            
            conf_cols = f.columns.str.contains(CONF_COLUMN)

            confs = f.loc[:, conf_cols]
            confs.columns = [c.replace(CONF_COLUMN, "") for c in confs.columns]

            Sis = f.loc[:, ~conf_cols]

            ax[idx] = Sis.plot(kind="bar", yerr=confs, ax=ax[idx])
        print(plt.ylim())
        plt.subplots_adjust(bottom=0.3)
        if not noTitle:
            plt.title(f"GPy {model.name} kernel: {output_names[n]}")
        plt.tight_layout()
        plt.show()
        plt.close()
    
    return results

# Single-valued features only for now
def create_spectral_features(
        directory : Path, 
        field : str, 
        input_data : dict, 
        spectrum_data : dict, 
        features : list = None, 
        log_fields = [], 
        peaks_only : bool = True,
        xLabel : str = None,
        yLabel : str = None,
        xUnit : str = None,
        yUnit : str = None):

    featureSets = []

    save_path = directory / "feature_extraction"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    write_path = directory / f"{field.replace('/', '-')}_features.csv"
    if os.path.exists(write_path):
        # Features file exists, read data
        featureSets = ml_utils.read_spectral_features_from_csv(write_path)
    else:
        for spectrum_idx in range(len(spectrum_data[field])):
            featureSets.append(ml_utils.extract_features_from_1D_power_spectrum(
                spectrum=spectrum_data[field][spectrum_idx], 
                coordinates=spectrum_data[field + "_coords"][spectrum_idx], 
                spectrum_name=spectrum_data["sim_ids"][spectrum_idx], 
                savePath=save_path,
                xLabel=xLabel,
                yLabel=yLabel,
                xUnit=xUnit,
                yUnit=yUnit))
        ml_utils.write_spectral_features_to_csv(write_path, featureSets)

    singleValueFields = [f.name for f in fields(ml_utils.SpectralFeatures1D) if f.type is float or f.type is int]
    if features is None or "all" in features:
        features = singleValueFields
    features_to_extract = np.intersect1d(features, singleValueFields)
    extracted_features = {f : [] for f in features_to_extract}
    extracted_indices = set()
    unextracted_indices = set()
    if peaks_only:
        for simIndex in range(len(featureSets)):
            if featureSets[simIndex].peaksFound:
                for feature in features_to_extract:
                    extracted_features[feature].append(featureSets[simIndex].__dict__[feature])
                extracted_indices.add(simIndex)
            else:
                unextracted_indices.add(simIndex)
    else:
        for simIndex in range(len(featureSets)):
            for feature in features_to_extract:
                extracted_features[feature].append(featureSets[simIndex].__dict__[feature])
                extracted_indices.add(simIndex)

    for f_name in features:
        if f_name in log_fields:
            extracted_features[f_name] = np.log10(extracted_features[f_name])
    for f_name in input_data.keys():
        if f_name in log_fields:
            input_data[f_name] = np.log10(input_data[f_name])
    
    if unextracted_indices:
        series_labels = ["spectral peaks", "no spectral peaks"]
        param_labels = list(input_data.keys())
        ex_data = []
        unex_data = []
        for p in range(len(param_labels)):
            param_name = param_labels[p]
            ex_data.append([input_data[param_name][i] for i in extracted_indices])
            unex_data.append([input_data[param_name][i] for i in unextracted_indices])
            if param_name in log_fields:
                param_labels[p] = "log " + param_name
            
        epoch_utils.my_matrix_plot([ex_data, unex_data], series_labels=series_labels, parameter_labels=param_labels, show = False, filename=save_path/"peaks_matrix.png", plot_style="hdi", equalise_pdf_heights=False)
        plt.clf()

    return input_data, extracted_features, list(extracted_indices), list(unextracted_indices)

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
        help="Spectral fields to use for GP input.",
        required = True,
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--outputField",
        action="store",
        help="Fields to use for GP output.",
        required = False,
        type=str
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
        "--spectralOutputs",
        action="store",
        help="Spectral features to extract from the input spectrum.",
        required = False,
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--kernels",
        action="store",
        help="Kernels to use for GP.",
        required = True,
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--wRank",
        action="store",
        help="W rank for coregionalization matrix.",
        required = False,
        type=int
    )
    parser.add_argument(
        "--spectrumXLabel",
        action="store",
        help="X-axis label of input spectrum for plotting.",
        required = False,
        type=str
    )
    parser.add_argument(
        "--spectrumYLabel",
        action="store",
        help="Y-axis label of input spectrum for plotting.",
        required = False,
        type=str
    )
    parser.add_argument(
        "--spectrumXUnits",
        action="store",
        help="X-axis units of input spectrum for plotting/output.",
        required = False,
        type=str
    )
    parser.add_argument(
        "--spectrumYUnits",
        action="store",
        help="Y-axis units of input spectrum for plotting/output.",
        required = False,
        type=str
    )
    parser.add_argument(
        "--normalise",
        action="store_true",
        help="Normalise model inputs and outputs to zero mean and unit variance.",
        required = False
    )
    parser.add_argument(
        "--peaksOnly",
        action="store_true",
        help="Use only spectra where peaks are detected.",
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
        "--cvFolds",
        action="store",
        help="Folds (k) for cross-validation.",
        required = False,
        type=int
    )
    parser.add_argument(
        "--cvRepeats",
        action="store",
        help="Resampling repeats for cross-validation.",
        required = False,
        type=int
    )
    parser.add_argument(
        "--plotModels",
        action="store_true",
        help="Plot regression models.",
        required = False
    )
    parser.add_argument(
        "--printLatex",
        action="store_true",
        help="Print results formatted for LaTeX tables.",
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
        "--writeToFile",
        action="store_true",
        help="Write results to file.",
        required = False
    )

    args = parser.parse_args()

    regress(args.dir, args.inputFields, args.outputField, args.spectralOutputs, args.kernels, args.wRank, args.normalise, args.peaksOnly, args.plotModels, args.sobol, args.evaluate, args.cvFolds, args.cvRepeats, args.printLatex, args.writeToFile, args.spectrumXLabel, args.spectrumYLabel, args.spectrumXUnits, args.spectrumYUnits)
    # demo()