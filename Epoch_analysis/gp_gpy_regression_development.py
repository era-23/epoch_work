import argparse
import glob
import os
import GPy
import SALib.sample as salsamp
import ml_utils
import epoch_utils
import pylab as pb
import numpy as np
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
        this_axis = training_data[:,input_index]
        other_axes = np.delete(training_data, input_index, axis=1)
        geometric_distances = []
        for t_idx in range(len(this_axis)):
            t = this_axis[t_idx]
            rows = [(t-x)**2 for x in other_axes[t_idx,:]]
            geometric_distances.append(np.sqrt(np.sum(rows)))
        g = geometric_distances / np.max(geometric_distances)
        dist_from_0 = np.clip(1.0 - g, a_min = 0.1, a_max = None)
        dist_from_0 = exponential_alpha_scaling_factor**dist_from_0 / np.max(exponential_alpha_scaling_factor**dist_from_0)
        p.set_color("red")
        p.set_alpha(dist_from_0)
        
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
        this_axis = training_data[:,input_index]
        other_axes = np.delete(training_data, input_index, axis=1)
        geometric_distances = []
        for t_idx in range(len(this_axis)):
            t = this_axis[t_idx]
            rows = [(t-x)**2 for x in other_axes[t_idx,:]]
            geometric_distances.append(np.sqrt(np.sum(rows)))
        g = geometric_distances / np.max(geometric_distances)
        dist_from_0 = np.clip(1.0 - g, a_min = 0.1, a_max = None)
        dist_from_0 = exponential_alpha_scaling_factor**dist_from_0 / np.max(exponential_alpha_scaling_factor**dist_from_0)
        p.set_color("red")
        p.set_alpha(dist_from_0)

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
        this_axis = training_data[:,input_index]
        other_axes = np.delete(training_data, input_index, axis=1)
        geometric_distances = []
        for t_idx in range(len(this_axis)):
            t = this_axis[t_idx]
            rows = [(t-x)**2 for x in other_axes[t_idx,:]]
            geometric_distances.append(np.sqrt(np.sum(rows)))
        g = geometric_distances / np.max(geometric_distances)
        dist_from_0 = np.clip(1.0 - g, a_min = 0.1, a_max = None)
        dist_from_0 = exponential_alpha_scaling_factor**dist_from_0 / np.max(exponential_alpha_scaling_factor**dist_from_0)
        p.set_color("red")
        p.set_alpha(dist_from_0)

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
        this_axis = training_data[:,input_index]
        other_axes = np.delete(training_data, input_index, axis=1)
        geometric_distances = []
        for t_idx in range(len(this_axis)):
            t = this_axis[t_idx]
            rows = [(t-x)**2 for x in other_axes[t_idx,:]]
            geometric_distances.append(np.sqrt(np.sum(rows)))
        g = geometric_distances / np.max(geometric_distances)
        dist_from_0 = np.clip(1.0 - g, a_min = 0.1, a_max = None)
        dist_from_0 = exponential_alpha_scaling_factor**dist_from_0 / np.max(exponential_alpha_scaling_factor**dist_from_0)
        p.set_color("red")
        p.set_alpha(dist_from_0)
        # Display
        plt.show()
        plt.close("all")

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

def get_model(kernels, X_in, Y_out, W_rank = 3, bound_length = True) -> GPy.Model:
    
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
        model['.*lengthscale'].constrain_bounded(lower=0.0, upper=0.2)
    # print(f"Default gradients: {m.gradient}")

    return model

def regress(
        directory : Path, 
        inputFieldNames : list, 
        outputField : str, 
        spectralFeatures : list = None, 
        normalise : bool = True, 
        peaksOnly : bool = True,
        spec_xLabel : str = None,
        spec_yLabel : str = None,
        spec_xUnit : str = None,
        spec_yUnit : str = None):
    
    if directory.name != "data":
        data_dir = directory / "data"
    else:
        data_dir = directory
    data_files = glob.glob(str(data_dir / "*.nc")) 

    # Input data
    inputs = {inp : [] for inp in inputFieldNames}
    inputs = ml_utils.read_data(data_files, inputs, with_names = False, with_coords = False)

    # Output data
    outputs = {outputField : []}
    outputs = ml_utils.read_data(data_files, outputs, with_names = True, with_coords=True)

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

    # Normalise
    if normalise:
        input_array = np.array([(p - np.nanmean(p)) / np.nanstd(p) for p in input_array])
        output_features_array = np.array([(f - np.nanmean(f)) / np.nanstd(f) for f in output_features_array])
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
    print("----------------------------------------------------------------------------------------------")
    print(f"Regressing {input_dim} inputs ({input_names}) against {output_dim} outputs ({output_feature_names})")
    print("----------------------------------------------------------------------------------------------")
    X_in = input_columns
    Y_out = output_features_columns
    all_Y = []
    for i in range(output_dim):
        all_Y.append(Y_out[:,i].reshape(-1,1))

    # kernel_names = ["white", "linear", "ratQuad"]
    kernel_names = ["linear", "ratQuad"]
    m = get_model(kernels=kernel_names, X_in=X_in, Y_out=all_Y, W_rank=3, bound_length=True)
    m.optimize(messages=True)
    print(m)
    # for part in m.kern.parts:
    #     sigs = part.get_most_significant_input_dimensions()
    #     if None not in sigs:
    #         print(f'{part.name}: 3 most significant input dims (descending): {feature_names[sigs[0]-1]}, {feature_names[sigs[1]-1]}, {feature_names[sigs[2]-1]}')

    # Visualise model
    # plot_outputs(m, input_names, output_feature_names, (-3,3))

    # Analyse model (sobol)
    # sobol_analysis(m, input_columns, input_names, output_features_columns, output_feature_names)

    # Evaluate model (CV)
    evaluate_model_k_folds(m, kernel_names, output_feature_names)
    # good_test_idx, bad_test_idx = evaluate_model_loo(m, kernel_names, output_names)
    # print(f"Simulations predicted better than baseline by GP: {good_test_idx}")
    # print(f"Simulations predicted worse than baseline by GP: {bad_test_idx}")
    # good_test_idx = np.array(good_test_idx)
    # bad_test_idx = np.array(bad_test_idx)
    # good_points = [np.array(arr)[good_test_idx] for arr in output_array]
    # bad_points = [np.array(arr)[bad_test_idx] for arr in output_array]
    # epoch_utils.my_matrix_plot(data_series=[good_points, bad_points], series_labels=["Simulations perdicted better by GP", "Simulations predicted worse by GP"], parameter_labels=output_names, plot_style="hdi", equalise_pdf_heights=False, filename="/home/era536/Documents/for_discussion/2025.08.07/R2_matrix.png")

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
            print(f'Fold {fold} -- {field} RMSE: {field_RMSE}')

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

    print(f"Mean overall R^2 across leave-one-out CV ({len(fold_R2s)} tests): {np.mean(fold_R2s)}+-{np.std(fold_R2s)/np.sqrt(len(fold_R2s))}")
    print(f"Mean overall RMSE across leave-one-out CV ({len(fold_R2s)} tests): {np.mean(fold_RMSEs)}+-{np.std(fold_RMSEs)/np.sqrt(len(fold_RMSEs))}")
    fold_individualField_RMSEs = np.array(fold_individualField_RMSEs).T
    for i in range(num_outputs):
        field = output_names[i]
        rmse = fold_individualField_RMSEs[i]
        print(f"Mean {field} RMSE across leave-one-out CV ({len(fold_R2s)} tests): {np.mean(rmse)}+-{np.std(rmse)/np.sqrt(len(rmse))}")

    return test_indices_yielding_positive_R2, test_indices_yielding_negative_R2

def evaluate_model_k_folds(model : GPy.Model, kernel_names, output_names, k_folds = 7, n_repeats = 3):
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
    fold_R2s = []
    fold_individualField_R2s = []
    fold_RMSEs = []
    fold_individualField_RMSEs = []
    for fold, (train, test) in enumerate(rkf.split(x_dummy)):
        print(f'FOLD {fold}:')
        # print(f'     TRAIN IDX: {train},\n     TEST IDX: {test}')

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
        fold_individualField_R2s.append([])
        fold_individualField_RMSEs.append([])
        print(f'Fold {fold} -- Overall R^2: {overallR2}, Overall RMSE: {overallRMSE}')
        assert len(y_preds)//num_test_samples == num_outputs
        for i in range(num_outputs):
            field = output_names[i]
            field_y_test = y_test[:,i]
            field_y_preds = y_preds[int(i*num_test_samples):int((i+1)*num_test_samples)]
            field_R2 = r2_score(field_y_test, field_y_preds)
            field_RMSE = root_mean_squared_error(field_y_test, field_y_preds)
            fold_individualField_R2s[-1].append(field_R2)
            fold_individualField_RMSEs[-1].append(field_RMSE)
            print(f'Fold {fold} -- {field} R^2: {field_R2}, RMSE: {field_RMSE}')

        fold_R2s.append(overallR2)
        fold_RMSEs.append(overallRMSE)

    print(f"Mean overall R^2 across {k_folds} folds and {n_repeats} repeats: {np.mean(fold_R2s)}+-{np.std(fold_R2s)/np.sqrt(len(fold_R2s))}")
    print(f"Mean overall RMSE across {k_folds} folds and {n_repeats} repeats: {np.mean(fold_RMSEs)}+-{np.std(fold_RMSEs)/np.sqrt(len(fold_RMSEs))}")
    fold_individualField_R2s = np.array(fold_individualField_R2s).T
    fold_individualField_RMSEs = np.array(fold_individualField_RMSEs).T
    for i in range(num_outputs):
        field = output_names[i]
        r2s = fold_individualField_R2s[i]
        rmses = fold_individualField_RMSEs[i]
        print(f"Mean {field} R^2 across {k_folds} folds and {n_repeats} repeats: {np.mean(r2s)}+-{np.std(r2s)/np.sqrt(len(r2s))}")
        print(f"Mean {field} RMSE across {k_folds} folds and {n_repeats} repeats: {np.mean(rmses)}+-{np.std(rmses)/np.sqrt(len(rmses))}")

def sobol_analysis(model : GPy.Model, features : np.ndarray, features_names : list, outputs : np.ndarray, output_names : list, noTitle : bool = False):

    num_inputs = len(features_names)
    num_outputs = len(output_names)

    # SALib SOBOL indices
    sp = ProblemSpec({
        'num_vars': num_inputs,
        'names': list(features_names),
        'bounds': [[np.min(column), np.max(column)] for column in features.T]
    })
    test_values = salsamp.sobol.sample(sp, int(2**12), calc_second_order = (num_inputs > 1))

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
    
    for n in range(num_outputs):
        print(f"SOBOL analysing {model.name} model of {features_names} against {output_names[n]}....")
        predictions = y_prediction[n*num_sample_points:(n+1)*num_sample_points][:,0]
        sobol_indices = sobol.analyze(sp, predictions, print_to_console=True, calc_second_order = (num_inputs > 1))
        print(f"Sobol indices for output {output_names[n]}:")
        print(sobol_indices)
        # print(f"Sum of SOBOL indices: ST = {np.sum(sobol_indices['ST'])}, S1 = {np.sum(sobol_indices['S1'])}, abs(S1) = {np.sum(abs(sobol_indices['S1']))} S2 = {np.nansum(sobol_indices['S2'])}, abs(S2) = {np.nansum(abs(sobol_indices['S2']))}")
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
            plt.title(f"GPy {model.name} kernel: {output_names[n]}")
        plt.tight_layout()
        plt.show()
        plt.close()

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
        "--spectralFeatures",
        action="store",
        help="Spectral features to extract from the input spectrum.",
        required = False,
        type=str,
        nargs="*"
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

    args = parser.parse_args()

    regress(args.dir, args.inputFields, args.outputField, args.spectralFeatures, args.normalise, args.peaksOnly, args.spectrumXLabel, args.spectrumYLabel, args.spectrumXUnits, args.spectrumYUnits)
    # demo()