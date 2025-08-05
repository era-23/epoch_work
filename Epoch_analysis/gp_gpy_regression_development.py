import argparse
import glob
import os
import GPy
import SALib.sample as salsamp
import ml_utils
import pylab as pb
import numpy as np
from SALib import ProblemSpec
from SALib.analyze import sobol
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, LeavePOut
from sklearn.metrics import r2_score
from pathlib import Path
from matplotlib import pyplot as plt
from dataclasses import fields

def plot_4outputs_training_data(xTrain, xName, yOut_1, yOut_2, yOut_3, yOut_4, name_1, name_2, name_3, name_4):
    fig = plt.figure(figsize=(12,12))
    plt.suptitle(f'Input: {xName}')
    # Output 1
    ax1 = fig.add_subplot(411)
    ax1.set_ylabel(f'{name_1}')
    ax1.plot(xTrain,yOut_1,'kx',mew=1.5)
    # Output 2
    ax2 = fig.add_subplot(412)
    ax2.set_ylabel(f'{name_2}')
    ax2.plot(xTrain,yOut_2,'kx',mew=1.5)
    # Output 3
    ax3 = fig.add_subplot(413)
    ax3.set_ylabel(f'{name_3}')
    ax3.plot(xTrain,yOut_3,'kx',mew=1.5)
    # Output 4
    ax4 = fig.add_subplot(414)
    ax4.set_ylabel(f'{name_4}')
    ax4.plot(xTrain,yOut_4,'kx',mew=1.5)
    plt.show()

def plot_outputs(m, output_names, feature_names, xlim):
    
    num_datapoints = m.Y.shape[0]//len(output_names)

    for i in range(len(feature_names)):

        # fixed_inputs = [(f, 0.0) for f in range(0, len(feature_names)) if f != i]
        #fixed_inputs.append((len(feature_names), i))
        # print(fixed_inputs)

        fig = plt.figure(figsize=(18,8))
        f_name = feature_names[i]
        plt.suptitle(f'Input: {f_name}')

        #Output 1
        output_index = 0
        fixed_inputs = [(len(feature_names),output_index)]
        # fixed_inputs.append((len(feature_names),output_index))
        y_name = output_names[output_index]
        ax1 = fig.add_subplot(221)
        ax1.set_xlim(xlim)
        ax1.set_ylabel(y_name)
        m.plot(plot_limits=xlim,fixed_inputs=fixed_inputs, which_data_rows=slice(0, num_datapoints), ax=ax1, plot_data=True, visible_dims=[i])
        
        #Output 2
        output_index = 1
        fixed_inputs = [(len(feature_names),output_index)]
        # fixed_inputs.append((len(feature_names),output_index))
        y_name = output_names[output_index]
        ax2 = fig.add_subplot(222)
        ax2.set_xlim(xlim)
        ax2.set_ylabel(y_name)
        m.plot(plot_limits=xlim,fixed_inputs=fixed_inputs, which_data_rows=slice(num_datapoints, 2*num_datapoints), ax=ax2, plot_data=True, visible_dims=[i])
        
        #Output 3
        output_index = 2
        fixed_inputs = [(len(feature_names),output_index)]
        # fixed_inputs.append((len(feature_names),output_index))
        y_name = output_names[output_index]
        ax3 = fig.add_subplot(223)
        ax3.set_xlim(xlim)
        ax3.set_ylabel(y_name)
        m.plot(plot_limits=xlim,fixed_inputs=fixed_inputs, which_data_rows=slice(2*num_datapoints, 3*num_datapoints), ax=ax3, plot_data=True, visible_dims=[i])
        
        #Output 4
        output_index = 3
        fixed_inputs = [(len(feature_names),output_index)]
        # fixed_inputs.append((len(feature_names),output_index))
        y_name = output_names[output_index]
        ax4 = fig.add_subplot(224)
        ax4.set_xlim(xlim)
        ax4.set_ylabel(y_name)
        m.plot(plot_limits=xlim,fixed_inputs=fixed_inputs, which_data_rows=slice(3*num_datapoints,4*num_datapoints), ax=ax4, plot_data=True, visible_dims=[i])
        
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

def regress(directory : Path, inputFieldName : str, outputFields : list, spectralFeatures : list = None, normalise : bool = True, peaksOnly : bool = True):
    
    if directory.name != "data":
        data_dir = directory / "data"
    else:
        data_dir = directory
    data_files = glob.glob(str(data_dir / "*.nc")) 

    # Input data
    inputs = {inputFieldName : []}
    inputs = ml_utils.read_data(data_files, inputs, with_names = True, with_coords = True)

    # Output data
    outputs = {outp : [] for outp in outputFields}
    outputs = ml_utils.read_data(data_files, outputs, with_names = False, with_coords=False)

    # Transformation of B0angle
    if "B0angle" in outputs:
        transf = np.array(outputs["B0angle"])
        transf = 90.0 - transf
        outputs["B0angle"] = transf
    
    features, indices = create_spectral_features(directory, inputFieldName, inputs, spectralFeatures, peaksOnly)

    logFields = [
        "maxPeakPower", 
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
    for f_name in features:
        if f_name in logFields:
            features[f_name] = np.log10(features[f_name])
    for o_name in outputs:
        if o_name in logFields:
            outputs[o_name] = np.log10(outputs[o_name])

    feature_names = np.array(list(features.keys()))
    feature_values = np.array(list(features.values()))
    features_array = np.array(feature_values)

    output_names = list(outputs.keys())
    output_array = np.array([a[indices] for a in np.array(list(outputs.values()))])

    # Normalise
    if normalise:
        features_array = np.array([(f - np.nanmean(f)) / np.nanstd(f) for f in features_array])
        output_array = np.array([(p - np.nanmean(p)) / np.nanstd(p) for p in output_array])
    features_columns = features_array.T
    output_columns = output_array.T

    # Plot training data
    # output = output_columns
    # input = features_columns
    # for f in range(len(feature_names)):
    #     y1 = output[:,0]
    #     y2 = output[:,1]
    #     y3 = output[:,2]
    #     y4 = output[:,3]
    #     y1_name = output_names[0]
    #     y2_name = output_names[1]
    #     y3_name = output_names[2]
    #     y4_name = output_names[3]
    #     plot_4outputs_training_data(input[:,f], feature_names[f], y1, y2, y3, y4, y1_name, y2_name, y3_name, y4_name)

    ##### Attempt regression
    # Format data
    # inputs_formatted = ml_utils.convert_input_for_multi_output_GPy_model(features_peaksOnly_normValues_columns, num_outputs=4)
    # Set up kernels
    input_dim = features_columns.shape[1]
    output_dim = output_columns.shape[1]
    print("----------------------------------------------------------------------------------------------")
    print(f"Regressing {input_dim} inputs ({feature_names}) against {output_dim} outputs ({output_names})")
    print("----------------------------------------------------------------------------------------------")
    X_in = features_columns
    Y_out = output_columns
    all_Y = []
    for i in range(output_dim):
        all_Y.append(Y_out[:,i].reshape(-1,1))
    #k_list = [GPy.kern.RBF(input_dim, ARD=True) for _ in range(output_dim)]
    k_list = []
    k_list.append(GPy.kern.White(input_dim=input_dim))
    k_list.append(GPy.kern.Linear(input_dim=input_dim))
    k_list.append(GPy.kern.RatQuad(input_dim=input_dim, ARD = True))
    kerns = GPy.util.multioutput.LCM(input_dim=input_dim, num_outputs=output_dim, kernels_list=k_list, W_rank=3)
    # kerns_mult = np.prod(k_list)
    # kerns = GPy.util.multioutput.ICM(input_dim=input_dim, num_outputs=output_dim, kernel=kerns_mult, W_rank=3)
    # print(X_in[:5])
    # print(Y_out[:5])
    m = GPy.models.GPCoregionalizedRegression([X_in for _ in all_Y],all_Y,kernel=kerns)
    m['.*lengthscale'].constrain_bounded(lower=0.0, upper=0.2)
    print(f"Default gradients: {m.gradient}")
    # m.optimize_restarts(num_restarts=5, verbose=True)
    #m.optimize(messages=True)
    # print(f"Optimised gradients: {m.gradient}")
    #print(m)
    # for part in m.kern.parts:
    #     sigs = part.get_most_significant_input_dimensions()
    #     if None not in sigs:
    #         print(f'{part.name}: 3 most significant input dims (descending): {feature_names[sigs[0]-1]}, {feature_names[sigs[1]-1]}, {feature_names[sigs[2]-1]}')

    # Visualise model
    plot_outputs(m, output_names, feature_names, (-3,3))

    # Analyse model (sobol)
    sobol_analysis(m, features_columns, feature_names, output_columns, output_names)

    # Evaluate model (CV)
    evaluate_model(m, output_names)

def evaluate_model(model : GPy.Model, output_names, k_folds = 7, n_repeats = 3, leave_p_out = 7):
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

        k_list = []
        # k_list.append(GPy.kern.White(input_dim=input_dim))
        k_list.append(GPy.kern.Linear(input_dim=num_features))
        k_list.append(GPy.kern.RatQuad(input_dim=num_features, ARD = True))
        kerns = GPy.util.multioutput.LCM(input_dim=num_features, num_outputs=num_outputs, kernels_list=k_list, W_rank=3)
        model = GPy.models.GPCoregionalizedRegression([x_train for _ in fold_Y],fold_Y,kernel=kerns)

        # Having to rebuild model above is not ideal. It seems set_XY is not properly implemented for coregionalized models where X and Y are lists.

        print(f"Fold {fold} -- Training data....")
        model.optimize()

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

        y_preds, y_var_diag = model.predict(x_test_formatted, Y_metadata = noise_dict)

        overallR2 = r2_score(y_test_formatted, y_preds)
        print(f'Fold {fold} -- Overall R^2: {overallR2}')
        assert len(y_preds)//num_test_samples == num_outputs
        for i in range(num_outputs):
            field = output_names[i]
            field_y_test = y_test[:,i]
            field_y_preds = y_preds[int(i*num_test_samples):int((i+1)*num_test_samples)]
            field_R2 = r2_score(field_y_test, field_y_preds)
            print(f'Fold {fold} -- {field} R^2: {field_R2}')

        fold_R2s.append(overallR2)

    print(f"Mean R^2 across {k_folds} folds and {n_repeats} repeats: {np.mean(fold_R2s)}+-{np.std(fold_R2s)/np.sqrt(len(fold_R2s))}")


def sobol_analysis(model : GPy.Model, features : np.ndarray, features_names : list, outputs : np.ndarray, output_names : list, noTitle : bool = False):

    num_inputs = len(features_names)
    num_outputs = len(output_names)

    # SALib SOBOL indices
    sp = ProblemSpec({
        'num_vars': num_inputs,
        'names': list(features_names),
        'bounds': [[np.min(column), np.max(column)] for column in features.T]
    })
    test_values = salsamp.sobol.sample(sp, int(2**14), calc_second_order = (num_inputs > 1))

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
def create_spectral_features(directory : Path, field : str, spectrum_data : dict, features : list = None, peaks_only : bool = True):

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
                # savePath=save_path,
                xLabel=field))
        ml_utils.write_spectral_features_to_csv(write_path, featureSets)

    singleValueFields = [f.name for f in fields(ml_utils.SpectralFeatures1D) if f.type is float or f.type is int]
    if features is None or "all" in features:
        features = singleValueFields
    features_to_extract = np.intersect1d(features, singleValueFields)
    extracted_features = {f : [] for f in features_to_extract}
    extracted_indices = set()
    if peaks_only:
        for simIndex in range(len(featureSets)):
            if featureSets[simIndex].peaksFound:
                for feature in features_to_extract:
                    extracted_features[feature].append(featureSets[simIndex].__dict__[feature])
                    extracted_indices.add(simIndex)
    else:
        for simIndex in range(len(featureSets)):
            for feature in features_to_extract:
                extracted_features[feature].append(featureSets[simIndex].__dict__[feature])
                extracted_indices.add(simIndex)
    
    return extracted_features, list(extracted_indices)

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
        "--inputField",
        action="store",
        help="Spectral field to use for GP input.",
        required = True,
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
        "--outputFields",
        action="store",
        help="Fields to use for GP output.",
        required = False,
        type=str,
        nargs="*"
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

    regress(args.dir, args.inputField, args.outputFields, args.spectralFeatures, args.normalise, args.peaksOnly)
    # demo()