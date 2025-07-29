import argparse
import glob
import os
import GPy
import ml_utils
import pylab as pb
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from dataclasses import fields

def plot_4outputs_training_data(xTrain, xName, yOut_1, yOut_2, yOut_3, yOut_4, name_1, name_2, name_3, name_4):
    fig = plt.figure(figsize=(12,12))
    plt.suptitle(f'Input: {xName}')
    # Output 1
    ax1 = fig.add_subplot(411)
    #ax1.set_xlim(xlim)
    ax1.set_title(f'Output 1: {name_1}')
    ax1.plot(xTrain,yOut_1,'kx',mew=1.5)
    # Output 2
    ax2 = fig.add_subplot(412)
    #ax2.set_xlim(xlim)
    ax2.set_title(f'Output 2: {name_2}')
    ax2.plot(xTrain,yOut_2,'kx',mew=1.5)
    # Output 3
    ax3 = fig.add_subplot(413)
    ax3.set_title(f'Output 3: {name_3}')
    ax3.plot(xTrain,yOut_3,'kx',mew=1.5)
    # Output 4
    ax4 = fig.add_subplot(414)
    ax4.set_title(f'Output 4: {name_4}')
    ax4.plot(xTrain,yOut_4,'kx',mew=1.5)
    plt.show()

def demo_plot_2outputs(m, Xt1, Yt1, Xt2, Yt2, xlim, ylim):
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
    
    features = create_spectral_features(directory, inputFieldName, inputs, spectralFeatures, peaksOnly)
    feature_names = np.array(list(features.keys()))
    feature_values = np.array(list(features.values()))
    # feature_values_columns = feature_values.T

    output_names = list(outputs.keys())
    # output_values = np.array(list(outputs.values()))
    # output_values_columns = output_values.T

    logFields = [
        "maxPeakPower", 
        # "meanPeak4Powers", 
        # "meanPeak3Powers", 
        # "meanPeak2Powers",
        # "varPeak4Powers", 
        # "varPeak3Powers", 
        # "varPeak2Powers",
        "maxPeakPowerProminence",
        # "activeRegionVarPeakSeparations",
        # "activeRegionMeanCoordWidths",
        # "activeRegionVarCoordWidths",
        "totalActiveProportion",
        "spectrumVar",
        "spectrumSum",
        "spectrumMean"
    ]
    for f_name in logFields:
        if f_name in features:
            features[f_name] = np.log(features[f_name])

    # Normalise
    features_array = np.array(list(features.values()))
    if normalise:
        features_array = np.array([(f - np.nanmean(f)) / np.nanstd(f) for f in features_array])
    features_columns = features_array.T
    
    # Normalise this as well
    output_array = np.array(list(outputs.values()))
    if normalise:
        output_array = np.array([(p - np.nanmean(p)) / np.nanstd(p) for p in output_array])
    output_columns = output_array.T

    # Plot training data
    # output = output_columns
    # input = features_columns
    # for f in range(len(feature_names)):
    #     y1 = output[:,0]
    #     y2 = output[:,1]
    #     y3 = np.log10(output[:,2])
    #     y4 = np.log10(output[:,3])
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
    k_list.append(GPy.kern.Matern32(input_dim=input_dim, ARD = True))
    lcm = GPy.util.multioutput.LCM(input_dim=input_dim, num_outputs=output_dim, kernels_list=k_list)
    m = GPy.models.GPCoregionalizedRegression([X_in for _ in all_Y],all_Y,kernel=lcm)
    print(f"Default gradients: {m.gradient}")
    # m.optimize_restarts(verbose=True)
    m.optimize(messages=True)
    print(f"Optimised gradients: {m.gradient}")
    print(m)
    for part in m.kern.parts:
        sigs = part.get_most_significant_input_dimensions()
        if None not in sigs:
            print(f'{part.name}: 3 most significant input dims (descending): {feature_names[sigs[0]-1]}, {feature_names[sigs[1]-1]}, {feature_names[sigs[2]-1]}')

# Single-valued features only for now
def create_spectral_features(directory : Path, field : str, spectrum_data : dict, features : list = None, peaks_only : bool = True):
    # if bigLabels:
    #     plt.rcParams.update({'axes.titlesize': 26.0})
    #     plt.rcParams.update({'axes.labelsize': 24.0})
    #     plt.rcParams.update({'xtick.labelsize': 20.0})
    #     plt.rcParams.update({'ytick.labelsize': 20.0})
    #     plt.rcParams.update({'legend.fontsize': 18.0})
    # else:
    #     plt.rcParams.update({'axes.titlesize': 18.0})
    #     plt.rcParams.update({'axes.labelsize': 16.0})
    #     plt.rcParams.update({'xtick.labelsize': 14.0})
    #     plt.rcParams.update({'ytick.labelsize': 14.0})
    #     plt.rcParams.update({'legend.fontsize': 14.0})

    # example_model = GPy.examples.regression.sparse_GP_regression_1D(plot=True, optimize=False)
    # plt.show()
    # print(example_model)

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
    extracted_features = {}

    if peaks_only:
        if features is None or "all" in features:
            for feature in singleValueFields:
                extracted_features[feature] = [simData.__dict__[feature] for simData in featureSets if simData.peaksFound]
        else:
            for feature in features:
                if feature in singleValueFields:
                    extracted_features[feature] = [simData.__dict__[feature] for simData in featureSets if simData.peaksFound]
    else:
        if features is None or "all" in features:
            for feature in singleValueFields:
                extracted_features[feature] = [simData.__dict__[feature] for simData in featureSets]
        else:
            for feature in features:
                if feature in singleValueFields:
                    extracted_features[feature] = [simData.__dict__[feature] for simData in featureSets]
    
    return extracted_features

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