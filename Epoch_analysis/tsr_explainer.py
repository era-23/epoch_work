import copy
import csv
import glob
import os
import argparse
from pathlib import Path

from tsCaptum.explainers import Feature_Ablation
from tsCaptum.visualization import plot_saliency_map_uni

# 0 = All logs (default)
# 1 = Filter out INFO logs
# 2 = Filter out INFO and WARNING logs
# 3 = Filter out INFO, WARNING, and ERROR logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Optional: Silence the oneDNN message explicitly
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from matplotlib import ticker
from scipy.stats import sem
from sklearn.preprocessing import MinMaxScaler

import epoch_utils
import ml_utils

def regress_epoch_vs_cottrell(
        dataDirectory : Path,
        inputSpectraNames : list,
        outputFields : list,
        logFields : list,
        algorithms : list,
        cottrellDatapath : Path,
        resultsFilepath : Path = None,
        includeFreqs : bool = False,
        nThreads : int = 1,
        lowFrequencyCleaningMethod : str = "linterpToSP",
        displayPlots : bool = False
):

    if displayPlots:
        SMALL_SIZE = 10
        MEDIUM_SIZE = 16
        BIGGER_SIZE = 20

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=12)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Initialise results objects

    if not ("data" in dataDirectory.name):
        data_dir = dataDirectory / "data"
    else:
        data_dir = dataDirectory
    data_files = glob.glob(str(data_dir / "*.nc")) 

    # Input data
    inputs = {name : [] for name in inputSpectraNames}
    inputs = ml_utils.read_data(data_files, inputs, with_names = True, with_coords = True, with_iciness = False, denorm_coords = True)

    # Output data
    outputs = {outputField : [] for outputField in outputFields}
    outputs = ml_utils.read_data(data_files, outputs, with_names = True, with_coords = False)

    if "B0angle" in outputs:
        transf = np.array(outputs["B0angle"])
        outputs["B0angle"] = np.abs(transf - 90.0) 

    spec_lengths = []
    for field in inputSpectraNames:
        spec_lengths.extend([len(s) for s in inputs[field]])
    min_l = np.min(spec_lengths)
    print(f"Max spec length: {np.max(spec_lengths)} min spec length: {min_l}")

    # Get Cottrell data
    cottrell_data = pd.read_csv(cottrellDatapath)
    max_frequency = float((cottrell_data["Frequency (MHz)"].max() * u.MHz).to(u.Hz).value)

    inputData = []
    sim_id_to_indexNumber = {}
    for field in inputSpectraNames:
        specs = copy.deepcopy(inputs[field])
        coords = np.zeros_like(specs)
        
        for i in range(len(specs)):

            # Record sim ID and index
            sim_id_to_indexNumber[inputs["sim_ids"][i]] = i

            gyro_coords = inputs[f"{field}_coords"][i]
            hz_coords = inputs[f"{field}_denorm_coords"][i]

            # If truncation needed
            if hz_coords[-1] > max_frequency:
                truncd_series, truncd_coords, truncd_gyro_coords = ml_utils.truncate_series(specs[i], hz_coords, max_frequency, altCoordinates = gyro_coords)
                resamp_series, _ = ml_utils.resample_series(truncd_series, truncd_coords, min_l + 1)
                # specs[i] = np.log10(resamp_series, out=np.zeros_like(resamp_series), where = (resamp_series!=0.0))[1:]
                specs[i] = resamp_series[1:]
                coords[i] = np.linspace(truncd_gyro_coords[0], truncd_gyro_coords[-1], coords.shape[1])

            # If the first point is a minimum
            spec = specs[i]
            if "SP" in lowFrequencyCleaningMethod:
                if np.argmin(specs[i]) == 0: 
                    stationary_point = int(next(i for i,(v0,v1) in enumerate(zip(spec[:-1], spec[1:])) if v0>v1))
                elif np.argmax(specs[i]) == 0: # Else if the first point is a maximum
                    stationary_point = int(next(i for i,(v0,v1) in enumerate(zip(spec[:-1], spec[1:])) if v0<v1))
                else:
                    continue
                after_sp = spec[stationary_point:]
                
                if lowFrequencyCleaningMethod == "minToSP":
                    before_sp = np.ones(stationary_point) * np.min(after_sp)
                elif lowFrequencyCleaningMethod == "meanToSP":
                    before_sp = np.ones(stationary_point) * np.mean(after_sp)
                else:
                    before_sp = np.linspace(np.mean(spec), spec[stationary_point], stationary_point)
                
                specs[i] = np.concatenate((before_sp, after_sp), axis = 0)
            else: # replace early frequencies up to less than known lowest possible harmonic (3MHz)
                
                replace_idx = np.argwhere(hz_coords < 2e6)[-1][-1]
                after_sp = spec[replace_idx:]
                if lowFrequencyCleaningMethod == "padMin":
                    before_sp = np.ones(replace_idx) * np.min(after_sp)
                elif lowFrequencyCleaningMethod == "padMean":
                    before_sp = np.ones(replace_idx) * np.mean(after_sp)

                specs[i] = np.concatenate((before_sp, after_sp), axis = 0)
        
        global_min = np.min([np.min(s) for s in specs])
        print(f"Global minimum : {global_min}")
        specs = 10.0 * np.log10(specs / global_min)
        
        inputData.append(specs)
    if includeFreqs:
        inputData.append(coords) # Append only the last set of coordinates (they should be the same for all fields)

    assert inputs["sim_ids"] == outputs["sim_ids"]
    assert inputs["sim_ids"] == list(sim_id_to_indexNumber.keys())

    # Reshape into 3D numpy array of shape (n_cases, n_channels, n_timepoints)
    inputData = np.array(inputData)
    print(f"Spectra in dB range from {np.min(inputData[0])}dB to {np.max(inputData[0])}dB.")
    inputSpectra = np.swapaxes(inputData, 0, 1)
    # outputs = np.array(list(outputs.values()))

    logFields = np.intersect1d(outputFields, logFields)

    # Sort and resample Cottrell
    sort_indices = np.argsort(cottrell_data["Frequency (MHz)"])
    equally_spaced_freqs = np.linspace(cottrell_data["Frequency (MHz)"].min(), cottrell_data["Frequency (MHz)"].max(), inputSpectra.shape[2])
    test_x = [np.interp(equally_spaced_freqs, np.sort(cottrell_data["Frequency (MHz)"]), cottrell_data["ICE Intensity (dB)"][sort_indices])]
    if includeFreqs:
        cottrell_gyro_freqs = np.linspace(cottrell_data["Frequency (MHz)"].min() / 17.0, cottrell_data["Frequency (MHz)"].max() / 17.0, inputSpectra.shape[2])
        test_x.append(cottrell_gyro_freqs)
    
    # True values (edge JET values )
    all_true_y = {"B0strength" : 2.21, "backgroundDensity" : 1.7E19, "beamFraction" : 1.5E-4, "pitch" : 0.4}

    train_x = inputSpectra

    test_x = np.array([np.swapaxes(np.array(test_x), 0, 1).T])
    # test_x[0][0] = scaler_train.fit_transform(test_x[0][0].reshape(-1, 1)).T # Only scale spectra, not frequencies

    for output_field, output_values in outputs.items():

        if output_field not in outputFields:
            continue

        # ##### Debugging
        # pct_diffs = np.array([np.abs(v - all_true_y[output_field]) for v in output_values])
        # print(f"Closest indices to {output_field} {all_true_y[output_field]}: {pct_diffs.argsort()[:20]}")
        # continue
        # ##### /Debugging

        assert len(output_values) == inputSpectra.shape[0]
        output_values = np.array(output_values)
        true_y = all_true_y[output_field]
        
        if output_field in logFields:
            output_values = np.log10(output_values)
            true_y = np.log10(true_y)
        
        train_y, scaler_y = ml_utils.normalise_data(output_values)
        true_y_norm, _ = ml_utils.normalise_data([true_y], scaler_y)
        print(f"True output value: {all_true_y[output_field]} ({true_y_norm} normalised)")

        # Record denormalisation parameters
        # print(f"Original data mean: {np.mean(output_values)}, original data SD: {np.std(output_values)}")

        for algorithm in algorithms:

            # print(f"Building {algorithm} model for {output_field} from {inputSpectraNames}....")
            
            # Results
            result = ml_utils.TSRResult()
            result.output = output_field
            result.algorithm = algorithm
            
            tsr = ml_utils.get_algorithm(algorithm, nThreads)

            # print(f"Testing algorithm {algorithm} on Cottrell data....")
            # print("Training model....")
            
            # Fit
            tsr.fit(train_x, train_y)

            # Predict
            # print("Predicting Cottrell parameters based on model....")
            prediction = tsr.predict(test_x)

            myFA = Feature_Ablation(tsr)
            exp = myFA.explain(samples=test_x, n_segments=10, normalise=False)
            print( "saliency map shape equal to input shape:", exp.shape, test_x[0].shape,
                "\n attributions for first 20 time points:\n", exp[0,:,:20])

            idx = 0
            plot_saliency_map_uni(test_x[idx,:,:], exp[idx,:,:], title = f'FeatureAblation-Instance {idx} - Target {true_y_norm[idx]}')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--dataDir",
        action="store",
        help="Directory containing netCDF files of simulation output.",
        required = True,
        type=Path
    )
    parser.add_argument(
        "--resultsFilepath",
        action="store",
        help="Filepath of csv to which to write results.",
        required = False,
        type=Path
    )
    parser.add_argument(
        "--algorithms",
        action="store",
        help="Algorithms to run.",
        required = False,
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--inputSpectra",
        action="store",
        help="Spectra to use for TSR input.",
        required = False,
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--cottrell",
        action="store_true",
        help="Run Cottrell experiment.",
        required = False
    )
    parser.add_argument(
        "--cottrellFilepath",
        action="store",
        help="Filepath of Cottrell 93 data to regress against.",
        required = False,
        type=Path
    )
    parser.add_argument(
        "--nThreads",
        action="store",
        help="Number of threads to use for training and prediction.",
        required = False,
        type=int,
        default=1
    )
    parser.add_argument(
        "--displayPlots",
        action="store_true",
        help="Generate and display plots (primarily for Cottrell regression).",
        required = False
    )

    args = parser.parse_args()

    if args.cottrell:
        regress_epoch_vs_cottrell(
            dataDirectory = args.dataDir, 
            inputSpectraNames = [
                "Magnetic_Field_Bz/power/frequencyPowerSpectrum",
            ], 
            outputFields = [
                "B0strength", 
                "pitch", 
                "backgroundDensity", 
                "beamFraction"
            ], 
            logFields = [
                "backgroundDensity", 
                "beamFraction"
            ], 
            algorithms = 
                args.algorithms,
            cottrellDatapath = args.cottrellFilepath,
            resultsFilepath=args.resultsFilepath,
            nThreads = args.nThreads,
            displayPlots=args.displayPlots,
        )