# General imports for the notebook
import argparse
import glob
from pathlib import Path
import warnings
import numpy as np
import xarray as xr
import torch
from matplotlib import pyplot as plt
import ml_utils
warnings.filterwarnings("ignore")
from autoemulate.simulations.projectile import Projectile
from autoemulate import AutoEmulate

def demo():
    projectile = Projectile(log_level="error")
    n_samples = 500
    x = projectile.sample_inputs(n_samples).float()
    y, _ = projectile.forward_batch(x)
    y = y.float()

    print(x.shape)
    print(y.shape)

    print(AutoEmulate.list_emulators())

    # Run AutoEmulate with default settings
    ae = AutoEmulate(x, y, only_probabilistic = True, log_level="progress_bar")

    print(ae.summarise())

    best = ae.best_result()
    print("Model with id: ", best.id, " performed best: ", best.model_name)

    ae.plot(best, fname="best_model_plot.png")

    ae.plot(best, input_ranges={0: (0, 4), 1: (200, 500)}, output_ranges={0: (0, 10)})

    print(best.model.predict(x[:10]))

def autoEmulate(inputDir : Path, inputFields : list, outputFile : Path):

    ##### Get input data
    data_files = glob.glob(str(inputDir / "*.nc"))
    
    # Input data
    inputs = {name : [] for name in inputFields}
    inputs = ml_utils.read_data(data_files, inputs, with_names = True, with_coords = False)

    ##### Get output data
    output_ds = xr.open_dataset(outputFile, engine="netcdf4")
    sim_ids = [int(id) for id in output_ds.coords["index"]]
    sorted_idx = np.array(sim_ids).argsort()
    outputData_list = []
    for feature in output_ds.variables:
        feature_values = output_ds.variables[feature].values.astype(float)
        if len(set(feature_values)) > 1 and not np.isnan(feature_values).any():
            outputData_list.append(np.array([float(feature_values[i]) for i in sorted_idx]))
    
    ##### Format
    inputData_arr = np.array([np.array(inputs[f]) for f in inputFields]).T
    outputData_arr = np.array(outputData_list).T
    input_ptt = torch.from_numpy(inputData_arr)
    output_ptt = torch.from_numpy(outputData_arr)

    print(input_ptt.shape)
    print(output_ptt.shape)

    # Run AutoEmulate with default settings
    ae = AutoEmulate(input_ptt, output_ptt, models = ["GaussianProcess", "GaussianProcessCorrelated", "GaussianProcessMatern32", "GaussianProcessMatern52", "GaussianProcessRBF"], log_level="progress_bar")

    print(ae.summarise())

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--inputDir",
        action="store",
        help="Directory containing netCDF files of simulation data from which to source inputs.",
        required = True,
        type=Path
    )
    parser.add_argument(
        "--inputFields",
        action="store",
        help="Fields to use as emulation inputs.",
        required = True,
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--outputFile",
        action="store",
        help="NetCDF file of output features.",
        required = True,
        type=Path
    )

    args = parser.parse_args()

    autoEmulate(args.inputDir, args.inputFields, args.outputFile)
    # demo()