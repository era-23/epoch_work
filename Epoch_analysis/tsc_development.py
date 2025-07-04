import argparse
import glob

from matplotlib import pyplot as plt
import ml_utils
import pandas as pd
import numpy as np
import xarray as xr
import shutil
import os
import dataclasses
from pathlib import Path
from inference.plotting import matrix_plot

from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.datasets import load_arrow_head
from sktime.datatypes import check_raise
from sktime.dists_kernels import FlatDist, ScipyDist
from sktime.clustering.spatio_temporal import STDBSCAN
from sktime.clustering.dbscan import TimeSeriesDBSCAN
from sktime.clustering.k_means import TimeSeriesKMeans
from sklearn.preprocessing import StandardScaler
from sktime.pipeline import Pipeline

import epoch_utils as e_utils

def load_data(directory : Path, input_fields : list, output_fields : list):
    
    # Input data
    input_dict = {f : [] for f in input_fields}

    # Output data
    output_dict = {f : [] for f in output_fields}

    get_time_points = True
    sim_numbers = []
    time_points = []
    metadata = {}

    data_files = glob.glob(str(directory / "*.nc"))

    for simulation in data_files:

        data = xr.open_datatree(
            simulation,
            engine="netcdf4"
        )
        sim_id = int(Path(simulation).name.split("_")[1])
        sim_numbers.append(sim_id)

        metadata[sim_id] = ml_utils.SimulationMetadata(
            simId=sim_id,
            backgroundDensity=float(data.attrs["backgroundDensity"]),
            beamFraction=float(data.attrs["beamFraction"]),
            B0=float(data.attrs["B0strength"]),
            B0angle=float(data.attrs["B0angle"])
        )

        for fieldPath in input_dict.keys():
            s = fieldPath.split("/")
            group = "/" if len(s) == 1 else '/'.join(s[:-1])
            fieldName = s[-1]
            if get_time_points:
                time_points = data[group].variables["time"]
                get_time_points = False
            input_dict[fieldPath].append(data[group].variables[fieldName].values)

        for fieldPath in output_dict.keys():
            s = fieldPath.split("/")
            group = "/" if len(s) == 1 else '/'.join(s[:-1])
            fieldName = s[-1]
            output_dict[fieldPath].append(data[group].variables[fieldName].values)

    input_dict = {k : np.ravel(np.array(v)) for k, v in input_dict.items()}
    headings_mi = pd.MultiIndex.from_product([sim_numbers, np.arange(len(time_points))], names=["sim_ID", "time"])
    input_df = pd.DataFrame(np.array(list(input_dict.values())).T, index = headings_mi, columns=list(input_dict.keys()))
    output_df = pd.DataFrame(np.array(list(output_dict.values())).T, index = headings_mi, columns=list(input_dict.keys())) if output_dict else None

    return input_df, output_df, time_points, metadata
   
def clustering(dataDirectory : Path, workingDir : Path, numClusters : int, algorithms : list, input_fields : list, output_fields : list):
    
    # Clear folder
    for f in os.listdir(workingDir):
        fullPath = os.path.join(workingDir, f)
        if Path(fullPath).is_dir():
            shutil.rmtree(fullPath)
        else:
            os.remove(fullPath)

    data_path = dataDirectory / "data"
    plots_path = dataDirectory / "plots" / "energy"

    input_data, _, _, metadata = load_data(data_path, input_fields, output_fields)
    scaler = StandardScaler()

    ids = list(dict.fromkeys([i[0] for i in input_data.index.to_list()]))
    param_labels = [p.name for p in dataclasses.fields(metadata[0]) if p.name != "simId"]
    
    for algo in algorithms:
        print(f"Clustering using {algo}....")

        # Get algorithm
        clusterer = ml_utils.get_algorithm(name = algo, n_clusters = numClusters)
        pipeline = scaler * clusterer
        
        # Fit
        pipeline.fit(input_data)
        print("\n".join(f"{k}: {v}" for k, v in pipeline.get_fitted_params().items()))

        # Create folder
        algorithm_path = workingDir / algo
        if os.path.exists(algorithm_path):
            shutil.rmtree(algorithm_path)
        os.mkdir(algorithm_path)
        
        cluster_sims = {c : [] for c in pipeline.clusterer_.labels_}
        for i in range(len(pipeline.clusterer_.labels_)):
            sim_id = ids[i]
            cluster_id = pipeline.clusterer_.labels_[i]
            cluster_sims[cluster_id].append(sim_id)

        cluster_sims_ordered = dict(sorted(cluster_sims.items()))
        # cluster_sims_ordered = cluster_sims
        c_metadata = []
        for c in cluster_sims_ordered.keys():
            
            cluster_path = algorithm_path / f"cluster_{c}/"
            os.mkdir(cluster_path)
            
            densities = []
            beamFracs = []
            b0s = []
            bAngles = []
            for s in cluster_sims_ordered[c]:
                print(f"cluster {c}: simulation {s}")
                print(f"Copying sim {s} energy plots to cluster {c} folder....")
                shutil.copy(plots_path / f"run_{s}_percentage_energy_change.png", cluster_path)

                densities.append(np.log10(metadata[s].backgroundDensity))
                beamFracs.append(np.log10(metadata[s].beamFraction))
                b0s.append(metadata[s].B0)
                bAngles.append(metadata[s].B0angle)
            
            c_metadata.append([densities, beamFracs, b0s, bAngles])
        
        # Matrix plot
        try:
            e_utils.my_matrix_plot(
                data_series=c_metadata, 
                series_labels=list(cluster_sims_ordered.keys()), 
                parameter_labels=param_labels, 
                plot_style="hdi", 
                colormap_list=["Reds", "Greens", "Blues", "Purples", "Greys", "Oranges"], 
                show=False,
                equalise_pdf_heights=False
            )       
        except ValueError:
            print("Error in matrix plot, skipping")
            continue
        plt.savefig(algorithm_path / "matrix.png")
        plt.close("all")

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
        "--workingDir",
        action="store",
        help="Directory in which to store TS files such as clustering results.",
        required = False,
        type=Path
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Run TS clustering.",
        required = False
    )
    parser.add_argument(
        "--numClusters",
        action="store",
        help="Number of clusters",
        required=False,
        type=int
    )
    parser.add_argument(
        "--algorithms",
        action="store",
        help="Algorithm names (from sktime) to use.",
        required = False,
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--inputFields",
        action="store",
        help="Fields to use for TS input.",
        required = False,
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--outputFields",
        action="store",
        help="Fields to use for TS output.",
        required = False,
        type=str,
        nargs="*"
    )

    args = parser.parse_args()

    if args.cluster:
        clustering(args.dataDir, args.workingDir, args.numClusters, args.algorithms, args.inputFields, [])