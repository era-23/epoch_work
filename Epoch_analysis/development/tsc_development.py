# import argparse
# import glob

# from matplotlib import pyplot as plt
# from sklearn.dummy import DummyRegressor
# from sklearn.metrics import mean_squared_error
# from sklearn.pipeline import make_pipeline
# import ml_utils
# import pandas as pd
# import numpy as np
# import xarray as xr
# import shutil
# import os
# import dataclasses
# from pathlib import Path
# from inference.plotting import matrix_plot

# from sktime.classification.interval_based import TimeSeriesForestClassifier
# from sktime.datasets import load_arrow_head
# from sktime.datatypes import check_raise
# from sktime.dists_kernels import FlatDist, ScipyDist
# from sktime.clustering.spatio_temporal import STDBSCAN
# from sktime.clustering.dbscan import TimeSeriesDBSCAN
# from sktime.clustering.k_means import TimeSeriesKMeans
# from sklearn.preprocessing import StandardScaler
# from sktime.pipeline import Pipeline
# from sktime.benchmarking.data import RAMDataset
# from sktime.benchmarking.tasks import TSRTask
# from sklearn.model_selection import KFold, cross_val_score, train_test_split
# from sktime.datasets import load_unit_test
# from sktime.split import TemporalTrainTestSplitter
# from sktime.transformations.compose import FitInTransform
# from sktime.regression.compose import RegressorPipeline

# import epoch_utils as e_utils

# def load_data_hierarchical(directory : Path, input_fields : list, input_xAxis_field : str, output_fields : list):
    
#     # Input data
#     input_dict = {f : [] for f in input_fields}

#     # Output data
#     output_dict = {f : [] for f in output_fields}

#     get_xAxis_points = True
#     sim_numbers = []
#     xAxis_points = []
#     metadata = {}

#     data_files = glob.glob(str(directory / "*.nc"))

#     for simulation in data_files:

#         data = xr.open_datatree(
#             simulation,
#             engine="netcdf4"
#         )
#         sim_id = int(Path(simulation).name.split("_")[1])
#         sim_numbers.append(sim_id)

#         metadata[sim_id] = ml_utils.SimulationMetadata(
#             simId=sim_id,
#             backgroundDensity=float(data.attrs["backgroundDensity"]),
#             beamFraction=float(data.attrs["beamFraction"]),
#             B0=float(data.attrs["B0strength"]),
#             B0angle=float(data.attrs["B0angle"])
#         )

#         for fieldPath in input_dict.keys():
#             s = fieldPath.split("/")
#             group = "/" if len(s) == 1 else '/'.join(s[:-1])
#             fieldName = s[-1]
#             input_dict[fieldPath].append(data[group].variables[fieldName].values)

#         if get_xAxis_points:
#             xAxis_points = data[input_xAxis_field]
#             get_xAxis_points = False

#         for fieldPath in output_dict.keys():
#             s = fieldPath.split("/")
#             group = "/" if len(s) == 1 else '/'.join(s[:-1])
#             fieldName = s[-1]
#             if fieldName in data[group].variables:
#                 output_dict[fieldPath].append(data[group].variables[fieldName].values)
#             elif fieldName in data[group].attrs:
#                 output_dict[fieldPath].append(data[group].attrs[fieldName])

#     input_dict_flat= {k : np.ravel(np.array(v)) for k, v in input_dict.items()}
#     headings_mi = pd.MultiIndex.from_product([sim_numbers, np.arange(len(xAxis_points))], names=["sim_ID", input_xAxis_field.split("/")[-1]])
#     input_df = pd.DataFrame(np.array(list(input_dict_flat.values())).T, index = headings_mi, columns=list(input_dict_flat.keys()))
#     if output_dict:
#         if len(output_dict.keys()) == 1:
#             output_df = np.array(list(output_dict.values())[0])
#         else:
#             output_df = pd.DataFrame(np.array(list(output_dict.values())).T, index = headings_mi, columns=list(input_dict_flat.keys()))

#     return input_df, output_df, metadata

# # Returns input data in a numpy array of shape (n_samples, n_time_points) and output data as a 1D numpy array
# def load_data_sklearn(directory : Path, input_field : str, output_field : str):

#     # Data
#     input_list = []
#     output_list = []

#     data_files = glob.glob(str(directory / "*.nc"))

#     for simulation in data_files:

#         data = xr.open_datatree(
#             simulation,
#             engine="netcdf4"
#         )

#         input_list.append(data[input_field].to_numpy())
#         output_list.append(data.attrs[output_field])

#     return np.array(input_list), np.array(output_list)
   
# def clustering(dataDirectory : Path, workingDir : Path, numClusters : int, algorithms : list, input_fields : list, output_fields : list):
    
#     # Clear folder
#     for f in os.listdir(workingDir):
#         fullPath = os.path.join(workingDir, f)
#         if Path(fullPath).is_dir():
#             shutil.rmtree(fullPath)
#         else:
#             os.remove(fullPath)

#     data_path = dataDirectory / "data"
#     plots_path = dataDirectory / "plots" / "energy"

#     input_data, _, _, metadata = load_data_hierarchical(data_path, input_fields, output_fields)
#     scaler = StandardScaler()

#     ids = list(dict.fromkeys([i[0] for i in input_data.index.to_list()]))
#     param_labels = [p.name for p in dataclasses.fields(metadata[0]) if p.name != "simId"]
    
#     for algo in algorithms:
#         print(f"Clustering using {algo}....")

#         # Get algorithm
#         clusterer = ml_utils.get_algorithm(name = algo, n_clusters = numClusters)
#         pipeline = scaler * clusterer
        
#         # Fit
#         pipeline.fit(input_data)
#         print("\n".join(f"{k}: {v}" for k, v in pipeline.get_fitted_params().items()))

#         # Create folder
#         algorithm_path = workingDir / algo
#         if os.path.exists(algorithm_path):
#             shutil.rmtree(algorithm_path)
#         os.mkdir(algorithm_path)
        
#         cluster_sims = {c : [] for c in pipeline.clusterer_.labels_}
#         for i in range(len(pipeline.clusterer_.labels_)):
#             sim_id = ids[i]
#             cluster_id = pipeline.clusterer_.labels_[i]
#             cluster_sims[cluster_id].append(sim_id)

#         cluster_sims_ordered = dict(sorted(cluster_sims.items()))
#         # seriesLabs = []
#         c_metadata = []
#         for c in cluster_sims_ordered.keys():
            
#             cluster_path = algorithm_path / f"cluster_{c}/"
#             os.mkdir(cluster_path)
            
#             densities = []
#             beamFracs = []
#             b0s = []
#             bAngles = []
#             for s in cluster_sims_ordered[c]:
#                 print(f"cluster {c}: simulation {s}")
#                 print(f"Copying sim {s} energy plots to cluster {c} folder....")
#                 shutil.copy(plots_path / f"run_{s}_percentage_energy_change.png", cluster_path)

#                 # if s == 32:
#                 #     seriesLabs.append("Complete MCI evolution")
#                 # elif s == 63:
#                 #     seriesLabs.append("Noise dominated")
#                 # elif s == 29:
#                 #     seriesLabs.append("Partial MCI evolution")


#                 densities.append(np.log10(metadata[s].backgroundDensity))
#                 beamFracs.append(np.log10(metadata[s].beamFraction))
#                 b0s.append(metadata[s].B0)
#                 bAngles.append(metadata[s].B0angle)
            
#             c_metadata.append([densities, beamFracs, b0s, bAngles])
        
#         # Matrix plot
#         try:
#             e_utils.my_matrix_plot(
#                 data_series=c_metadata, 
#                 series_labels=list(cluster_sims_ordered.keys()), 
#                 # series_labels=seriesLabs, 
#                 parameter_labels=param_labels, 
#                 plot_style="hdi", 
#                 colormap_list=["Reds", "Greens", "Blues", "Purples", "Greys", "Oranges"], 
#                 show=False,
#                 equalise_pdf_heights=False
#             )       
#         except ValueError:
#             print("Error in matrix plot, skipping")
#             continue
#         plt.savefig(algorithm_path / "matrix.png")
#         plt.close("all")

# def regression(dataDirectory : Path, workingDir : Path, algorithms : list, input_fields : list, input_xAxis_field : str, output_field : str):

#     data_path = dataDirectory / "data"
#     scaler = StandardScaler()

#     for input_field in input_fields:

#         inputs, outputs = load_data_sklearn(data_path, input_field, output_field)

#         X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.25)

#         for algo in algorithms:
#             print(f"Clustering using {algo}....")

#             # Get algorithm
#             regressor = ml_utils.get_algorithm(name = algo, n_neighbours = 3, distance='euclidean')
#             pipeline = make_pipeline(scaler, regressor)
#             dummyPipeline = make_pipeline(scaler, DummyRegressor())
            
#             # Fit
#             pipeline.fit(X_train, y_train)
#             dummyPipeline.fit(X_train, y_train)
#             # print("\n".join(f"{k}: {v}" for k, v in pipeline.get_fitted_params().items()))
#             y_pred = pipeline.predict(X_test)
#             dummy_pred = dummyPipeline.predict(X_test)

#             # Evaluate
#             y_test = scaler.fit_transform(y_test.reshape(-1, 1))
#             rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#             rmse_dummy = np.sqrt(mean_squared_error(y_test, dummy_pred))
#             print(f"Model Root Mean Squared Error: {rmse:.3f}")
#             print(f"Dummy Root Mean Squared Error: {rmse_dummy:.3f}")

# if __name__ == "__main__":
    
#     parser = argparse.ArgumentParser("parser")
#     parser.add_argument(
#         "--dataDir",
#         action="store",
#         help="Directory containing netCDF files of simulation output.",
#         required = True,
#         type=Path
#     )
#     parser.add_argument(
#         "--workingDir",
#         action="store",
#         help="Directory in which to store TS files such as clustering results.",
#         required = False,
#         type=Path
#     )
#     parser.add_argument(
#         "--cluster",
#         action="store_true",
#         help="Run TS clustering.",
#         required = False
#     )
#     parser.add_argument(
#         "--regress",
#         action="store_true",
#         help="Run TS regression.",
#         required = False
#     )
#     parser.add_argument(
#         "--numClusters",
#         action="store",
#         help="Number of clusters",
#         required=False,
#         type=int
#     )
#     parser.add_argument(
#         "--algorithms",
#         action="store",
#         help="Algorithm names (from sktime) to use.",
#         required = False,
#         type=str,
#         nargs="*"
#     )
#     parser.add_argument(
#         "--inputFields",
#         action="store",
#         help="Fields to use for TS input.",
#         required = False,
#         type=str,
#         nargs="*"
#     )
#     parser.add_argument(
#         "--inputXAxisField",
#         action="store",
#         help="Fields to use for input x-axis.",
#         required = False,
#         type=str
#     )
#     parser.add_argument(
#         "--outputField",
#         action="store",
#         help="Fields to use for TS output.",
#         required = False,
#         type=str
#     )

#     args = parser.parse_args()

#     if args.cluster:
#         clustering(args.dataDir, args.workingDir, args.numClusters, args.algorithms, args.inputFields, [])
#     if args.regress:
#         regression(args.dataDir, args.workingDir, args.algorithms, args.inputFields, args.inputXAxisField, args.outputField)