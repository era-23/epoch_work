from pathlib import Path
import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt
from plasmapy.formulary import frequencies as ppf
from plasmapy.formulary import speeds as pps
from plasmapy.formulary import lengths as ppl
from plasmapy import particles as ppp
from scipy import constants as constants
from scipy.stats import linregress
import astropy.units as u
import csv
import epoch_utils
import copy
import pandas as pd
import epydeck
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, RationalQuadratic
import itertools

def analyse_bad_predictions_2(csvPredictions : Path):
    
    plt.rcParams.update({'axes.titlesize': 26.0})
    plt.rcParams.update({'axes.labelsize': 20.0})
    plt.rcParams.update({'xtick.labelsize': 20.0})
    plt.rcParams.update({'ytick.labelsize': 20.0})
    plt.rcParams.update({'legend.fontsize': 18.0})

    data_pd = pd.read_csv(csvPredictions)
    fields = ["B0strength", "pitch", "backgroundDensity", "beamFraction"]
    logFields = ["backgroundDensity", "beamFraction"]
    fieldCombos = list(itertools.combinations(fields, 2))

    for f1, f2 in fieldCombos:
        
        mean_e = np.mean([data_pd[f"error_rank_{f1}"], data_pd[f"error_rank_{f2}"]], axis=0)
        data_pd.plot.scatter(x = f1, y = f2, c = mean_e, s=50, cmap = "bwr", figsize=(8,6))
        if f1 in logFields:
            plt.xscale("log")
        if f2 in logFields:
            plt.yscale("log")
        fig = plt.gcf()
        cax = fig.get_axes()[1]
        cax.set_ylabel('Mean error rank')
        plt.tight_layout()
        plt.savefig(Path("/home/era536/Documents/for_discussion/2026.02.12/") / f"errors_{f1}_{f2}.png")
    
    for f in fields:
        data_pd.plot.scatter(x = "vPerpToAlfvenRatio", y = f, c = f"error_rank_{f}", s=50, cmap = "bwr", figsize=(8,6))
        if f in logFields:
            plt.yscale("log")
        fig = plt.gcf()
        cax = fig.get_axes()[1]
        cax.set_ylabel(f'{f} error rank')
        plt.tight_layout()
        plt.savefig(Path("/home/era536/Documents/for_discussion/2026.02.12/") / f"errors_vPerpAlfvenRatio_{f}.png")


def analyse_bad_predictions(csvPredictions : Path, inputDecks : Path):
    
    newObs = []
    with open(csvPredictions, "r") as csvFile:
        csvr = csv.DictReader(csvFile)
        for row in csvr:
            newPredictionObject = copy.deepcopy(row)
            newPredictionObject["inputChannels"] = np.array(row["inputChannels"].removeprefix("[").removesuffix("]").replace("'", "").replace("\n", "").split(", "))
            for k, v in row.items():
                try:
                    newPredictionObject[k] = float(v)
                except Exception:
                    pass
            newObs.append(newPredictionObject)
        
    preds_pd = pd.DataFrame(newObs)
    preds_pd["error"] = preds_pd["predictedValue_normalised"] - preds_pd["trueValue_normalised"]
    preds_pd["squaredError"] = preds_pd["error"]**2

    # Select MRH only
    mrh_preds_pd : pd.DataFrame = preds_pd[preds_pd["algorithm"] == "aeon.MultiRocketHydraRegressor"]

    # For each output
    squaredRanks = []
    for outputField in list(dict.fromkeys(mrh_preds_pd["outputQuantity"].values).keys()): # Dict keys preserve order
        fieldPredictions : pd.DataFrame = mrh_preds_pd[mrh_preds_pd["outputQuantity"] == outputField]
        ranks = fieldPredictions["squaredError"].rank().tolist()
        squaredRanks.extend(ranks)

    mrh_preds_pd["squaredError_rank"] = squaredRanks

    columns = [
        "sim_ID", 
        "alfvenSpeed",
        "fastIonVelocity",
        "fastIonVPerp",
        "B0strength",
        "B0strength_percentile",
        "pitch",
        "pitch_percentile",
        "backgroundDensity",
        "backgroundDensity_percentile",
        "beamFraction",
        "beamFraction_percentile",
        "prediction_error_B0strength", 
        "prediction_error_pitch", 
        "prediction_error_backgroundDensity", 
        "prediction_error_beamFraction",
        "error_rank_B0strength",
        "error_rank_pitch",
        "error_rank_backgroundDensity",
        "error_rank_beamFraction",
        "mean_error_rank"]
    
    preds_by_all_sim = []
    for sim_id in list(dict.fromkeys(mrh_preds_pd["datapoint_ID"].values).keys()):
        preds_by_single_sim = dict.fromkeys(columns)
        simData = mrh_preds_pd[mrh_preds_pd["datapoint_ID"] == sim_id]
        preds_by_single_sim["sim_ID"] = int(sim_id)
        ranks = []
        norm_errors = []
        for field in list(dict.fromkeys(mrh_preds_pd["outputQuantity"].values).keys()):
            preds_by_single_sim[field] = simData[simData["outputQuantity"] == field]["trueValue_denormalised"].values[0]
            preds_by_single_sim[f"prediction_error_{field}"] = simData[simData["outputQuantity"] == field]["squaredError"].values[0]
            norm_errors.append(preds_by_single_sim[f"prediction_error_{field}"])
            rank = simData[simData["outputQuantity"] == field]["squaredError_rank"].values[0]
            preds_by_single_sim[f"error_rank_{field}"] = rank
            ranks.append(rank)
        preds_by_single_sim["mean_error_rank"] = np.mean(ranks)
        preds_by_single_sim["mean_normed_error"] = np.mean(norm_errors)

        with open(inputDecks / f"run_{int(sim_id)}/input.deck") as f:
            inputDeck = epydeck.loads(f.read())["constant"]
            assert np.isclose(inputDeck["b0_strength"], preds_by_single_sim["B0strength"])
            assert np.isclose(inputDeck["background_density"], preds_by_single_sim["backgroundDensity"])
            ring_beam_energy = inputDeck["ring_beam_energy"]
            ring_beam_momentum = np.sqrt(2.0 * ppp.alpha.mass.value * ring_beam_energy * constants.elementary_charge)
            preds_by_single_sim["alfvenSpeed"] = pps.Alfven_speed(
                B = inputDeck["b0_strength"] * u.T, 
                density = ((inputDeck["background_density"] - (inputDeck["frac_beam"] * inputDeck["background_density"] * ppp.alpha.charge_number)) / ppp.deuteron.charge_number) * u.m**-3,
                ion = "D+"
            ).value
            preds_by_single_sim["fastIonVelocity"] = np.sqrt((2.0 * ring_beam_energy * constants.elementary_charge) / ppp.alpha.mass.value)
            preds_by_single_sim["fastIonVPerp"] = (ring_beam_momentum * np.sqrt(1.0 - inputDeck["pitch"]**2)) / ppp.alpha.mass.value
            preds_by_single_sim["vPerpToAlfvenRatio"] = preds_by_single_sim["fastIonVPerp"] / preds_by_single_sim["alfvenSpeed"]

        preds_by_all_sim.append(preds_by_single_sim)
    
    preds_df = pd.DataFrame(preds_by_all_sim)
    preds_df["B0strength_percentile"] = preds_df["B0strength"].rank(ascending=True, pct=True)
    preds_df["pitch_percentile"] = preds_df["pitch"].rank(ascending=True, pct=True)
    preds_df["backgroundDensity_percentile"] = preds_df["backgroundDensity"].rank(ascending=True, pct=True)
    preds_df["beamFraction_percentile"] = preds_df["beamFraction"].rank(ascending=True, pct=True)
    
    centre_of_parameter_space = np.array([epoch_utils.norm_values(preds_df["B0strength"]).median(), epoch_utils.norm_values(preds_df["pitch"]).median(), epoch_utils.norm_values(preds_df["backgroundDensity"]).median(), epoch_utils.norm_values(preds_df["beamFraction"]).median()])
    centre_of_parameter_space = np.tile(centre_of_parameter_space, (100, 1)).transpose()
    data = np.array([epoch_utils.norm_values(preds_df["B0strength"]).values, epoch_utils.norm_values(preds_df["pitch"]).values, epoch_utils.norm_values(preds_df["backgroundDensity"]).values, epoch_utils.norm_values(preds_df["beamFraction"]).values])
    dists = data - centre_of_parameter_space
    squareDists = dists**2
    sumSquareDists = np.sum(squareDists, axis=0)
    sqrtSumSquareDists = np.sqrt(sumSquareDists)
    euclidean_distance_to_centre_of_parameter_space = sqrtSumSquareDists
    preds_df["normedEuDist_to_median_params"] = euclidean_distance_to_centre_of_parameter_space
    preds_df["normedEuDist_to_median_params_rank"] = preds_df["normedEuDist_to_median_params"].rank()

    preds_df.to_csv(csvPredictions.parent / "predictionRanks_bySim.csv")
    mrh_preds_pd.to_csv(csvPredictions.parent / "predictionRanks_all.csv")

    # preds_df.plot.scatter(x = "B0strength_percentile", y = "error_rank_B0strength")
    # plt.show()

    # preds_df.plot.scatter(x = "pitch_percentile", y = "error_rank_pitch")
    # plt.show()

    # preds_df.plot.scatter(x = "backgroundDensity_percentile", y = "error_rank_backgroundDensity")
    # plt.show()

    # preds_df.plot.scatter(x = "beamFraction_percentile", y = "error_rank_beamFraction")
    # plt.show()

    # preds_df["density percentile, absolute distance from 0.5"] = abs(0.5 - preds_df["backgroundDensity_percentile"])
    # xv = preds_df["density percentile, absolute distance from 0.5"]
    # preds_df.plot.scatter(x = "density percentile, absolute distance from 0.5", y = "error_rank_backgroundDensity")
    # resultLR = linregress(xv.values, preds_df["error_rank_backgroundDensity"].values)
    # plt.plot(xv.values, resultLR.slope * xv.values + resultLR.intercept, color="red")
    # plt.title(f"r^2 = {resultLR.rvalue**2}")
    # plt.show()

    # mean_parameter_percentile = np.abs(0.5 - preds_df[["B0strength_percentile", "pitch_percentile", "backgroundDensity_percentile", "beamFraction_percentile"]].mean(axis=1).values)
    # plt.scatter(x = mean_parameter_percentile, y = preds_df["mean_error_rank"].values)
    # resultLR = linregress(mean_parameter_percentile, preds_df["mean_error_rank"].values)
    # plt.plot(mean_parameter_percentile, resultLR.slope * mean_parameter_percentile + resultLR.intercept, color="red")
    # plt.xlabel("mean parameter percentile, absolute distance from 0.5")
    # plt.ylabel("mean error rank")
    # plt.title(f"r^2 = {resultLR.rvalue**2}")
    # plt.show()

    # preds_df.plot.scatter(x = "B0strength_percentile", y = "mean_error_rank")
    # plt.show()
    # preds_df.plot.scatter(x = "pitch_percentile", y = "mean_error_rank")
    # plt.show()
    # preds_df.plot.scatter(x = "backgroundDensity_percentile", y = "mean_error_rank")
    # plt.show()
    # preds_df.plot.scatter(x = "beamFraction_percentile", y = "mean_error_rank")
    # plt.show()

    # preds_df["beamFraction percentile, absolute distance from 0.5"] = abs(0.5 - preds_df["beamFraction_percentile"])
    # xv = preds_df["beamFraction percentile, absolute distance from 0.5"]
    # preds_df.plot.scatter(x = "beamFraction percentile, absolute distance from 0.5", y = "mean_error_rank")
    # resultLR = linregress(xv.values, preds_df["mean_error_rank"].values)
    # plt.plot(xv.values, resultLR.slope * xv.values + resultLR.intercept, color="red")
    # plt.title(f"r^2 = {resultLR.rvalue**2}")
    # plt.show()

    xv = preds_df["normedEuDist_to_median_params"]
    preds_df.plot.scatter(x = "normedEuDist_to_median_params", y = "mean_normed_error")
    resultLR = linregress(xv.values, preds_df["mean_normed_error"].values)
    plt.plot(xv.values, resultLR.slope * xv.values + resultLR.intercept, color="red")
    plt.title(f"r^2 = {resultLR.rvalue**2}")
    # plt.xscale("log")
    # plt.yscale("log")
    plt.show()

    print(preds_df["vPerpToAlfvenRatio"].to_string())
    print(f"vPerp/Alfven speed ranges from {preds_df['vPerpToAlfvenRatio'].min()} to {preds_df['vPerpToAlfvenRatio'].max()}, median = {preds_df['vPerpToAlfvenRatio'].median()}, mean = {preds_df['vPerpToAlfvenRatio'].mean()}, SD = {preds_df['vPerpToAlfvenRatio'].std()}")

    epoch_utils.my_matrix_plot(
        data_series=[[
            # preds_df["B0strength_percentile"], 
            # preds_df["pitch_percentile"], 
            preds_df["backgroundDensity_percentile"], 
            preds_df["beamFraction_percentile"], 
            # np.log(preds_df["prediction_error_B0strength"]),
            # np.log(preds_df["prediction_error_pitch"]),
            np.log(preds_df["prediction_error_backgroundDensity"]),
            np.log(preds_df["prediction_error_beamFraction"]),
            preds_df["vPerpToAlfvenRatio"],
        ]],
        parameter_labels=["density pct", "BF pct", "ln dens error", "ln BF error", "vPerp/AS"],
        show=True,
        plot_style="contour"
    )
    epoch_utils.my_matrix_plot(
        data_series=[[
            preds_df["B0strength_percentile"], 
            preds_df["pitch_percentile"], 
            preds_df["backgroundDensity_percentile"], 
            preds_df["beamFraction_percentile"], 
            preds_df["vPerpToAlfvenRatio"],
            preds_df["normedEuDist_to_median_params"],
            preds_df["mean_normed_error"],
        ]],
        parameter_labels=["B0 pct", "pitch pct", "density pct", "beam frac. pct", "vPerp/AS", "NEDTPSM", "mean error"],
        show=True,
        plot_style="contour"
    )
    epoch_utils.my_matrix_plot(
        data_series=[[
            np.log(preds_df["prediction_error_B0strength"]),
            np.log(preds_df["prediction_error_pitch"]),
            np.log(preds_df["prediction_error_backgroundDensity"]),
            np.log(preds_df["prediction_error_beamFraction"]),
            preds_df["vPerpToAlfvenRatio"],
            preds_df["normedEuDist_to_median_params"]
        ]],
        # parameter_labels=["B0 pct", "pitch pct", "density pct", "beam frac. pct", "B0 error", "pitch error", "density error", "beam frac. error", "vPerp/AS", "NEDTPSM", "mean error"],
        parameter_labels=["ln B0 error", "ln pitch error", "ln dens error", "ln BF error", "vPerp/AS", "NEDTPSM"],
        show=True,
        plot_style="contour"
    )

def sensitivity_analysis(prediction_error_csv : Path):

    parameter_names = ["B0strength", "pitch", "backgroundDensity", "beamFraction", "mean"]

    # Load data
    parameter_data = []
    error_data = {k : [] for k in parameter_names}
    # columns = [
    #     "sim_ID", 
    #     "alfvenSpeed",
    #     "fastIonVelocity",
    #     "fastIonVPerp",
    #     "B0strength",
    #     "B0strength_percentile",
    #     "pitch",
    #     "pitch_percentile",
    #     "backgroundDensity",
    #     "backgroundDensity_percentile",
    #     "beamFraction",
    #     "beamFraction_percentile",
    #     "prediction_error_B0strength", 
    #     "prediction_error_pitch", 
    #     "prediction_error_backgroundDensity", 
    #     "prediction_error_beamFraction",
    #     "error_rank_B0strength",
    #     "error_rank_pitch",
    #     "error_rank_backgroundDensity",
    #     "error_rank_beamFraction",
    #     "mean_error_rank"]
    with open(prediction_error_csv, "r") as csvf:
        csvr = csv.DictReader(csvf)
        for row in csvr:
            parameter_data.append(np.array([float(row["B0strength_percentile"]), float(row["pitch_percentile"]), float(row["backgroundDensity_percentile"]), float(row["beamFraction_percentile"])]))
            error_data["B0strength"].append(float(row["prediction_error_B0strength"]))
            error_data["pitch"].append(float(row["prediction_error_pitch"]))
            error_data["backgroundDensity"].append(float(row["prediction_error_backgroundDensity"]))
            error_data["beamFraction"].append(float(row["prediction_error_beamFraction"]))
            error_data["mean"].append(float(row["mean_normed_error"]))

    error_data = {k : np.array(v) for k,v in error_data.items()}
    parameter_data = np.array([np.array(a) for a in parameter_data])

    # Initialise GP kernel
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * RBF(length_scale=np.ones(4), length_scale_bounds=(1e-2, 1e5))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1.0))
    )
    kernel = (
        RationalQuadratic() + WhiteKernel()
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=20,
        random_state=0,
    )

    # Fit surrogate
    for param_name in parameter_names:

        print(f"Running PCE for parameter {param_name}....")
        error_output = error_data[param_name]

        # Cross-validate model
        kf = KFold(n_splits=10, shuffle=True)
        scores = []

        for train_idx, test_idx in kf.split(parameter_data):
            
            p = parameter_data[train_idx]
            e = error_output[train_idx]
            gp_fold = gp.fit(p, e)
            pred_fold = gp_fold.predict(parameter_data[test_idx])
            fold_r2 = r2_score(error_output[test_idx], pred_fold)
            print(f"Fold R^2: {fold_r2}")
            scores.append(fold_r2)

        print(f"Surrogate CV R^2: {np.mean(scores):.3f}")

        # # If R^2 < ~0.6–0.7, Sobol results will be questionable

        # # -----------------------------
        # # 6. Sobol indices (analytical)
        # # -----------------------------
        # S_first = cp.Sens_m(pce_model, dist)
        # S_total = cp.Sens_t(pce_model, dist)

        # param_names = ["x1", "x2", "x3", "x4"]

        # print("\nFirst-order Sobol indices:")
        # for name, s in zip(param_names, S_first):
        #     print(f"{name}: {s:.3f}")

        # print("\nTotal-order Sobol indices:")
        # for name, s in zip(param_names, S_total):
        #     print(f"{name}: {s:.3f}")

        # print(f"\nSum of first-order indices: {np.sum(S_first):.3f}")

def demo():

    # -----------------------------
    # 1. Load or define your data
    # -----------------------------
    # X: (n_samples, 4)
    # e: (n_samples,)
    # Replace these with your real data

    np.random.seed(0)
    n_samples = 100

    X = np.random.uniform(
        low=[0.5, 1.0, 10.0, 0.1],
        high=[1.5, 2.0, 20.0, 1.0],
        size=(n_samples, 4),
    )

    e = np.random.randn(n_samples) ** 2  # placeholder TSER error

    # -----------------------------
    # 2. Define input distributions
    # -----------------------------
    # Must match how X was sampled

    dist = cp.J(
        cp.Uniform(0.5, 1.5),   # x1
        cp.Uniform(1.0, 2.0),   # x2
        cp.Uniform(10.0, 20.0), # x3
        cp.Uniform(0.1, 1.0),   # x4
    )

    # Transpose X for chaospy (dim, samples)
    X_cp = X.T

    basis = cp.orth_ttr(2, dist)

    pce = cp.fit_regression(basis, X_cp, e)

    # In-sample diagnostic only
    e_pred = cp.call(pce, (X_cp[:,0]))
    e2_pred = pce(X_cp[:,0])
    print("In-sample R^2:", r2_score(e, e_pred[0,:]))
    print("In-sample R^2:", r2_score(e, e_pred[1,:]))
    print("In-sample R^2:", r2_score(e, e_pred[2,:]))
    print("In-sample R^2:", r2_score(e, e_pred[3,:]))


if __name__ == "__main__":

    # demo()
    # sensitivity_analysis(Path("/home/era536/Documents/Epoch/Data/2026_analysis/tsr_aeon_viking/LOOCV/predictionRanks_bySim.csv"))
    analyse_bad_predictions_2(Path("/home/era536/Documents/Epoch/Data/2026_analysis/tsr_aeon_viking/LOOCV/predictionRanks_bySim.csv"))