import argparse
import glob
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from plasmapy.formulary import frequencies as ppf
from plasmapy.formulary import speeds as pps
from plasmapy.formulary import lengths as ppl
from plasmapy import particles as ppp
from scipy import constants as constants
from scipy.stats import linregress
from sklearn.preprocessing import MinMaxScaler
from sdf_xarray import SDFPreprocess
import xarray as xr
import astropy.units as u
import dataclass_csv
import csv
import ml_utils
import epoch_utils
import copy
import pandas as pd
import epydeck
import math
import copy
import netCDF4 as nc
import xrft

def cold_plasma_dispersion_relation():
    # Constants
    B0 = 0.5  * u.T  # Background magnetic field (T)
    n_e = 1e19 * u.m**-3  # Electron density (m^-3)
    n_i = n_e  # Assume quasi-neutrality
    TWO_PI_RAD = 2.0 * np.pi * u.rad

    # Cyclotron frequencies
    Omega_i = ppf.gyrofrequency(B0, "p+")   # Ion cyclotron frequency rad/s
    Omega_e = ppf.gyrofrequency(B0, "e-")  # Electron cyclotron frequency rad/s
    normalised_frequency = Omega_e
    Omega_i_norm = Omega_i / normalised_frequency
    Omega_e_norm = Omega_e / normalised_frequency
    #Tau_i = (1.0 / Omega_i)
    v_A = pps.Alfven_speed(B0, n_i, "p+")  # Alfvén speed m/s

    c = constants.c * (u.m/u.s)

    # Plasma frequencies
    omega_pe = ppf.plasma_frequency(n_e, "e-")  # Electron plasma frequency rad/s
    omega_pe_norm = omega_pe / normalised_frequency
    omega_pi = ppf.plasma_frequency(n_e, "p+")  # Ion plasma frequency rad/s
    omega_pi_norm = omega_pi / normalised_frequency

    # Hybrid frequencies
    omega_uh = ppf.upper_hybrid_frequency(B0, n_e)  # Upper hybrid frequency rad/s
    omega_uh_norm = omega_uh / normalised_frequency
    omega_lh = ppf.lower_hybrid_frequency(B0, n_i, "p+")  # Lower hybrid frequency rad/s
    omega_lh_norm = omega_lh / normalised_frequency

    # Wavenumber range
    simLength = 0.0001 * u.m
    #normalised_wavenumber = normalised_frequency / c
    k_vals = np.linspace(0.0, TWO_PI_RAD/simLength, 100000)# Wavenumber range
    k_vals_norm = (k_vals * c) / normalised_frequency

    # O-mode dispersion (unmagnetized plasma mode)
    #omega_o_mode = np.sqrt(omega_pe**2 + (c * k_vals)**2)

    # X-mode dispersion
    #omega_x_mode = np.sqrt(0.5 * (omega_pe**2 + Omega_e**2) +
    #                        0.5 * np.sqrt((omega_pe**2 + Omega_e**2)**2 - 4 * Omega_e**2 * omega_pe**2))

    # Whistler mode (parallel propagation)
    omega_whistler = (k_vals**2 * c**2 * Omega_e) / omega_pe**2
    omega_whistler_norm = omega_whistler / normalised_frequency

    # Ion cyclotron mode
    #omega_ion_cyclotron = Omega_i / Omega_i  # Ion cyclotron resonance (constant)

    # Lower hybrid wave mode
    #omega_lower_hybrid = np.ones_like(k_vals) * omega_lh  # Constant lower hybrid frequency

    # Alfvén wave dispersion (low k approximation: ω = k * v_A)
    omega_alfven = k_vals * v_A  # Alfvén wave linear dispersion
    omega_alfven_norm = omega_alfven / normalised_frequency  # Alfvén wave linear dispersion

    omega_light = k_vals * c
    omega_light_norm = omega_light / normalised_frequency

    # Plot dispersion relations
    plt.figure(figsize=(8, 6))

    #plt.plot(k_vals, omega_o_mode, label="O-mode (Ordinary)", color="blue")
    #plt.axhline(y=omega_pe_norm.value, color="blue", linestyle="dashed", label=r"$\Omega_{pe}$")
    #plt.axhline(y=omega_pi_norm.value, color="cyan", linestyle="dashed", label=r"$\Omega_{pi}$")

    #plt.plot(k_vals, omega_x_mode * np.ones_like(k_vals), label="X-mode (Extraordinary)", color="red")
    #plt.axhline(y=omega_uh_norm.value, color="red", linestyle="dashdot", label=r"$\Omega_{UH}$")
    #plt.axhline(y=omega_lh_norm.value, color="orange", linestyle="dashdot", label=r"$\Omega_{LH}$ (Lower Hybrid)")

    plt.axhline(y=Omega_i_norm.value, color="purple", linestyle="dashed", label=r"$\Omega_i$ (Ion Cyclotron)")
    plt.axhline(y=Omega_e_norm.value, color="brown", linestyle="dashed", label=r"$\Omega_e$ (Electron Cyclotron)")

    plt.plot(k_vals_norm, omega_whistler_norm, label="Whistler Mode", color="green")
    plt.plot(k_vals_norm, omega_alfven_norm, label="Alfvén Wave", color="black", linestyle="dotted")
    plt.plot(k_vals_norm, omega_light_norm, label="Light Wave", color="black", linestyle="dotted")

    plt.xlabel("Wavenumber k (ω_ce/c)")
    plt.ylabel("Frequency ω (ω_ce)")
    plt.title("Cold Plasma Dispersion Relations with Ion Modes")
    plt.legend()
    #plt.yscale("log")
    #plt.xscale("log")
    plt.grid()
    plt.ylim(1E-4, 100)
    plt.xlim(1E-2, 100)
    plt.show()

def X(omega, omega_pe):
    return omega_pe**2 / omega**2

def Y(omega, omega_ce):
    return omega_ce / omega

def appleton_hartree():
    # https://en.wikipedia.org/wiki/Appleton%E2%80%93Hartree_equation

    # Constants
    B0 = 0.5 * u.T  # Background magnetic field (T)
    n_e = 1e18 * u.m**-3  # Electron density (m^-3)
    n_i = n_e  # Assume quasi-neutrality

    omega_pe = ppf.plasma_frequency(n_e, "e-").value
    omega_ce = ppf.gyrofrequency(B0, "e-").value
    omega_ci = ppf.gyrofrequency(B0, "p-").value
    #theta = 0.0 # Angle between k and B
    theta = np.pi / 2.0 # Angle between k and B

    omega_uh = ppf.upper_hybrid_frequency(B0, n_e).value
    omega_lh = ppf.lower_hybrid_frequency(B0, n_i, 'p+').value
    max_omega = 1.5 * omega_uh
    omega_vals = np.linspace(omega_lh, max_omega, 100000)

    # omega_pe^2 * (1 - X)
    numerator = 1.0 * X(omega_vals, omega_pe)

    # Part before +/-: 1 - X - (0.5Y^2 * sin^2(theta))
    denominator_1 = 1.0 - X(omega_vals, omega_pe) - (0.5 * Y(omega_vals, omega_ce)**2 * np.sin(theta)**2)

    # Part after +/-: sqrt((0.5Y^2 * sin^2(theta))^2 + ((1-X)^2 * Y^2 * cos^2(theta)))
    denominator_2 = np.sqrt(
        (0.5 * Y(omega_vals, omega_ce)**2 * np.sin(theta)**2)**2 
        + ((1.0 - X(omega_vals, omega_pe))**2 * Y(omega_vals, omega_ce)**2 * np.cos(theta)**2)
    )

    n_squared_plus = 1.0 - (numerator / (denominator_1 + denominator_2))
    n_squared_minus = 1.0 - (numerator / (denominator_1 - denominator_2))

    plt.axvline(omega_uh, color = 'black', linestyle = 'dotted', label = "Upper hybrid")
    plt.axvline(omega_lh, color = 'black', linestyle = 'dashed', label = "Lower hybrid")
    plt.axvline(omega_ce, color = 'blue', linestyle = 'dotted', label = "Electron gyrofrequency")
    plt.axvline(omega_pe, color = 'blue', linestyle = 'dashed', label = "Electron plasma frequency")
    plt.title(f"theta = {theta * 180.0 / np.pi} degrees")
    plt.plot(omega_vals, n_squared_plus, color = 'red', label="n^2 +")
    plt.plot(omega_vals, n_squared_minus, color = 'green', label="n^2 -")
    plt.xlabel("omega")
    plt.ylabel("refractive index^2 [(kc/omega)^2]")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    #plt.ylim(0.0, max_omega)
    #plt.xlim(np.min([np.min(k_squared_plus), np.min(k_squared_minus)]), np.max([np.max(k_squared_plus), np.max(k_squared_minus)]))
    #plt.xscale('log')
    #plt.yscale('log')
    plt.show()

def trigonometry():

    theta = np.linspace(0.0, 2.0 * np.pi, 10000)

    me_over_mi = constants.electron_mass / constants.proton_mass
    trig_func = np.cos(theta)**2

    crosses = []
    below = False
    index = 0

    for y in trig_func:
        if below:
            if y >= me_over_mi:
                crosses.append(theta[index])
                below = False
        else:
            if y <= me_over_mi:
                crosses.append(theta[index])
                below = True
        index += 1

    print(f"Crossing points: {crosses}")
    print(f"cos^2 (theta) is <= me/mi from {crosses[0]} ({crosses[0] * 180/np.pi}deg) to {crosses[1]} ({crosses[1] * 180/np.pi}deg) and {crosses[2]} ({crosses[2] * 180/np.pi}deg) to {crosses[3]} ({crosses[3] * 180/np.pi}deg)")

    plt.plot(theta, trig_func, "r")
    plt.axhline(me_over_mi, linestyle="dashed", color="blue", label = "me/mi")
    plt.xlabel("Theta/rad")
    plt.legend()
    plt.ylabel("cos^2 (theta)")
    plt.show()

    # Function is <= me/mi from 1.5477rad - 1.5942rad (88.677 - 91.341)

def verdon_et_al():
    # Follows equ 2.1 and 2.2 in: https://www.cambridge.org/core/services/aop-cambridge-core/content/view/S1743921309029871

    # Constants
    B0 = 0.5 * u.T  # Background magnetic field (T)
    n_e = 1e18 * u.m**-3  # Electron density (m^-3)
    n_i = n_e  # Assume quasi-neutrality
    thetaDeg_all = np.linspace(0.0, 180.0, 2000)
    theta_all = thetaDeg_all * (np.pi / 180.0)
    k_max = 2000.0
    k = np.linspace(0.00001, k_max, 100000)
    k_norm_factor =  ppf.gyrofrequency(B0, "p+").value / pps.Alfven_speed(B0, n_i, "p+").value
    normalised_k = k / k_norm_factor
    all_intersections = []

    for theta in theta_all:
        # 2.1: plasma is cold but incluidng EM effects
        omega_pe = ppf.plasma_frequency(n_e, "e-").value
        w_squared_over_wLH_squared = ((1.0 / (1.0 + (omega_pe**2 / (k**2 * constants.c**2))))
                                    * (1.0 + ((constants.proton_mass / constants.electron_mass) * (np.cos(theta)**2 / (1.0 + (omega_pe**2 / (k**2 * constants.c**2))))))
                                    )
        
        diff = abs(w_squared_over_wLH_squared - 1.0)
        index = np.argmin(diff)
        all_intersections.append(normalised_k[index])

    plt.plot(thetaDeg_all, all_intersections)
    plt.xlabel("Theta/degrees")
    plt.ylabel(r"k at $\Omega = \pm\Omega_{LH}$")
    plt.title(r"Intersections of k and $\Omega = \pm\Omega_{LH}$, by angle")
    plt.show()

    midpoint_index = int(len(thetaDeg_all)/2)
    plt.plot(thetaDeg_all[midpoint_index-29:midpoint_index+30], all_intersections[midpoint_index-29:midpoint_index+30])
    plt.xlabel("Theta/degrees")
    plt.grid()
    plt.ylabel(r"k at $\Omega = \pm\Omega_{LH}$")
    plt.title(r"Intersections of k and $\Omega = \pm\Omega_{LH}$, near theta = 90")
    plt.show()

    # plt.title(f"Theta = {thetaDeg}deg")
    # plt.axhline(1.0, color = "gray", linestyle="dotted", label = r"$\Omega = \pm\Omega_{LH}$")
    # plt.axvline(intersection_1, color = "green", linestyle="dashed", label = f"k = {intersection_1:.3f}")
    # plt.axvline(intersection_2, color = "green", linestyle="dashed", label = f"k = {intersection_2:.3f}")
    # plt.plot(normalised_k, w_squared_over_wLH_squared)
    # plt.legend()
    # plt.xlabel(r"$kV_A/\Omega_i$")
    # plt.ylabel(r"$\Omega^2/\Omega_{LH}^2$")
    # plt.show()

    # 2.2: plasma is warm but assumes wave is electrostatic/longitudinal

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

def analyse_x_gyrofreqs(dataDir: Path, num_gyro_freqs : float, outputFolder : Path):
    
    spectraTypes = ["Magnetic_Field_Bz", "Electric_Field_Ex", "Electric_Field_Ey"]

    if dataDir.name != "data":
        data_dir = dataDir / "data"
    else:
        data_dir = dataDir
    data_files = glob.glob(str(data_dir / "*.nc"))

    # Pre-check for data length:
    index_counts = []
    for simulation in data_files:

        data = xr.open_datatree(
            simulation,
            engine="netcdf4"
        )

        for spectra in spectraTypes:
            num_indices = float(data[f"/{spectra}/power/powerByFrequency"].sel(frequency = slice(0.0, num_gyro_freqs)).size)
            index_counts.append(num_indices)
    
    if np.max(index_counts) != np.min(index_counts):
        print(f"Spectra would be different lengths: {np.max(index_counts)} vs {np.min(index_counts)}. Defaulting to minimum.")
    
    indices_to_keep = int(np.min(index_counts))

    dataset_len = 0
    max_frequencies = []
    for simulation in data_files:

        data = xr.open_datatree(
            simulation,
            engine="netcdf4"
        )
        newData : xr.DataTree = copy.deepcopy(data)

        sim_name = Path(simulation).name

        new_data_len = []
        for spectra in spectraTypes:

            dataset_len = newData[f"/{spectra}/power/powerByFrequency"].size
            frequencies = newData[f"/{spectra}/power/powerByFrequency"].coords["frequency"]
            new_range = newData[f"/{spectra}/power/powerByFrequency"].isel(frequency = slice(0, indices_to_keep))
            new_freqs = newData[f"/{spectra}/power/powerByFrequency"].coords["frequency"].isel(frequency = slice(0, indices_to_keep))
            new_len = new_freqs.size

            print(f"Spectra: {spectra}: Current len: {dataset_len}, Max frequency: {float(frequencies[-1])} gyrofrequencies.")
            print(f"Spectra: {spectra}: Truncating to {num_gyro_freqs} gyrofrequencies ({new_len} indices).")

            data_attrs = newData[f"/{spectra}/power/powerByFrequency"].attrs
            coords_attrs = newData[f"/{spectra}/power/frequency"].attrs
            newData[f"/{spectra}/power"] = newData[f"/{spectra}/power"].to_dataset().drop_vars("powerByFrequency")
            newData[f"/{spectra}/power"] = newData[f"/{spectra}/power"].to_dataset().drop_dims("frequency")
            newData[f"/{spectra}/power/powerByFrequency"] = new_range
            newData[f"/{spectra}/power/frequency"] = new_freqs
            newData[f"/{spectra}/power/powerByFrequency"] = newData[f"/{spectra}/power/powerByFrequency"].assign_coords(frequency = new_freqs)

            newPowerSpectra = newData[f"/{spectra}/power/powerByFrequency"]

            max_real_f = (float(newPowerSpectra.coords["frequency"][-1] * (data.ionGyrofrequency_radPs / (2.0 * np.pi))) * u.Hz).to(u.MHz)
            max_frequencies.append(max_real_f)
            print(f"Truncated spectra has len {newPowerSpectra.size}, max freq {float(newPowerSpectra.coords['frequency'][-1])} gyrofrequencies ({max_real_f}).")

            # print(newData[f"/{spectra}/power/frequency"].to_numpy())
            # print(newData[f"/{spectra}/power/powerByFrequency"].to_numpy())
            # plt.plot(data[f"/{spectra}/power/frequency"].to_numpy(), data[f"/{spectra}/power/powerByFrequency"].to_numpy(), color = "blue")
            # plt.plot(newData[f"/{spectra}/power/frequency"].to_numpy(), newData[f"/{spectra}/power/powerByFrequency"].to_numpy(), color = "red")
            # plt.xlabel("Gyrofrequencies")
            # plt.show()

            assert new_len == newData[f"/{spectra}/power/powerByFrequency"].size
            new_data_len.append(new_len)
        
        assert new_data_len[0] == new_data_len[1]
        assert new_data_len[1] == new_data_len[2]

        newData.to_netcdf(outputFolder / sim_name.replace("stats.nc", f"{num_gyro_freqs}_gyros_stats.nc"))

    fs = [f.value for f in max_frequencies]
    print(f"Max frequencies range from {np.min(fs)} to {np.max(fs)}MHz.")

def analyse_real_frequencies(combined_stats_folder : Path):

    spectraTypes = ["Magnetic_Field_Bz", "Electric_Field_Ex", "Electric_Field_Ey"]

    if combined_stats_folder.name != "data":
        data_dir = combined_stats_folder / "data"
    else:
        data_dir = combined_stats_folder
    data_files = glob.glob(str(data_dir / "*.nc"))

    all_simulation_data = {}
    maximum_frequencies = []
    first_frequencies = []
    dataset_len = 0

    for simulation in data_files:

        data = xr.open_datatree(
            simulation,
            engine="netcdf4"
        )
        
        all_simulation_data[Path(simulation).name] = data
        dataset_len = data["/Magnetic_Field_Bz/power/frequencyPowerSpectrum"].size
        first_frequencies.append((data["/Magnetic_Field_Bz/power/frequencyPowerSpectrum"].coords["frequency"][0] * data.ionGyrofrequency_radPs) / (2.0 * np.pi))

        max_gyrofrequency_radPs = data["/Magnetic_Field_Bz/power/frequency"] * data.ionGyrofrequency_radPs
        f1 = u.Hz * float(max_gyrofrequency_radPs[-1]) / (2.0 * np.pi)
        # f2 = 80.0 * ppf.gyrofrequency(B = data.B0strength * u.T, particle = "alpha") / (2.0 * np.pi * u.rad)
        # print(f"f1 = {f1}, f2 = {f2} ({((f1-f2.value)/f1)*100.0}% difference)")
        print(f"Max frequency = {f1}, {f1.to(u.kHz)}, {f1.to(u.MHz)}")
        
        maximum_frequencies.append(f1.value)

    lowest_max_idx = np.argmin(maximum_frequencies)
    print(f"Lowest maximum frequency of {(maximum_frequencies[lowest_max_idx] * u.Hz).to(u.MHz)} found in simulation '{[n for n in all_simulation_data.keys()][lowest_max_idx]}'")
    lowest_max_freq = maximum_frequencies[lowest_max_idx]

    # Get new time coordinates
    new_time_points_start = first_frequencies[lowest_max_idx]
    if new_time_points_start == 0:
        new_time_points_start = 1E-20

    equal_f_folder = Path("/home/era536/Documents/Epoch/Data/2026_analysis/combined_spectra_equal_freqs_4/data/")
    if not os.path.exists(equal_f_folder):
        os.makedirs(equal_f_folder)
    cottrell_folder = Path("/home/era536/Documents/Epoch/Data/2026_analysis/combined_spectra_cottrell_4/data/")
    if not os.path.exists(cottrell_folder):
        os.makedirs(cottrell_folder)

    # 0-1 scaler for Cottrell data
    scaler_cottrell = MinMaxScaler(feature_range=(0, 1))
    # Correll frequency
    cottrell_max_freq = (186.6285833 * u.MHz).to(u.Hz).value

    for sim_name, data in all_simulation_data.items():
        newData : xr.DataTree = copy.deepcopy(data)
        print(f"Equal Real F: Truncating {sim_name} to a maximum frequency of {(lowest_max_freq * u.Hz).to(u.MHz)}....")

        # Lowest max real frequency
        for spectra in spectraTypes:
            # All series should be the same length (will need to be interpolated onto new coordinates)
            assert dataset_len == newData[f"/{spectra}/power/frequencyPowerSpectrum"].size

            frequencies_hz = (newData[f"/{spectra}/power/frequencyPowerSpectrum"].coords["frequency"] * newData.ionGyrofrequency_radPs) / (2.0 * np.pi)

            print(f"Current len: {dataset_len}, Max frequency: {(float(frequencies_hz[-1]) * u.Hz).to(u.MHz)} ({float(newData[f'/{spectra}/power/frequencyPowerSpectrum'].coords['frequency'][-1])} gyrofrequencies).")

            newData[f"/{spectra}/power/frequency"] = frequencies_hz
            newData[f"/{spectra}/power/frequencyPowerSpectrum"] = newData[f"/{spectra}/power/frequencyPowerSpectrum"].assign_coords(frequency = frequencies_hz)

            newPowerSpectra = newData[f"/{spectra}/power/frequencyPowerSpectrum"].sel(frequency = slice(0.0, lowest_max_freq))

            print(f"Truncated spectra has len {newPowerSpectra.size}, Max freq {(float(newPowerSpectra.coords['frequency'][-1]) * u.Hz).to(u.MHz)} ({float(newPowerSpectra.coords['frequency'][-1]) * (2.0 * np.pi / newData.ionGyrofrequency_radPs)} gyrofrequencies), interpolating to size {dataset_len}....")

            new_freq_points = np.linspace(float(new_time_points_start), lowest_max_freq, dataset_len)
            newPowerSpectra = newPowerSpectra.interp(frequency = new_freq_points, kwargs={"fill_value": "extrapolate"})
            
            newData[f"/{spectra}/power/frequencyPowerSpectrum"].data = newPowerSpectra.data
            newData[f"/{spectra}/power/frequency"] = new_freq_points
            newData[f"/{spectra}/power/frequencyPowerSpectrum"] = newData[f"/{spectra}/power/frequencyPowerSpectrum"].assign_coords(frequency = new_freq_points)
            newData[f"/{spectra}/power"] = newData[f"/{spectra}/power"].assign(frequency_gyro = xr.DataArray((new_freq_points * (2.0 * np.pi)) / newData.ionGyrofrequency_radPs))

            # print(newData[f"/{spectra}/power/frequency_gyro"].to_numpy())
            # print(newData[f"/{spectra}/power/frequencyPowerSpectrum"].to_numpy())
            # plt.plot(data[f"/{spectra}/power/frequency"].to_numpy(), data[f"/{spectra}/power/frequencyPowerSpectrum"].to_numpy(), color = "blue")
            # plt.plot(newData[f"/{spectra}/power/frequency_gyro"].to_numpy(), newData[f"/{spectra}/power/frequencyPowerSpectrum"].to_numpy(), color = "red")
            # plt.xlabel("Gyrofrequencies")
            # plt.show()

            print(f"New len: {newData[f'/{spectra}/power/frequencyPowerSpectrum'].size}, Max frequency: {(float(newData[f'/{spectra}/power/frequencyPowerSpectrum'].coords['frequency'][-1]) * u.Hz).to(u.MHz)} ({float(newData[f'/{spectra}/power/frequency_gyro'][-1])} gyrofrequencies).")

        newData.to_netcdf(equal_f_folder / sim_name.replace("stats.nc", "equalF_stats.nc"))

        newData : xr.DataTree = copy.deepcopy(data)
        print(f"Cottrell F: Truncating {sim_name} to a maximum frequency of {(cottrell_max_freq * u.Hz).to(u.MHz)} and normalising....")
        for spectra in spectraTypes:

            # All series should be the same length (will need to be interpolated onto new coordinates)
            assert dataset_len == newData[f"/{spectra}/power/frequencyPowerSpectrum"].size
            
            frequencies_hz = (newData[f"/{spectra}/power/frequencyPowerSpectrum"].coords["frequency"] * newData.ionGyrofrequency_radPs) / (2.0 * np.pi)

            print(f"Current len: {len(frequencies_hz)}, Max frequency: {(float(frequencies_hz[-1]) * u.Hz).to(u.MHz)} ({float(newData[f'/{spectra}/power/frequencyPowerSpectrum'].coords['frequency'][-1])} gyrofrequencies).")

            newData[f"/{spectra}/power/frequency"] = frequencies_hz
            newData[f"/{spectra}/power/frequencyPowerSpectrum"] = newData[f"/{spectra}/power/frequencyPowerSpectrum"].assign_coords(frequency = frequencies_hz)

            newPowerSpectra = newData[f"/{spectra}/power/frequencyPowerSpectrum"].sel(frequency = slice(0.0, cottrell_max_freq))

            print(f"Truncated spectra has len {newPowerSpectra.size}, Max freq {(float(newPowerSpectra.coords['frequency'][-1]) * u.Hz).to(u.MHz)} ({float(newPowerSpectra.coords['frequency'][-1])} gyrofrequencies), interpolating to size {dataset_len}....")

            new_freq_points = np.linspace(float(new_time_points_start), cottrell_max_freq, dataset_len)
            newPowerSpectra = newPowerSpectra.interp(frequency = new_freq_points, kwargs={"fill_value": "extrapolate"})

            # Avoid spurious first value
            newPowerSpectrum = newPowerSpectra.data
            newPowerSpectrum[0] = newPowerSpectra.data[1]

            # Log, scale and write
            newData[f"/{spectra}/power/frequencyPowerSpectrum"].data = scaler_cottrell.fit_transform(np.log10(newPowerSpectrum).reshape(-1, 1)).flatten()
            newData[f"/{spectra}/power/frequency"] = new_freq_points
            newData[f"/{spectra}/power/frequencyPowerSpectrum"] = newData[f"/{spectra}/power/frequencyPowerSpectrum"].assign_coords(frequency = new_freq_points)
            newData[f"/{spectra}/power"] = newData[f"/{spectra}/power"].assign(frequency_gyro = xr.DataArray((new_freq_points * (2.0 * np.pi)) / newData.ionGyrofrequency_radPs))

            # print(newData[f"/{spectra}/power/frequency_gyro"].to_numpy())
            # print(newData[f"/{spectra}/power/frequencyPowerSpectrum"].to_numpy())
            # _, ax1 = plt.subplots()
            # ax1.plot(data[f"/{spectra}/power/frequency"].to_numpy(), data[f"/{spectra}/power/frequencyPowerSpectrum"].to_numpy(), color = "blue")
            # ax2 = ax1.twinx()
            # ax2.plot(newData[f"/{spectra}/power/frequency_gyro"].to_numpy(), newData[f"/{spectra}/power/frequencyPowerSpectrum"].to_numpy(), color = "red")
            # plt.xlabel("Gyrofrequencies")
            # plt.show()

            # plt.plot(newData[f"/{spectra}/power/frequency"].to_numpy() / 1e6, newData[f"/{spectra}/power/frequencyPowerSpectrum"].to_numpy())
            # plt.xlabel("Frequencies")
            # plt.show()

            # plt.plot(newData[f"/{spectra}/power/frequency_gyro"].to_numpy(), newData[f"/{spectra}/power/frequencyPowerSpectrum"].to_numpy())
            # plt.xlabel("Gyrofrequencies")
            # plt.show()

            print(f"New len: {newData[f'/{spectra}/power/frequencyPowerSpectrum'].size}, Max frequency: {(float(newData[f'/{spectra}/power/frequencyPowerSpectrum'].coords['frequency'][-1]) * u.Hz).to(u.MHz)} ({float(newData[f'/{spectra}/power/frequency_gyro'][-1])} gyrofrequencies).")

        newData.to_netcdf(cottrell_folder / sim_name.replace("stats.nc", "cottrellRange_stats.nc"))

def fourier_power(simulation_dataPath : Path):

    ##### Known synthetic signal
    print("Evaluating synthetic signal....")
    # Generate signal
    num_time_points = 1024
    num_space_points = 1024
    max_t = 4.0
    max_x = 4.0
    omega = 0.8 * 2.0 * np.pi
    omega_2 = 3.7 * 2.0 * np.pi
    k = 2.0 * 2.0 * np.pi
    k_2 = 0.2 * 2.0 * np.pi
    x_flat, dx = np.linspace(0.0, max_x, num_space_points, endpoint=False, retstep=True)
    t_flat, dt = np.linspace(0.0, max_t, num_time_points, endpoint=False, retstep=True)
    # generate 2 2d grids for the x & y bounds
    t, x = np.meshgrid(t_flat, x_flat)
    y = 2.7 * (np.sin((k * x) - (omega * t)) + np.sin((k_2 * x) - (omega_2 * t)))
    max_range = 5

    print(f"Input ")
    print(f"Max y: {np.max(y)}, Min y: {np.min(y)}, RMS: {np.sqrt(np.mean(y**2))}, RMS^2: {np.mean(y**2)}, V_RMS: {(1/np.sqrt(2))*np.max(y)}, V_RMS^2: {((1/np.sqrt(2))*np.max(y))**2}")
    print(f"Parseval: Original Signal: sum(abs(signal)**2) * dx * dt = {np.sum(np.abs(y)**2) * dx * dt}")
    plt.title("Original signal")
    qm = plt.pcolormesh(x, t, y, cmap="plasma")
    plt.colorbar(qm, label = "Field [T]")
    plt.xlabel("space [m]")
    plt.ylabel("time [s]")
    plt.grid()
    plt.show()
    
    signal = xr.DataArray(y, coords={"space" : x_flat, "time" : t_flat}, dims=["space", "time"])
    fft = xrft.fft(signal, true_phase=True, true_amplitude=True)
    print(f"Parseval: Fourier Transform: sum(abs(fft)**2) * dk * dw = {float(np.sum(np.abs(fft)**2) * fft.coords['freq_space'].spacing * fft.coords['freq_time'].spacing)}")
    plt.title("FFT")
    np.abs(fft).plot(cbar_kwargs={'label': "Power [Tm/Hz]"})
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Wavenumber [m^-1]")
    plt.xlim(-max_range,max_range)
    plt.ylim(-max_range,max_range)
    plt.grid()
    plt.show()

    # spec = fft.sel(freq_time=fft.freq_time>0.0)
    # freq_power_spectra = spec.sum(dim = "freq_space")
    # naive_freq_sum = float(2.0 * np.sum(np.abs(freq_power_spectra)**2) * fft.coords['freq_space'].spacing * freq_power_spectra.coords['freq_time'].spacing)
    # naive_freq_sum += float(np.abs(fft.sel(freq_time=fft.freq_time==0.0)).sum(dim = "freq_space") * fft.coords['freq_space'].spacing * freq_power_spectra.coords['freq_time'].spacing) # Add 0-freq component
    # print(f"Parseval: Freq spectrum (naive sum, y = sum(fft)): 2 * sum(abs(y)**2) * dk * dw = {naive_freq_sum}")
    # np.abs(freq_power_spectra).plot()
    # plt.xlabel("Frequency [Hz]")
    # plt.ylabel("Power [Tm/Hz]")
    # plt.title("Naive sum of FFT over all k")
    # plt.xlim((-0.5, max_range))
    # plt.grid()
    # plt.show()

    # freq_power_spectra_better = np.sqrt((np.abs(spec)**2).sum(dim = "freq_space"))
    # better_freq_sum = float(2.0 * np.sum(np.abs(freq_power_spectra_better)**2) * fft.coords['freq_space'].spacing * freq_power_spectra_better.coords['freq_time'].spacing)
    # better_freq_sum += float(np.sqrt((np.abs(fft.sel(freq_time=fft.freq_time==0.0))**2).sum()) * fft.coords['freq_space'].spacing * freq_power_spectra_better.coords['freq_time'].spacing) # Add 0-freq component
    # print(f"Parseval: Freq spectrum (better sum, y = sum(sqrt(abs(fft)**2))) 2 * sum(abs(y)**2) * dk * dw =  {better_freq_sum}")
    # np.abs(freq_power_spectra_better).plot()
    # plt.title("Sqrt of Squared sum of FFT over all k")
    # plt.xlabel("Frequency [Hz]")
    # plt.ylabel("Power [Tm/Hz]")
    # plt.xlim((-0.5, max_range))
    # plt.grid()
    # plt.show()

    # Power spectrum
    ps = xrft.power_spectrum(signal, scaling = "spectrum", window = "hann", window_correction=False)
    print(f"Parseval: PS: sum(PS) / (dk * dw) = {float(np.sum(ps) / (ps.coords['freq_space'].spacing * ps.coords['freq_time'].spacing))}")
    ps.plot(cbar_kwargs={'label': "Power [T^2]"})
    plt.title("Power spectrum")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Wavenumber [m^-1]")
    plt.xlim(-max_range,max_range)
    plt.ylim(-max_range,max_range)
    plt.grid()
    plt.show()

    spec = ps.sel(freq_time=ps.freq_time>0.0)
    freq_power_spectra = spec.sum(dim = "freq_space")
    ps_freq_sum = float(2.0 * np.sum(freq_power_spectra) / (ps.coords['freq_space'].spacing * freq_power_spectra.coords['freq_time'].spacing))
    ps_freq_sum += float(np.sum(ps.sel(freq_time=ps.freq_time==0.0)) / (ps.coords['freq_space'].spacing * freq_power_spectra.coords['freq_time'].spacing))
    print(f"Parseval: Freq PS (naive sum, y = sum(PS)): 2 * sum(y) / (dk * dw) = {ps_freq_sum}")
    freq_power_spectra.plot()
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power [T^2]")
    plt.title("Naive sum of Power spectrum over all k")
    plt.xlim((-0.5, max_range))
    plt.grid()
    plt.show()

    # freq_power_spectra = xrft.power_spectrum(signal, dim="time", scaling = "spectrum")
    # freq_power_spectra = freq_power_spectra.sel(freq_time=ps.freq_time>0.0).sum(dim = "space")
    # ps_freq_sum = float((2.0 * np.sum(freq_power_spectra) * dx) / freq_power_spectra.coords['freq_time'].spacing)
    # ps_freq_sum += float((np.sum(ps.sel(freq_time=ps.freq_time==0.0)) * dx) / freq_power_spectra.coords['freq_time'].spacing)
    # print(f"Parseval: Freq PS (single dim, y = sum(PS)): (2 * sum(y) * dx) / dw = {ps_freq_sum}")
    # freq_power_spectra.plot()
    # plt.xlabel("Frequency [Hz]")
    # plt.ylabel("Power [T^2]")
    # plt.title("Power spectrum over one dim only")
    # plt.xlim((-0.5, max_range))
    # plt.grid()
    # plt.show()

    # Power spectral density
    psd = xrft.power_spectrum(signal, scaling = "density", window = "hann", window_correction=True)
    print(f"Parseval: PSD: sum(PSD) = {float(np.sum(psd))}")
    psd.plot(cbar_kwargs={'label': "Power [T^2 m/Hz]"})
    plt.title("Power spectral density")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Wavenumber [m^-1]")
    plt.xlim(-max_range,max_range)
    plt.ylim(-max_range,max_range)
    plt.grid()
    plt.show()

    spec = psd.sel(freq_time=psd.freq_time>0.0)
    freq_power_spectra = spec.sum(dim = "freq_space")
    psd_freq_sum = float(2.0 * np.sum(freq_power_spectra))
    psd_freq_sum += float(np.sum(psd.sel(freq_time=psd.freq_time==0.0)))
    print(f"Parseval: Freq PSD spectrum (naive sum, y = sum(PSD)): 2 * sum(y) = {psd_freq_sum}")
    freq_power_spectra.plot()
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power Density [T^2 m/Hz]")
    plt.title("Naive sum of PSD over all k")
    plt.xlim((-0.5, max_range))
    plt.grid()
    plt.show()

def test_simulation_power_spectra(dataDirectory : Path, outputFolder : Path, window : str = None):
    
    print(f"Analyzing simulation in '{dataDirectory.name}'")

    # Read dataset
    print(f"Reading simulation data....")
    ds = xr.open_mfdataset(
        str(dataDirectory / "*.sdf"),
        data_vars='minimal', 
        coords='minimal', 
        compat='override', 
        preprocess=SDFPreprocess()
    )

    # Drop initial conditions because they may not represent a solution
    ds = ds.sel(time=ds.coords["time"]>ds.coords["time"][0])

    # Read input deck
    inputDeck = {}
    with open(str(dataDirectory / "input.deck")) as id:
        inputDeck = epydeck.loads(id.read())
    
    statsRoot = nc.Dataset(dataDirectory / "stats.nc", "a", format="NETCDF4")

    print(f"Calculating metadata....")
    ion_gyroperiod, alfven_velocity = epoch_utils.calculate_simulation_metadata(inputDeck, ds, statsRoot)

    print(f"Normalising data....")
    ds : xr.Dataset = epoch_utils.normalise_data(ds, ion_gyroperiod, alfven_velocity)
    ds = ds.load()

    bz = ds["Magnetic_Field_Bz"]

    # FFT method
    fft_spec : xr.DataArray = xrft.xrft.fft(bz, true_amplitude=True, true_phase=True, window=None)
    fft_spec = fft_spec.rename(freq_time="frequency", freq_x_space="wavenumber")
    fft_spec = fft_spec.where(fft_spec.wavenumber!=0.0, None)
    np.log10(np.abs(fft_spec)).plot(cmap="plasma")
    plt.title("Basic FFT")
    plt.xlim(-100,100)
    # plt.ylim(-100,100)
    plt.grid()
    plt.savefig(outputFolder / f"{dataDirectory.name}_fft_spectrum.png")
    plt.close("all")

    # Spacings
    a = []
    b = []
    assert len(bz.coords['time']) == len(fft_spec.coords['frequency'])
    for i in range(len(bz.coords['time'])-1):
        a.append(bz.coords['time'][i+1] - bz.coords['time'][i])
        b.append(fft_spec.coords['frequency'][i+1] - fft_spec.coords['frequency'][i])
    assert np.std(a) < 0.0001*np.mean(a)
    time_spacing = np.mean(a)
    assert np.std(b) < 0.0001*np.mean(b)
    frequency_spacing = np.mean(b)
    a = []
    b = []
    assert len(bz.coords['x_space']) == len(fft_spec.coords['wavenumber'])
    for i in range(len(bz.coords['x_space'])-1):
        a.append(bz.coords['x_space'][i+1] - bz.coords['x_space'][i])
        b.append(fft_spec.coords['wavenumber'][i+1] - fft_spec.coords['wavenumber'][i])
    assert np.std(a) < 0.0001*np.mean(a)
    cell_spacing = np.mean(a)
    assert np.std(b) < 0.0001*np.mean(b)
    wavenumber_spacing = np.mean(b)

    print(f"Parseval: Original Signal: sum(abs(signal - mean)**2) * dx * dt = {float((np.abs(bz - bz.mean())**2).sum() * time_spacing * cell_spacing)}")
    print(f"Parseval: Fourier Transform: sum(abs(fft)**2) * dk * dw = {float((np.abs(fft_spec)**2).sum() * frequency_spacing * wavenumber_spacing)}")

    fft_pos = fft_spec.sel(frequency=fft_spec.frequency>0.0)
    freq_naive_fft = fft_pos.sum(dim = "wavenumber")
    naive_freq_sum = float(2.0 * (np.abs(freq_naive_fft)**2).sum() * wavenumber_spacing * frequency_spacing)
    naive_freq_sum += float((np.abs(fft_spec.sel(frequency=0.0))**2).sum() * wavenumber_spacing * frequency_spacing) # Add 0-freq component
    print(f"Parseval: Freq spectrum (naive sum, y = sum(fft)): 2 * sum(abs(y)**2) * dk * dw = {naive_freq_sum}")
    np.abs(freq_naive_fft).plot()
    plt.title("Naive FFT spectrum over all k")
    plt.grid()
    plt.savefig(outputFolder / f"{dataDirectory.name}_fft_naive_sum.png")
    plt.close("all")

    freq_squared_fft = np.sqrt((np.abs(fft_pos)**2).sum(dim = "wavenumber"))
    squared_freq_sum = float(2.0 * (np.abs(freq_squared_fft)**2).sum() * wavenumber_spacing * frequency_spacing)
    squared_freq_sum += float((np.abs(fft_spec.sel(frequency=0.0))**2).sum() * wavenumber_spacing * frequency_spacing) # Add 0-freq component
    print(f"Parseval: Freq spectrum (better sum, y = sqrt(sum(abs(fft)**2))) 2 * sum(abs(y)**2) * dk * dw =  {squared_freq_sum}")
    np.abs(freq_squared_fft).plot()
    plt.title("Energy conserved FFT spectrum over all k")
    plt.grid()
    plt.savefig(outputFolder / f"{dataDirectory.name}_fft_better_sum.png")
    plt.close("all")

    # Intentionally remove DC offset -- debugging
    # bz = bz - bz.mean()

    # Power spectrum
    ps = xrft.power_spectrum(bz, scaling = "spectrum", window = window, window_correction=(window is not None), detrend = "constant")
    ps = ps.where(ps.freq_x_space!=0.0, None)
    # print(f"Parseval: PS: sum(PS) / (dk * dw) = {float(np.sum(ps) / (wavenumber_spacing * frequency_spacing))}")
    print(f"Parseval: PS: sum(PS) = {float(np.sum(ps))}")
    np.abs(ps).plot()
    plt.title("Power spectrum")
    plt.xlim(-100,100)
    plt.grid()
    plt.savefig(outputFolder / f"{dataDirectory.name}_ps.png")
    plt.close("all")

    ps_pos = ps.sel(freq_time=ps.freq_time>0.0)
    freq_ps = ps_pos.sum(dim = "freq_x_space")
    # ps_freq_sum = float(2.0 * np.sum(freq_ps) / (wavenumber_spacing * frequency_spacing))
    # ps_freq_sum += float(np.sum(ps.sel(freq_time=0.0)) / (wavenumber_spacing * frequency_spacing))
    ps_freq_sum = float(np.sum(freq_ps) / (wavenumber_spacing * frequency_spacing))
    # ps_freq_sum = float(2.0 * np.sum(freq_ps))
    # ps_freq_sum += float(np.sum(ps.sel(freq_time=0.0)))
    print(f"Parseval: Freq PS (naive sum, y = sum(PS)): sum(y)  / (dk * dw) = {ps_freq_sum}")
    # print(f"Parseval: Freq PS (naive sum, y = sum(PS)): 2 * sum(y) = {ps_freq_sum}")
    # print(f"Parseval: Freq PS (naive sum, y = sum(PS)): 2 * sum(y) / (dk * dw) = {ps_freq_sum}")
    np.abs(freq_ps).plot()
    plt.title("Naive sum of power spectrum over all k")
    plt.grid()
    plt.savefig(outputFolder / f"{dataDirectory.name}_ps_sum.png")
    plt.close("all")

    # Power spectral density
    psd = xrft.power_spectrum(bz, scaling = "density", window = window, window_correction=(window is not None), detrend = "constant")
    psd = psd.where(psd.freq_x_space!=0.0, None)
    print(f"Parseval: PSD: sum(PSD) = {float(np.sum(psd))}")
    np.abs(psd).plot()
    plt.title("Power spectral density")
    plt.grid()
    plt.xlim(-100,100)
    plt.savefig(outputFolder / f"{dataDirectory.name}_psd.png")
    plt.close("all")

    psd_pos = psd.sel(freq_time=psd.freq_time>0.0)
    freq_psd = psd_pos.sum(dim = "freq_x_space")
    psd_freq_sum = float(2.0 * np.sum(freq_psd))
    # psd_freq_sum += float(np.sum(psd.sel(freq_time=0.0)))
    print(f"Parseval: Freq PSD spectrum (naive sum, y = sum(PSD)): 2 * sum(y) = {psd_freq_sum}")
    np.abs(freq_psd).plot()
    plt.title("Naive sum of PSD over all k")
    plt.grid()
    plt.savefig(outputFolder / f"{dataDirectory.name}_psd_sum.png")
    plt.close("all")

    # one_ds = ds.sel(x_space = 2.1, method="nearest")
    # two_ds = ds.sel(x_space = 3.75, method="nearest")
    # three_ds = ds.sel(x_space = 7.5, method="nearest")

    # # Take FFT
    # one_ds = one_ds.load()
    # spec_one : xr.DataArray = xrft.xrft.fft(one_ds["Magnetic_Field_Bz"], true_amplitude=True, true_phase=True, window=None)
    # spec_one = spec_one.rename(freq_time="frequency")
    # spec_one = spec_one.sel(frequency=spec_one.frequency>0.0)
    # np.abs(spec_one).plot(label = "1.0")

    # two_ds = two_ds.load()
    # spec_two : xr.DataArray = xrft.xrft.fft(two_ds["Magnetic_Field_Bz"], true_amplitude=True, true_phase=True, window=None)
    # spec_two = spec_two.rename(freq_time="frequency")
    # spec_two = spec_two.sel(frequency=spec_two.frequency>0.0)
    # np.abs(spec_two).plot(label = "3.75")

    # three_ds = three_ds.load()
    # spec_three : xr.DataArray = xrft.xrft.fft(three_ds["Magnetic_Field_Bz"], true_amplitude=True, true_phase=True, window=None)
    # spec_three = spec_three.rename(freq_time="frequency")
    # spec_three = spec_three.sel(frequency=spec_three.frequency>0.0)
    # np.abs(spec_three).plot(label = "7.5")
    # plt.yscale("log")
    # plt.show() 

if __name__ == "__main__":

    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--dataDir",
        action="store",
        help="Directory containing sdf files of simulation output.",
        required = False,
        type=Path
    )
    parser.add_argument(
        "--outputDir",
        action="store",
        help="Filepath to which to write plots.",
        required = False,
        type=Path
    )
    parser.add_argument(
        "--window",
        action="store",
        help="Window to apply to the power spectrum.",
        required = False,
        type=str
    )

    args = parser.parse_args()

    #cold_plasma_dispersion_relation()
    #appleton_hartree()
    #trigonometry()
    #verdon_et_al()
    # analyse_bad_predictions(
    #     Path("/home/era536/Documents/Epoch/Data/2026_analysis/tsr_aeon_viking/LOOCV/aeon_combined_loocv_csvPredictions_predictions.csv"),
    #     Path("/home/era536/Documents/Epoch/Data/2026_analysis/original_input_decks/run_0_100/")
    # )

    analyse_real_frequencies(Path("/home/era536/Documents/Epoch/Data/2026_analysis/combined_spectra_4/data/"))
    # analyse_x_gyrofreqs(Path("/home/era536/Documents/Epoch/Data/2026_analysis/combined_spectra_2/data/"), num_gyro_freqs=11.0, outputFolder=Path("/home/era536/Documents/Epoch/Data/2026_analysis/combined_spectra_reduced/"))

    # fourier_power(Path("/home/era536/Documents/Epoch/Data/batch_testing_4/total_run_43/"))

    # test_simulation_power_spectra(args.dataDir, args.outputDir, args.window)