import argparse
import glob
from pathlib import Path
import json
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from latextable import texttable
import ml_utils
import epoch_utils

SMALL_SIZE = 10
MEDIUM_SIZE = 20
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def plotAeResultsByPcaComponents(folder : Path, resultsFilePattern : str, field : str = "r2_test"):
    ##### Get input data
    results_files = glob.glob(str(folder / resultsFilePattern))

    # Dumb hardcode
    totalFolds = 40

    xLabels = set()
    initialised = False
    for file in results_files:
        with open(file, "r") as f:
            json_results = json.load(f)
            pcaNum = json_results["pcaComponents"]['0'] if json_results["pca"]['0'] else 0
            xLabels.add(pcaNum)
            if not initialised:
                results_dict = {model : {} for model in json_results["model_name"].values()}
                std_dict = {model : {} for model in json_results["model_name"].values()}
                initialised = True
            for id, m in json_results["model_name"].items():
                # results_dict[m][pcaNum] = np.max([json_results[field][id], -1.0])
                results_dict[m][pcaNum] = json_results[field][id]
                std_dict[m][pcaNum] = (json_results[field + "_std"][id])/np.sqrt(totalFolds)

    for m, v in results_dict.items():
        results_dict[m] = dict(sorted(v.items()))
    for m, v in std_dict.items():
        std_dict[m] = dict(sorted(v.items()))
    
    xLabels = list(sorted(xLabels))
    x = np.arange(len(xLabels))
    fig, ax = plt.subplots(figsize=(12, 15))
    for model, value in results_dict.items():
        ax.errorbar(x, value.values(), fmt="o", label=model, yerr=std_dict[model].values(), elinewidth=2.0, capsize=10.0, capthick=2.0)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    if field == "r2_test":
        ax.set_ylabel(r'Mean $R^2$')
    elif field == "cvRMSE":
        ax.set_ylabel('Mean RMSE')
    ax.set_title(f'AutoEmulate CV results')
    ax.set_xlabel('PCA Components')
    ax.set_xticks(x, xLabels)
    ax.legend(loc='center', ncols = 2, bbox_to_anchor = (0.5, -0.3))
    ax.set_ylim(top= 1.0, bottom=-0.5)
    ax.set_xlim(left=-1)
    ax.grid()
    ax.axhline(0.0, color="black", lw=0.5)

    plt.show()

def plotScatter(resultsDict : dict, metric : str = "cvR2"):
    # results
    results = resultsDict["results"]
    
    # x-labels: outputs
    xLabels = resultsDict["outputFields"]

    # bar values in output order, grouped by algorithm
    barVals = {algorithm : [] for algorithm in resultsDict["algorithms"]}
    barErrs = {algorithm : [] for algorithm in resultsDict["algorithms"]}
    for field in xLabels:
        fieldResults = [r for r in results if r["output"] == field]
        for res in fieldResults:
            barVals[res["algorithm"]].append(np.round(res[f"{metric}_mean"], 3))
            barErrs[res["algorithm"]].append(res[f"{metric}_stderr"])
    
    x = np.arange(len(xLabels))

    fig, ax = plt.subplots(figsize=(12, 15))

    for algorithm, value in barVals.items():
        ax.errorbar(x, value, fmt="o", label=algorithm, yerr=barErrs[algorithm], elinewidth=2.0, capsize=10.0, capthick=2.0)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    if metric == "cvR2":
        ax.set_ylabel(r'Mean $R^2$')
    elif metric == "cvRMSE":
        ax.set_ylabel('Mean RMSE')
    if resultsDict["cvStrategy"] == "RepeatedKFolds":
        ax.set_title(f'{resultsDict["cvFolds"]}-fold CV results ({resultsDict["cvRepeats"]} repeats)')
    ax.set_xlabel('Output')
    xLabels = [ml_utils.fieldNameToText(lab) for lab in xLabels]
    ax.set_xticks(x, xLabels)
    ax.legend(loc='upper left', ncols= 2 if len(resultsDict["algorithms"]) > 5 else 1)
    ax.set_ylim(top= 1.0 if metric == "cvR2" else np.round(np.max([v for v in barVals.values()]) + 0.2, 1))
    ax.grid()
    ax.axhline(0.0, color="black", lw=0.5)

    plt.show()

def plotBar(resultsDict : dict, metric : str = "cvR2", errors : str = "rmseSE", dropAlgorithms : list = []): 

    patterns = [ "/" , ".", "\\" , "|" , "-" , "+" , "x",  "o", "O", "*" ]
    field_names = {"B0strength" : r"$B_0$", "backgroundDensity" : r"$n_e$", "pitch" : r"$\lambda$", "beamFraction" : "alpha concentration"}
    
    # results
    results = resultsDict["results"]
    results = [r for r in results if r["algorithm"] not in dropAlgorithms]
    
    # x-labels: outputs
    xLabels = resultsDict["outputFields"]

    # bar values in output order, grouped by algorithm
    algorithms = [a for a in resultsDict["algorithms"] if a not in dropAlgorithms]
    barVals = {algorithm : [] for algorithm in algorithms}
    barErrs = {algorithm : [] for algorithm in algorithms}
    for field in xLabels:
        fieldResults = [r for r in results if r["output"] == field]
        for res in fieldResults:
            barVals[res["algorithm"]].append(np.round(res[f"{metric}_mean"], 3))
            if errors == "rmseSD":
                barErrs[res["algorithm"]].append(np.sqrt(res[f"cvRMSE_var"]))
            elif errors == "rmseSE":
                barErrs[res["algorithm"]].append(res[f"cvRMSE_stderr"])
            else:
                barErrs[res["algorithm"]].append(res[f"{metric}_stderr"])
    
    x = np.arange(len(xLabels))
    
    width = 1.0/(len(algorithms) + 1)  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(figsize=(15, 15))

    for algorithm, value in barVals.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, value, width, label=algorithm, yerr=barErrs[algorithm], edgecolor='black')
        if list(barErrs.values())[0]:
            ax.errorbar(x + offset, value, yerr=barErrs[algorithm], fmt=",", color = "k")
        ax.bar_label(rects, padding=3, fontsize = 17, rotation=90)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    if metric == "cvR2":
        ax.set_ylabel(r'$R^2$')
    elif metric == "cvRMSE":
        ax.set_ylabel('Mean RMSE')
    if resultsDict["cvStrategy"] == "RepeatedKFolds":
        ax.set_title(f'{resultsDict["cvFolds"]}-fold CV results ({resultsDict["cvRepeats"]} repeats)')
    ax.set_xlabel('Output')
    xLabels = [field_names[lab] for lab in xLabels]
    ax.set_xticks(x + (0.5 * (len(algorithms) -1) * width), xLabels)
    if len(resultsDict["algorithms"]) > 5:
        ax.legend(loc='center', ncols = 2, bbox_to_anchor = (0.5, 1.12))
    else:
        ax.legend(loc='upper left' if metric == "cvR2" else 'lower left', ncols = 1)
    ax.set_ylim(top= 1.0 if metric == "cvR2" else np.round(np.max([v for v in barVals.values()]) + 0.2, 1))
    ax.axhline(0.0, color="black", lw=0.5)
    ax.grid(axis="y")
    ax.set_ylim(0.0, 1.1)
    plt.tight_layout()

    plt.show()

def plotResults(resultsFile : Path, metric : str = "cvR2", errors : str = "rmseSE", dropAlgorithms : list = []):
    with open(resultsFile, "r") as f:
        parser = json.load(f)
        plotBar(parser, metric, errors, dropAlgorithms)
        # plotScatter(parser, metric)

def latexTable(resultsFile : Path, experimentName : str):
    with open(resultsFile, "r") as f:
        parser = json.load(f)

        results = parser["results"]
        results_df = pd.DataFrame(results)
        fields = parser["outputFields"]
        algorithms = parser["algorithms"]
        metrics_displayNames = {
            "cvR2_mean" : "$R^2$", 
            "cvRMSE_mean" : "RMSE", 
            "cvRMSE_var" : "RMSE var", 
            "cvRMSE_stderr" : "RMSE S.E.", 
            "cvMAE_mean" : "MAE", 
            "cvMAPE_mean" : "MAPE"
        }
        num_metrics = len(metrics_displayNames.keys())
        
        # Metrics on top, algorithms down side
        table = texttable.Texttable()
        table.set_cols_align(["l"] + (["r"] * num_metrics))
        # Header
        table.add_row(["Algorithm"] + [v for v in metrics_displayNames.values()])
        for field in fields:
            table.add_row([field] + ([""] * num_metrics))
            for algo in algorithms:
                result = results_df[(results_df["output"] == field) & (results_df["algorithm"] == algo)]
                assert len(result) == 1
                table.add_row([algo] + [float(result[m].iloc[0]) for m in metrics_displayNames.keys()])
                break
            break

        print(table.draw())

def plotAccuracyByFrequency(resultsFile : Path):
    with open(resultsFile, "r") as f:
        parser = json.load(f)

        results = pd.DataFrame.from_dict(parser["results"])
        results : pd.DataFrame = results[results["algorithm"] == "aeon.HydraRegressor"].sort_values(by="frequencyBandwidth")

        if results["frequencyBandwidth"].max() < 80.0:
            unit = "gyros"
        else:
            unit = "hz"
            results["frequencyBandwidth"] = results["frequencyBandwidth"].apply(lambda x : x / 1e6)

        B0_results : pd.DataFrame = results[results["output"] == "B0strength"]
        pitch_results : pd.DataFrame = results[results["output"] == "pitch"]
        density_results : pd.DataFrame = results[results["output"] == "backgroundDensity"]
        beamFraction_results : pd.DataFrame = results[results["output"] == "beamFraction"]

        print(results[["algorithm", "output", "frequencyBandwidth", "cvR2_mean"]].sort_values(by="output"))

        cottrell_freq = 187.0
        cottrell_max_gyros = 48.7
        cottrell_min_gyros = 4.87

        plt.figure(figsize=(8,8))
        plt.errorbar(B0_results["frequencyBandwidth"], B0_results["cvR2_mean"], linewidth=3.0, yerr=B0_results["cvRMSE_stderr"], label=r"$B_0$ strength")
        plt.errorbar(pitch_results["frequencyBandwidth"], pitch_results["cvR2_mean"], linewidth=3.0, yerr=pitch_results["cvRMSE_stderr"], label=r"$\alpha$ velocity pitch")
        plt.errorbar(density_results["frequencyBandwidth"], density_results["cvR2_mean"], linewidth=3.0, yerr=density_results["cvRMSE_stderr"], label="electron density")
        plt.errorbar(beamFraction_results["frequencyBandwidth"], beamFraction_results["cvR2_mean"], linewidth=3.0, yerr=beamFraction_results["cvRMSE_stderr"], label=r"$\alpha$ concentration")
        plt.ylim(bottom=0.0, top=1.0)
        if unit == "gyros":
            plt.xlabel(r"Spectra frequency bandwidth [$\Omega_{c,\alpha}$]")
            plt.fill_betweenx(y = [0, 1], x1= cottrell_min_gyros, x2 = cottrell_max_gyros, color = "black", alpha = 0.15, label = "Cottrell '93 range")
        else:
            plt.xlabel("Spectra frequency bandwidth [MHz]")
            # plt.axvline(x = cottrell_freq, color = "black", linestyle="--", lw = 2.0, label="Cottrell '93 maximum")
        plt.ylabel(r"LOOCV accuracy [$R^2$]")
        plt.legend(loc="center right")
        plt.tight_layout()
        plt.show()

def print_results(resultsFile : Path):
    with open(resultsFile, "r") as f:
        parser = json.load(f)

        results : pd.DataFrame = pd.DataFrame.from_dict(parser["results"])
        algorithms = results["algorithm"].unique()
        fields = results["output"].unique()

        for field in fields:
            field_results = results[results["output"] == field].sort_values(by="cvR2_mean", ascending=False)
            print("--------------------- BY R^2 ---------------------")
            print(field_results.to_string(columns=["output", "algorithm", "cvR2_mean", "cvRMSE_mean", "cvMAE_mean"]))
            print("--------------------- BY RMSE ---------------------")
            field_results = field_results.sort_values(by="cvRMSE_mean")
            print(field_results.to_string(columns=["output", "algorithm", "cvR2_mean", "cvRMSE_mean", "cvMAE_mean"]))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--file",
        action="store",
        help="JSON file of results output.",
        required = False,
        type=Path
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot a bar chart of results.",
        required = False
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print results.",
        required = False
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Print a latex table of results.",
        required = False
    )
    parser.add_argument(
        "--accuracyByFrequency",
        action="store_true",
        help="Plot accuracy by freqency.",
        required = False
    )
    parser.add_argument(
        "--metric",
        action="store",
        help="Scoring metric: \'cvR2\' or \'cvRMSE\'.",
        required = False,
        type = str
    )
    parser.add_argument(
        "--experimentName",
        action="store",
        help="Name of the experiment.",
        required = False,
        type = str
    )
    parser.add_argument(
        "--errors",
        action="store",
        help="Error metric: \'rmseSD\'.",
        required = False,
        type = str
    )
    parser.add_argument(
        "--folder",
        action="store",
        help="Folder of multiple JSON results files.",
        required = False,
        type=Path
    )
    parser.add_argument(
        "--filePattern",
        action="store",
        help="File pattern to use for finding results files.",
        required = False,
        type=str
    )
    parser.add_argument(
        "--dropAlgorithms",
        action="store",
        help="Algorithms to ignore.",
        required = False,
        type=str,
        nargs="*"
    )

    args = parser.parse_args()

    if args.plot:
        if args.folder is not None:
            plotAeResultsByPcaComponents(args.folder, args.filePattern)
        dropAlgorithms = [] if args.dropAlgorithms is None else args.dropAlgorithms
        plotResults(args.file, args.metric, args.errors if args.errors is not None else "rmseSE", dropAlgorithms)
    if args.latex:
        latexTable(args.file, args.experimentName)
    if args.accuracyByFrequency:
        plotAccuracyByFrequency(args.file)
    if args.print:
        print_results(args.file)