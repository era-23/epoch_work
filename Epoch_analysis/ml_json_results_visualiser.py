import argparse
import glob
from pathlib import Path
import json

from matplotlib import pyplot as plt
import numpy as np

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

def plotBar(resultsDict : dict, metric : str = "cvR2"): 

    patterns = [ "/" , ".", "\\" , "|" , "-" , "+" , "x",  "o", "O", "*" ]
    drop_algorithms = ["aeon.KNeighborsTimeSeriesRegressor", "aeon.RDSTRegressor", "aeon.DrCIFRegressor", "aeon.Catch22Regressor"]
    
    # results
    results = resultsDict["results"]
    results = [r for r in results if r["algorithm"] not in drop_algorithms]
    
    # x-labels: outputs
    xLabels = resultsDict["outputFields"]

    # bar values in output order, grouped by algorithm
    algorithms = [a for a in resultsDict["algorithms"] if a not in drop_algorithms]
    barVals = {algorithm : [] for algorithm in algorithms}
    barErrs = {algorithm : [] for algorithm in algorithms}
    for field in xLabels:
        fieldResults = [r for r in results if r["output"] == field]
        for res in fieldResults:
            barVals[res["algorithm"]].append(np.round(res[f"{metric}_mean"], 3))
            barErrs[res["algorithm"]].append(res[f"{metric}_stderr"])
    
    x = np.arange(len(xLabels))
    
    width = 1.0/(len(algorithms) + 1)  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(figsize=(15, 15))

    for algorithm, value in barVals.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, value, width, label=algorithm, yerr=barErrs[algorithm], edgecolor='black')
        #ax.errorbar(x + offset, value, yerr=barErrs[algorithm], fmt=",", color = "k")
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    if metric == "cvR2":
        ax.set_ylabel(r'Mean $R^2$')
    elif metric == "cvRMSE":
        ax.set_ylabel('Mean RMSE')
    if resultsDict["cvStrategy"] == "RepeatedKFolds":
        ax.set_title(f'{resultsDict["cvFolds"]}-fold CV results ({resultsDict["cvRepeats"]} repeats)')
    ax.set_xlabel('Output')
    xLabels = [epoch_utils.fieldNameToText(lab) for lab in xLabels]
    ax.set_xticks(x + (0.5 * (len(algorithms) -1) * width), xLabels)
    if len(resultsDict["algorithms"]) > 5:
        ax.legend(loc='center', ncols = 2, bbox_to_anchor = (0.5, -0.3))
    else:
        ax.legend(loc='upper left' if metric == "cvR2" else 'lower left', ncols = 1)
    ax.set_ylim(top= 1.0 if metric == "cvR2" else np.round(np.max([v for v in barVals.values()]) + 0.2, 1))
    ax.axhline(0.0, color="black", lw=0.5)
    ax.grid(axis="y")
    fig.tight_layout()

    plt.show()

def plotResults(resultsFile : Path, metric : str = "cvR2"):
    with open(resultsFile, "r") as f:
        parser = json.load(f)
        plotBar(parser, metric)
        # plotScatter(parser, metric)

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
        "--metric",
        action="store",
        help="Scoring metric: \'cvR2\' or \'cvRMSE\'.",
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

    args = parser.parse_args()

    if args.folder is not None:
        plotAeResultsByPcaComponents(args.folder, args.filePattern)
    else:
        plotResults(args.file, args.metric)