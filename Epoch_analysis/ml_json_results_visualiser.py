import argparse
from pathlib import Path
import json

from matplotlib import pyplot as plt
import numpy as np

SMALL_SIZE = 10
MEDIUM_SIZE = 18
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def plotScatter(resultsDict : dict):
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
            barVals[res["algorithm"]].append(np.round(res["cvR2_mean"], 3))
            barErrs[res["algorithm"]].append(res["cvR2_stderr"])
    
    x = np.arange(len(xLabels))

    fig, ax = plt.subplots(figsize=(12, 15))

    for algorithm, value in barVals.items():
        ax.errorbar(x, value, fmt="o", label=algorithm, yerr=barErrs[algorithm], elinewidth=2.0, capsize=10.0, capthick=2.0)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(r'Mean $R^2$')
    ax.set_xlabel('Output')
    ax.set_title('9-fold CV results')
    ax.set_xticks(x, xLabels)
    ax.legend(loc='upper left', ncols= 2 if len(resultsDict["algorithms"]) > 5 else 1)
    ax.set_ylim(top=1.0)
    ax.grid()
    ax.axhline(0.0, color="black", lw=0.5)

    plt.show()

def plotBar(resultsDict : dict, metric : str = "cvR2"): 

    patterns = [ "/" , ".", "\\" , "|" , "-" , "+" , "x",  "o", "O", "*" ]
    
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
            barVals[res["algorithm"]].append(np.round(res["cvR2_mean"], 3))
            barErrs[res["algorithm"]].append(res["cvR2_stderr"])
    
    x = np.arange(len(xLabels))
    
    width = 1.0/(len(resultsDict["algorithms"]) + 1)  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(figsize=(15, 15))

    for algorithm, value in barVals.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, value, width, label=algorithm, yerr=barErrs[algorithm], edgecolor='black')
        #ax.errorbar(x + offset, value, yerr=barErrs[algorithm], fmt=",", color = "k")
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(r'Mean $R^2$')
    ax.set_xlabel('Output')
    ax.set_title('9-fold CV results')
    ax.set_xticks(x + width, xLabels)
    ax.legend(loc='upper left', ncols= 2 if len(resultsDict["algorithms"]) > 5 else 1)
    ax.set_ylim(top=1.0)
    ax.axhline(0.0, color="black", lw=0.5)

    plt.show()


def plotResults(resultsFile : Path):
    with open(resultsFile, "r") as f:
        parser = json.load(f)
        plotBar(parser)
        plotScatter(parser)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--file",
        action="store",
        help="JSON file of results output.",
        required = True,
        type=Path
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot a bar chart of results.",
        required = False
    )

    args = parser.parse_args()

    if args.plot:
        plotResults(args.file)