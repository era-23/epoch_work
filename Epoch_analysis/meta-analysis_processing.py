import argparse
import glob
import xarray as xr
import netCDF4 as nc
import numpy as np
import epoch_utils as utils
from pathlib import Path

def collate_growth_rate_data(dataDir : Path, outputToOriginalFile : bool = True):
    
    dataFiles = glob.glob(str(dataDir / "*.nc"))
    
    for simulation in dataFiles:

        data = xr.open_datatree(
            simulation,
            engine="netcdf4"
        )

        growthRateGroups = [g for g in data.groups if "growthRates" in g]
        fields = {}
        for groupPath in growthRateGroups:
            fieldData = data[groupPath]
            keyGrowthRates = {}
            keyGrowthRateIndices = {}
            keyGrowthRateIndices["max"] = np.argmax(fieldData.growthRate.data) 
            keyGrowthRateIndices["maxInHighPeakPowerK"] = np.argmax(fieldData.peakPower.data)
            keyGrowthRateIndices["maxInHighTotalPowerK"] = np.argmax(fieldData.totalPower.data)
            for metric, i in keyGrowthRateIndices.items():
                keyGrowthRates[metric] = utils.LinearGrowthRate(
                    gamma = float(fieldData.growthRate[i]),
                    timeMidpoint=float(fieldData.time[i]),
                    yIntercept=float(fieldData.yIntercept[i]),
                    residual=float(fieldData.residual[i]),
                    wavenumber=float(fieldData.wavenumber[i]),
                    peakPower=float(fieldData.peakPower[i]),
                    totalPower=float(fieldData.totalPower[i])
                )
            fields[groupPath] = keyGrowthRates
        data.close()

        ncData = nc.Dataset(simulation, "a", format="NETCDF4")
        for field, stats in fields.items():
            for metric, metricData in stats.items():
                metricGroup = ncData[field].createGroup(metric)
                metricGroup.growthRate = float(metricData.gamma)
                metricGroup.time=float(metricData.timeMidpoint)
                metricGroup.yIntercept=float(metricData.yIntercept)
                metricGroup.residual=float(metricData.residual)
                metricGroup.wavenumber=float(metricData.wavenumber)
                metricGroup.peakPower=float(metricData.peakPower)
                metricGroup.totalPower=float(metricData.totalPower)
        ncData.close()

        print(f"Done {simulation}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--dir",
        action="store",
        help="Directory containing simulation metadata, plots and calculated data, e.g. growth rates.",
        required = True,
        type=Path
    )
    parser.add_argument(
        "--outputToOriginalFile",
        action="store_true",
        help="Write output to each original netCDF data file. Else collate in csv",
        required = False
    )
    parser.add_argument(
        "--outputFile",
        action="store",
        help="Optional CSV file to which to write output.",
        required = False,
        type=str
    )

    args = parser.parse_args()

    collate_growth_rate_data(args.dir, args.outputToOriginalFile)