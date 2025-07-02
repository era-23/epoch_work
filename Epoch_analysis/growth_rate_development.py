import argparse
import glob
import os
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path

def correlate_growth_rate_with_frequency(
        dir : Path, 
        outputDir : Path, 
        fields : list,
        displayPlots : bool = False,
    ):
    
    data_files = glob.glob(str(dir / "*.nc"))

    for simulation in data_files:

        data = xr.open_datatree(
            simulation,
            engine="netcdf4"
        )

        sim_id = Path(simulation).name.split("/")[-1]

        for field in fields:
            topDir = outputDir / field
            saveDir = topDir / "growth_rate_analysis"
            if os.path.exists(topDir):
                if not os.path.exists(saveDir):
                    os.mkdir(saveDir)
            else:
                os.mkdir(topDir)
                os.mkdir(saveDir)

            gammaData = data["/" + field + "/growthRates/positive"].to_dataset()

            frequencies = xr.DataArray(gammaData.variables["frequencyOfMaxPowerInK"], dims=["wavenumber"], coords=[gammaData.variables["wavenumber"]])
            growthRates = xr.DataArray(gammaData.variables["growthRate"], dims=["wavenumber"], coords=[gammaData.variables["wavenumber"]])

            assert frequencies.size == growthRates.size
            assert np.allclose(frequencies.coords["wavenumber"], growthRates.coords["wavenumber"])
            
            plt.scatter(frequencies.data, growthRates.data, marker='x')
            plt.title(f"{sim_id} {field}: Best fit growth rate by frequency (raw data)")
            plt.xlabel(r"Frequency/$\omega_{cp}$")
            plt.ylabel(r"Growth rate/$\omega_{cp}$")
            plt.tight_layout()
            #plt.savefig(saveDir / f"{sim_id}_{field}_gamma_by_omega_raw.png")
            if displayPlots:
                plt.show()
            plt.close("all")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--dir",
        action="store",
        help="Directory containing simulation data analysis (netCDF) files for evaluation.",
        required = True,
        type=Path
    )
    parser.add_argument(
        "--outputDir",
        action="store",
        help="Directory in which to save plots.",
        required = True,
        type=Path
    )
    parser.add_argument(
        "--fields",
        action="store",
        help="EPOCH output fields to use for growth rate analysis, e.g. Magnetic_Field_Bz, Electric_Field_Ex.",
        required = True,
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--displayPlots",
        action="store_true",
        help="Display plots.",
        required = False
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug.",
        required = False
    )

    args = parser.parse_args()

    correlate_growth_rate_with_frequency(args.dir, args.outputDir, args.fields, args.displayPlots)