import glob
from pathlib import Path
import numpy as np
import argparse
import netCDF4 as nc
import xarray as xr
import epoch_utils as eu

def calculate_iciness(combined_directory : Path):

    combined_statsFiles = glob.glob(str(combined_directory / "_combined_stats.nc"))

    for simFile in combined_statsFiles:
        data_xr = xr.open_datatree(simFile, engine="netcdf4")

def combine_spectra(dataDirectory : Path, outputDirectory : Path):

    # Get folders for each angle
    angle_folders = glob.glob(str(dataDirectory / "9[0-9]/data/"))

    assert len(angle_folders) > 0

    # Get first angle
    first_angle_folder = Path(angle_folders[0])

    data_files = glob.glob(str(first_angle_folder / "*.nc"))
    individual_data_filenames = [Path(n).name for n in data_files]

    # Each simulation (at one angle)
    for filename in individual_data_filenames:

        sim_id = filename.removesuffix("_stats.nc")
        print(f"Processing {sim_id}...")
        
        # sim_id = filename.removeprefix("run_").removesuffix("_stats.nc")

        all_sim_angle_datapaths = [Path(fp) / filename for fp in angle_folders]

        num_angles = len(all_sim_angle_datapaths)
        print(f"Composing spectrum from {num_angles} B0 angles...")

        # Create single NC file for summary of combined spectra
        # Get existing spectra as xarray
        data_xr = xr.open_datatree(all_sim_angle_datapaths[0], engine="netcdf4")
        # New data without growth rates as they no longer make sense
        # Energy may make sense but needs further thought
        newData_xr = xr.DataTree(dataset=data_xr.to_dataset(inherit=False), children={
            "Energy" : data_xr.children["Energy"],
            "Magnetic_Field_Bz" : xr.DataTree(dataset=data_xr.children["Magnetic_Field_Bz"].to_dataset(), children={
                "power" : data_xr.children["Magnetic_Field_Bz"].children["power"]
            }),
            "Electric_Field_Ex" : xr.DataTree(dataset=data_xr.children["Electric_Field_Ex"].to_dataset(), children={
                "power" : data_xr.children["Electric_Field_Ex"].children["power"]
            }),
            "Electric_Field_Ey" : xr.DataTree(dataset=data_xr.children["Electric_Field_Ey"].to_dataset(), children={
                "power" : data_xr.children["Electric_Field_Ey"].children["power"]
            }),
        }, name = "combined_simulation_data")
        data_xr.close()
        newData_xr.attrs.pop("B0angle") # Now multivalued
        
        # Remove frequencies of max power as they're no longer correct
        # Broken, figure out how to do this
        # newData_xr["Magnetic_Field_Bz/power"].drop_vars("frequencyOfMaxPowerByK")
        # newData_xr["Electric_Field_Ex/power"].drop_vars("frequencyOfMaxPowerByK")
        # newData_xr["Electric_Field_Ey/power"].drop_vars("frequencyOfMaxPowerByK")

        # Set spectral variables to 0
        newData_xr["Magnetic_Field_Bz/power/powerByFrequency"] *= 0.0
        newData_xr["Magnetic_Field_Bz/power/powerByWavenumber"] *= 0.0
        newData_xr["Electric_Field_Ex/power/powerByFrequency"] *= 0.0
        newData_xr["Electric_Field_Ex/power/powerByWavenumber"] *= 0.0
        newData_xr["Electric_Field_Ey/power/powerByFrequency"] *= 0.0
        newData_xr["Electric_Field_Ey/power/powerByWavenumber"] *= 0.0
        newData_xr["Energy/backgroundIonMeanEnergyDensity"] *= 0.0
        newData_xr["Energy/electronMeanEnergyDensity"] *= 0.0
        newData_xr["Energy/electricFieldMeanEnergyDensity"] *= 0.0
        newData_xr["Energy/magneticFieldMeanEnergyDensity"] *= 0.0
        newData_xr["Energy/fastIonMeanEnergyDensity"] *= 0.0
        
        simulation_total_original_power = 0.0

        for angle_sim_path in all_sim_angle_datapaths:

            print(f"Processing {sim_id} -- {angle_sim_path.absolute()}...")

            # Get this angle simulation's data
            angle_data_xr = xr.open_datatree(angle_sim_path, engine="netcdf4")

            # Ensure key parameters are equal to those already in the new dataset
            assert angle_data_xr.attrs["B0strength"] == newData_xr.attrs["B0strength"]
            assert angle_data_xr.attrs["backgroundDensity"] == newData_xr.attrs["backgroundDensity"]
            assert angle_data_xr.attrs["beamFraction"] == newData_xr.attrs["beamFraction"]
            assert angle_data_xr.attrs["pitch"] == newData_xr.attrs["pitch"]

            # angle_data_xr.to_netcdf("testToBeAdded.nc")

            # Check coordinates are equal
            assert (angle_data_xr["Magnetic_Field_Bz/power"].coords["wavenumber"] == newData_xr["Magnetic_Field_Bz/power"].coords["wavenumber"]).all()
            assert (angle_data_xr["Magnetic_Field_Bz/power"].coords["frequency"] == newData_xr["Magnetic_Field_Bz/power"].coords["frequency"]).all()
            assert (angle_data_xr["Electric_Field_Ex/power"].coords["wavenumber"] == newData_xr["Electric_Field_Ex/power"].coords["wavenumber"]).all()
            assert (angle_data_xr["Electric_Field_Ex/power"].coords["frequency"] == newData_xr["Electric_Field_Ex/power"].coords["frequency"]).all()
            assert (angle_data_xr["Electric_Field_Ey/power"].coords["wavenumber"] == newData_xr["Electric_Field_Ey/power"].coords["wavenumber"]).all()
            assert (angle_data_xr["Electric_Field_Ey/power"].coords["frequency"] == newData_xr["Electric_Field_Ey/power"].coords["frequency"]).all()
            assert (angle_data_xr["Energy/time"] == newData_xr["Energy/time"]).all()

            # Add spectra
            # Electric and magnetic fields
            newData_xr["Magnetic_Field_Bz/power/powerByWavenumber"] += angle_data_xr["Magnetic_Field_Bz/power/powerByWavenumber"]
            newData_xr["Magnetic_Field_Bz/power/powerByFrequency"] += angle_data_xr["Magnetic_Field_Bz/power/powerByFrequency"]
            newData_xr["Electric_Field_Ex/power/powerByWavenumber"] += angle_data_xr["Electric_Field_Ex/power/powerByWavenumber"]
            newData_xr["Electric_Field_Ex/power/powerByFrequency"] += angle_data_xr["Electric_Field_Ex/power/powerByFrequency"]
            newData_xr["Electric_Field_Ey/power/powerByWavenumber"] += angle_data_xr["Electric_Field_Ey/power/powerByWavenumber"]
            newData_xr["Electric_Field_Ey/power/powerByFrequency"] += angle_data_xr["Electric_Field_Ey/power/powerByFrequency"]

            # Energy
            newData_xr["Energy/backgroundIonMeanEnergyDensity"] += angle_data_xr["Energy/backgroundIonMeanEnergyDensity"]
            newData_xr["Energy/electronMeanEnergyDensity"] += angle_data_xr["Energy/electronMeanEnergyDensity"]
            newData_xr["Energy/electricFieldMeanEnergyDensity"] += angle_data_xr["Energy/electricFieldMeanEnergyDensity"]
            newData_xr["Energy/magneticFieldMeanEnergyDensity"] += angle_data_xr["Energy/magneticFieldMeanEnergyDensity"]
            newData_xr["Energy/fastIonMeanEnergyDensity"] += angle_data_xr["Energy/fastIonMeanEnergyDensity"]

            simulation_total_original_power += angle_data_xr["Magnetic_Field_Bz/power/powerByWavenumber"].sum()
            simulation_total_original_power += angle_data_xr["Magnetic_Field_Bz/power/powerByFrequency"].sum()
            simulation_total_original_power += angle_data_xr["Electric_Field_Ex/power/powerByWavenumber"].sum()
            simulation_total_original_power += angle_data_xr["Electric_Field_Ex/power/powerByFrequency"].sum()
            simulation_total_original_power += angle_data_xr["Electric_Field_Ey/power/powerByWavenumber"].sum()
            simulation_total_original_power += angle_data_xr["Electric_Field_Ey/power/powerByFrequency"].sum()
            simulation_total_original_power += angle_data_xr["Energy/backgroundIonMeanEnergyDensity"].sum()
            simulation_total_original_power += angle_data_xr["Energy/electronMeanEnergyDensity"].sum()
            simulation_total_original_power += angle_data_xr["Energy/electricFieldMeanEnergyDensity"].sum()
            simulation_total_original_power += angle_data_xr["Energy/magneticFieldMeanEnergyDensity"].sum()
            simulation_total_original_power += angle_data_xr["Energy/fastIonMeanEnergyDensity"].sum()

        # Power check
        simulation_total_new_power = newData_xr["Magnetic_Field_Bz/power/powerByWavenumber"].sum()
        simulation_total_new_power += newData_xr["Magnetic_Field_Bz/power/powerByFrequency"].sum()
        simulation_total_new_power += newData_xr["Electric_Field_Ex/power/powerByWavenumber"].sum()
        simulation_total_new_power += newData_xr["Electric_Field_Ex/power/powerByFrequency"].sum()
        simulation_total_new_power += newData_xr["Electric_Field_Ey/power/powerByWavenumber"].sum()
        simulation_total_new_power += newData_xr["Electric_Field_Ey/power/powerByFrequency"].sum()
        simulation_total_new_power += newData_xr["Energy/backgroundIonMeanEnergyDensity"].sum()
        simulation_total_new_power += newData_xr["Energy/electronMeanEnergyDensity"].sum()
        simulation_total_new_power += newData_xr["Energy/electricFieldMeanEnergyDensity"].sum()
        simulation_total_new_power += newData_xr["Energy/magneticFieldMeanEnergyDensity"].sum()
        simulation_total_new_power += newData_xr["Energy/fastIonMeanEnergyDensity"].sum()
        assert np.allclose(simulation_total_original_power.data, simulation_total_new_power.data)

        # Make energy an average
        newData_xr["Energy/backgroundIonMeanEnergyDensity"] /= num_angles
        newData_xr["Energy/electronMeanEnergyDensity"] /= num_angles
        newData_xr["Energy/electricFieldMeanEnergyDensity"] /= num_angles
        newData_xr["Energy/magneticFieldMeanEnergyDensity"] /= num_angles
        newData_xr["Energy/fastIonMeanEnergyDensity"] /= num_angles

        # Fix energy stats
        for k, _ in newData_xr["Energy"].attrs.items():
            newData_xr["Energy"].attrs[k] = 0.0
        newData_xr["Energy"].attrs["fastIonEnergyDensity_delta"] = newData_xr["Energy/fastIonMeanEnergyDensity"][-1].data - newData_xr["Energy/fastIonMeanEnergyDensity"][0].data
        newData_xr["Energy"].attrs["backgroundIonEnergyDensity_delta"] = newData_xr["Energy/backgroundIonMeanEnergyDensity"][-1].data - newData_xr["Energy/backgroundIonMeanEnergyDensity"][0].data
        newData_xr["Energy"].attrs["electronEnergyDensity_delta"] = newData_xr["Energy/electronMeanEnergyDensity"][-1].data - newData_xr["Energy/electronMeanEnergyDensity"][0].data
        newData_xr["Energy"].attrs["electricFieldEnergyDensity_delta"] = newData_xr["Energy/electricFieldMeanEnergyDensity"][-1].data - newData_xr["Energy/electricFieldMeanEnergyDensity"][0].data
        newData_xr["Energy"].attrs["magneticFieldEnergyDensity_delta"] = newData_xr["Energy/magneticFieldMeanEnergyDensity"][-1].data - newData_xr["Energy/magneticFieldMeanEnergyDensity"][0].data
        newData_xr["Energy"].attrs["totalEnergyDensity_delta"] = newData_xr["Energy"].attrs["fastIonEnergyDensity_delta"] + newData_xr["Energy"].attrs["backgroundIonEnergyDensity_delta"] + newData_xr["Energy"].attrs["electronEnergyDensity_delta"] + newData_xr["Energy"].attrs["electricFieldEnergyDensity_delta"] + newData_xr["Energy"].attrs["magneticFieldEnergyDensity_delta"]
        newData_xr["Energy"].attrs["fastIonToBackgroundTransfer"] = newData_xr["Energy"].attrs["backgroundIonEnergyDensity_delta"] - newData_xr["Energy"].attrs["fastIonEnergyDensity_delta"]
        newData_xr["Energy"].attrs["fastIonToBackgroundTransfer_percentage"] = 100.0 * (newData_xr["Energy"].attrs["fastIonToBackgroundTransfer"] / newData_xr["Energy/fastIonMeanEnergyDensity"][0].data)

        # Write new file
        outFname = outputDirectory / f"{sim_id}_combined_stats.nc"
        newData_xr.to_netcdf(outFname)
        newData_xr.close()

        print(f"{filename} complete and written to {outFname.absolute()}.")

if __name__ == "__main__":
    # Run python setup.py -h for list of possible arguments
    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--dir",
        action="store",
        help="Directory containing analysis of all simulations. Expects folders of angles, each with data and plots folders.",
        required = True,
        type=Path
    )
    parser.add_argument(
        "--outputDir",
        action="store",
        help="Output directory.",
        required = False,
        type=Path
    )
    parser.add_argument(
        "--combineSpectra",
        action="store_true",
        help="Combine Ex, Ey and Bz spectra.",
        required = False
    )
    parser.add_argument(
        "--doIciness",
        action="store_true",
        help="Calculate ICEiness characteristics for combined spectra.",
        required = False
    )
    
    args = parser.parse_args()

    if args.combineSpectra:
        combine_spectra(args.dir, args.outputDir)
        combined_dir = args.outputDir
    else:
        combined_dir = args.dir

    if args.doIciness:
        calculate_iciness(combined_dir)

