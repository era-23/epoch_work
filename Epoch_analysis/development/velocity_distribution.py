from pathlib import Path
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy import constants
from scipy.stats import skew, skewtest, kurtosis, kurtosistest
from decimal import Decimal
from plasmapy.formulary import speeds as pps
from plasmapy.formulary import frequencies as ppf
from scipy.stats import linregress
import astropy.units as u
import xarray as xr
import numpy as np
import epydeck
import argparse

def gaussian(x, a, mean, sigma):
    return a * np.exp(-((x - mean)**2 / (2 * sigma**2)))

def plot_velo_dists(directory : Path, files : list, vPerp : bool = False, saveFolder : Path = None, doLog : bool = True):

    startTime_s = 0.0
    speciess = ["electron", "deuteron", "alpha"]
    dims = ["Vx", "Vy", "Vz"]
    colors = ["red", "orange", "green", "blue", "black"]

    # Read input deck
    inputDeck = {}
    with open(str(directory / "input.deck")) as id:
        inputDeck = epydeck.loads(id.read())
    temp = inputDeck['constant']['background_temp'] * u.K

    for species in speciess:
        vTh = pps.thermal_speed(temp, "e-" if species == "electron" else "p+")
        velocities = {dim : {} for dim in dims}
        for file in files:
            # Read dataset
            ds = xr.open_dataset(
                str(directory / f"{file}.sdf"),
                keep_particles = True
            )
            
            if file == "0000":
                startTime_s = float(ds.time)       
            
            time_s = float(ds.time) - startTime_s

            # Read input deck
            input = {}
            with open(str(directory / "input.deck")) as id:
                input = epydeck.loads(id.read())
            
            B0 = input['constant']['b0_strength']
            # alfven_velo = pps.Alfven_speed(B = B0 * u.T, density = input['constant']['background_density'] / u.m**3, ion = "p")
            fi_gyroperiod = (2.0 * np.pi * u.rad) / ppf.gyrofrequency(B = B0 * u.T, particle = "p")
            time_tci = time_s * u.s / fi_gyroperiod

            for dim in velocities.keys(): # Runs out of memory here with many particles on my laptop
                data = ds[f"Particles_{dim}_{species}"].load().data
                #print(f"Max  velo: {data.max()}")
                #print(f"Min  velo: {data.min()}")
                #print(f"Mean velo: {np.mean(data)}")
                #print(f"S.D. velo: {np.std(data)}")
                velocities[dim][time_tci] = data
                del(data)
        
        for dimension in velocities.keys():
            print(f"Processing {species}: {dimension}....")
            i = 0
            doPlot = False
            # For bin widths/range:
            dimData = velocities[dimension]
            upper_lim = max(vels.max() for vels in dimData.values())
            lower_lim = min(vels.min() for vels in dimData.values())
            fig, ax = plt.subplots(figsize=(10,8))
            for timepoint in dimData.keys():
                data = dimData[timepoint]
                print(f"timepoint: {timepoint:.3f}, total particles: {len(data)}")
                bin_num = round(np.sqrt(len(data)))
                # print(f"Default bin count: {bin_num}")
                bin_num = int(np.min([1000, bin_num]))
                # print(f"Using {bin_num} bins")
                mu = r'$\mu$'
                sigma = r'$\sigma$'
                vTh_unit = r'$v_{Th}$'
                vTh2_unit = r'$v_{Th}^2$'
                mean_vTh = (np.mean(data)*u.m/u.s) / vTh
                sigma_vTh2 = (np.var(data)*u.m**2/u.s**2) / vTh**2
                stats = f"{mu}: {mean_vTh:.3e}{vTh_unit}, {sigma}: {sigma_vTh2:.3e}{vTh2_unit}, Skew: {skew(data):.3e} (p-value: {skewtest(data).pvalue:.3f}),  Kurtosis: {kurtosis(data):.3e} (p-value: {kurtosistest(data).pvalue:.3f})"
                #print(f"{species}, {dimension}, t={timepoint:.2f} -- {stats}")
                if i == 0:
                    _, bins, _ = ax.hist(
                        data, 
                        bins = bin_num, 
                        range=[lower_lim, upper_lim], 
                        log=doLog, 
                        histtype='bar', 
                        edgecolor = colors[i], 
                        alpha = 1.0, 
                        ls = "solid", 
                        label = f"t = {timepoint:.2f} -- {stats}"
                    )
                else:
                    ax.hist(
                        data, 
                        bins = bins, 
                        log=doLog, 
                        histtype='bar', 
                        edgecolor = colors[i], 
                        alpha = 1.0, 
                        ls = "solid", 
                        label = f"t = {timepoint:.2f} -- {stats}"
                    )
                i += 1
                doPlot = True

            if doPlot:
                # Shrink current axis by 20%
                box = ax.get_position()
                ax.set_position([box.x0, box.y0 + (0.2 * box.height), box.width, (box.height*0.8)])
                ax.legend(bbox_to_anchor=(0.5, -0.1), loc = "upper center")
                ax.set_xlabel("Velocity [m/s]")
                ax.set_ylabel("Particle count")
                ax.set_title(f"{species}: {dimension}")
                if saveFolder is not None:
                    filepath = saveFolder / f"{species}_{dimension}_log.png" if doLog else saveFolder / f"{species}_{dimension}.png"  
                    fig.savefig(filepath, bbox_inches = "tight")
                else:
                    plt.show()
                plt.close()

def plot_velo_3d(directory : Path, files : list, saveFolder : Path = None):
    
    startTime_s = 0.0
    speciess = ["electron", "deuteron", "alpha"]
    dims = ["Vx", "Vy", "Vz"]

    # Read input deck
    inputDeck = {}
    with open(str(directory / "input.deck")) as id:
        inputDeck = epydeck.loads(id.read())
    temp = inputDeck['constant']['background_temp'] * u.K

    for species in speciess:
        vTh = pps.thermal_speed(temp, "e-" if species == "electron" else "p+")
        velocities = {dim : {} for dim in dims}
        for file in files:
            # Read dataset
            ds = xr.open_dataset(
                str(directory / f"{file}.sdf"),
                keep_particles = True
            )
            
            if file == "0000":
                startTime_s = float(ds.time)       
            
            time_s = float(ds.time) - startTime_s

            # Read input deck
            input = {}
            with open(str(directory / "input.deck")) as id:
                input = epydeck.loads(id.read())
            
            B0 = input['constant']['b0_strength']
            # alfven_velo = pps.Alfven_speed(B = B0 * u.T, density = input['constant']['background_density'] / u.m**3, ion = "p")
            fi_gyroperiod = (2.0 * np.pi * u.rad) / ppf.gyrofrequency(B = B0 * u.T, particle = "p")

            for dim in velocities.keys(): # Runs out of memory here with many particles on my laptop
                data = ds[f"Particles_{dim}_{species}"].load().data
                #print(f"Max  velo: {data.max()}")
                #print(f"Min  velo: {data.min()}")
                #print(f"Mean velo: {np.mean(data)}")
                #print(f"S.D. velo: {np.std(data)}")
                velocities[dim] = data
                del(data)
        
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(projection="3d")
        if saveFolder is not None:
            ax.scatter(velocities["Vx"][0::5000], velocities["Vy"][0::5000], velocities["Vz"][0::5000], marker='o')
        else:
            ax.scatter(velocities["Vx"][0::10000], velocities["Vy"][0::10000], velocities["Vz"][0::10000], marker='o')
        ax.set_title(f"{species}: t = 0.0")
        ax.set_xlabel('Vx')
        ax.set_ylabel('Vy')
        ax.set_zlabel('Vz')
        # Fit line to Vx-Vz
        if species == "alpha":
            x = np.linspace(np.min(velocities["Vx"]), np.max(velocities["Vx"]), 10000)
            ax.plot(x, [np.mean(velocities["Vy"])] * len(x), [np.mean(velocities["Vz"])] * len(x), color = "g", label="mean Vz")
            
            vx_vz_line = linregress(velocities["Vx"][0::1000], velocities["Vz"][0::1000])
            vx_vz = np.array([(x[-1] - x[0]), (((vx_vz_line.slope * x[-1]) + vx_vz_line.intercept) - ((vx_vz_line.slope * x[0]) + vx_vz_line.intercept))])
            normal = np.array([(x[-1] - x[0]), 0.0])
            dot = np.dot(vx_vz, normal)
            magnitude_vxvz = np.linalg.norm(vx_vz)
            magnitude_normal = np.linalg.norm(normal)
            angle_deg = np.degrees(np.arccos(dot / (magnitude_vxvz * magnitude_normal)))
            print(f"angle: {angle_deg}")
            ax.plot(x, [np.mean(velocities["Vy"])] * len(x), (vx_vz_line.slope * x) + vx_vz_line.intercept, color = "r", label = f"fit to Vx-Vz plane, angle to normal = {angle_deg:.5f}")
            fig.legend()
        if saveFolder is not None:
            filepath = saveFolder / f"{species}_3D.png"  
            fig.savefig(filepath, bbox_inches = "tight")
        else:
            plt.show()
        plt.close()

        

        # for dimension in velocities.keys():
        #     print(f"Processing {species}: {dimension}....")
        #     # For bin widths/range:
        #     dimData = velocities[dimension]
        #     upper_lim = max(vels.max() for vels in dimData.values())
        #     lower_lim = min(vels.min() for vels in dimData.values())
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--dir",
        action="store",
        help="Directory containing all sdf files from simulation.",
        required = True,
        type=Path
    )
    parser.add_argument(
        "--files",
        action="store",
        help="Indices of SDF files to process.",
        required = True,
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--vperp",
        action="store_true",
        help="Do vPerp calculations.",
        required = False
    )
    parser.add_argument(
        "--saveFolder",
        action="store",
        help="Directory in which to save plots.",
        required = False,
        type=Path
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Use log scale for histogram.",
        required = False
    )
    parser.add_argument(
        "--ddd",
        action="store_true",
        help="Plot particle velocities in 3D.",
        required = False
    )
    args = parser.parse_args()

    if args.ddd:
        plot_velo_3d(args.dir, args.files, args.saveFolder)
    else:
        plot_velo_dists(args.dir, args.files, args.vperp, args.saveFolder, args.log)