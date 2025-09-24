from pathlib import Path
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy import constants
from scipy.stats import skew, skewtest, kurtosis, kurtosistest
from decimal import Decimal
from plasmapy.formulary import speeds as pps
from plasmapy.formulary import frequencies as ppf
import astropy.units as u
import xarray as xr
import numpy as np
import epydeck
import argparse

def gaussian(x, a, mean, sigma):
    return a * np.exp(-((x - mean)**2 / (2 * sigma**2)))

def plot_velo_dist(directory : Path, files : list, vPerp : bool = False, saveFolder : Path = None, doLog : bool = True):

    startTime_s = 0.0
    speciess = ["electron", "proton", "ion_ring_beam"]
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

    # if vPerp:
    #     v_perp = np.sqrt(vx**2 + vy**2)

    #     yhist, xhist, _ = plt.hist(v_perp.data, bins = round(np.sqrt(len(v_perp))))
    #     bin_centres = []
    #     for b in range(len(xhist) - 1):
    #         bin_centres.append(np.mean([xhist[b], xhist[b + 1]]))
        
    #     guesses = [np.max(yhist), np.mean(v_perp.data), np.std(v_perp.data)]
    #     popt, pcov = curve_fit(gaussian, bin_centres, yhist, guesses)
    #     #plt.hist(v_perp.data, bins=round(np.sqrt(len(v_perp))))
    #     print(*popt)
    #     plt.plot(bin_centres, gaussian(bin_centres, *popt),'r--')
    #     plt.title("V_perp")
    #     plt.show()

    #     mean_vperp = np.mean(v_perp.data)
    #     ring_beam_energy = float(str(input['constant']['ring_beam_energy']))
    #     fast_ion_mass = constants.proton_mass
    #     p_ring_beam = np.sqrt(2.0 * ring_beam_energy)
    #     p_beam = p_ring_beam * np.sin(-np.pi/3.0)
    #     p_ring = p_ring_beam * np.cos(-np.pi/3.0)
    #     v_beam = p_beam / fast_ion_mass
    #     v_ring = p_ring / fast_ion_mass
    #     effective_E_rb = fast_ion_mass * (v_beam**2 + v_ring**2) / (2.0 * constants.e)

    #     print(f"Alfven velo: {'%.2e' % Decimal(alfven_velo)}m/s")
    #     print(f"Set Ring beam E  = {'%.2e' % Decimal(ring_beam_energy)}")
    #     print(f"Calc Ring beam E = {'%.2e' % Decimal(effective_E_rb)}")
    #     print(f"Mean v_perp: {'%.2e' % Decimal(mean_vperp)}m/s")
    #     print(f"v_ring (v_perp): {'%.2e' % Decimal(v_ring)}")
    #     print(f"v_beam (v_para): {'%.2e' % Decimal(v_beam)}")
        
    #     velo_ratio = mean_vperp / alfven_velo
    #     print(f"Velocity ratio, v_perp/v_alfven: {'%.2e' % Decimal(velo_ratio)}")

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
    args = parser.parse_args()

    plot_velo_dist(args.dir, args.files, args.vperp, args.saveFolder, args.log)