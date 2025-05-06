from pathlib import Path
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy import constants
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

def plot_velo_dist(directory : Path, files : list, vPerp : bool):

    startTime_s = 0.0
    dims = ["vx", "vy", "vz"]
    colors = ["red", "orange", "yellow", "green", "blue"]
    # alphas = [1.0, 0.8, 0.6, 0.4, 0.2]
    lss = ["solid", "dashed", "dashdot", "dotted", "dotted"]

    speciess = {"electron" : {dim : {} for dim in dims}, "proton" : {dim : {} for dim in dims}, "ion_ring_beam" : {dim : {} for dim in dims}}

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

        for species in speciess: # Runs out of memory here with many particles on my laptop
            # data = ds[f"Particles_Vx_{species}"].load().data
            # #print(f"Max  velo: {data.max()}")
            # #print(f"Min  velo: {data.min()}")
            # #print(f"Mean velo: {np.mean(data)}")
            # #print(f"S.D. velo: {np.std(data)}")
            # speciess[species]["vx"][time_tci] = data
            # del(data)
            # data = ds[f"Particles_Vy_{species}"].load().data
            # #print(f"Max  velo: {data.max()}")
            # #print(f"Min  velo: {data.min()}")
            # #print(f"Mean velo: {np.mean(data)}")
            # #print(f"S.D. velo: {np.std(data)}")
            # speciess[species]["vy"][time_tci] = data
            # del(data)
            data = ds[f"Particles_Vz_{species}"].load().data
            #print(f"Max  velo: {data.max()}")
            #print(f"Min  velo: {data.min()}")
            #print(f"Mean velo: {np.mean(data)}")
            #print(f"S.D. velo: {np.std(data)}")
            speciess[species]["vz"][time_tci] = data
            del(data)
    
    for species in speciess.keys():
        
        for dimension in speciess[species].keys():
            i = 0
            for timepoint in speciess[species][dimension].keys():
                # bin_num = round(np.sqrt(len(vx)))
                bin_num = 100
                plt.hist(speciess[species][dimension][timepoint], bins = bin_num, color=colors[i], edgecolor = "black", alpha = 0.4, ls = lss[i], label = f"t = {timepoint:.2f}")
                i += 1

            plt.legend()
            plt.xlabel("Velocity [m/s]")
            plt.ylabel("Particle count")
            plt.title(f"{species}: {dimension}")
            plt.tight_layout()
            plt.show()

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
    args = parser.parse_args()

    plot_velo_dist(args.dir, args.files, args.vperp)