from pathlib import Path
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy import constants
from scipy.stats import skew, skewtest, kurtosis, kurtosistest
from decimal import Decimal
from plasmapy.formulary import speeds as pps
from plasmapy.formulary import frequencies as ppf
from scipy.stats import linregress
from scipy.optimize import curve_fit
import astropy.units as u
import xarray as xr
import numpy as np
import epydeck
import argparse

def gaussian(x, a, mean, sigma):
    return a * np.exp(-((x - mean)**2 / (2 * sigma**2)))

def eval_pRingBeam(fi_mass : float, fi_charge : float, ring_beam_energy : float):
    return np.sqrt(2.0 * fi_mass * fi_charge * ring_beam_energy)

def eval_pBeam(p_ring_beam : float, pitch : float):
    return p_ring_beam * pitch

def eval_pRing(p_ring_beam : float, pitch : float):
    return p_ring_beam * np.sqrt(1.0 - pitch**2)

def eval_pFastSpread(p_ring : float):
    return 0.06 * p_ring

def eval_px_peak(p_ring : float, p_fast_spread : float):
    return (p_ring / 2.0) * (1.0 + np.sqrt(1.0 + (2.0 * (p_fast_spread/p_ring))**2))

def dist_func_orig(V, px_peak : float, p_ring : float, p_fast_spread : float, p_beam : float):
    vx, vz = V
    px = vx * 7294.3 * constants.electron_mass
    pz = vz * 7294.3 * constants.electron_mass
    dist_fn = []
    assert len(px) == len(pz)
    for i in range(len(px)):
        dist_fn.append((px[i] / px_peak) * np.exp(-0.5 * ((px[i] - p_ring) / p_fast_spread)**2) * np.exp(-0.5 * ((pz[i] - p_beam) / p_fast_spread)**2))
    return np.array(dist_fn)

# def dist_func_forFitting(V, px_peak : float, p_ring : float, p_fast_spread : float, p_beam : float):
#     vx, vz = V
#     px = vx * 7294.3 * constants.electron_mass
#     pz = vz * 7294.3 * constants.electron_mass
#     dist_fn = []
#     assert len(px) == len(pz)
#     for i in range(len(px)):
#         dist_fn.append((px[i] / px_peak) * np.exp(-0.5 * (((px[i] - p_ring)**2 + (pz[i] - p_beam)**2) / p_fast_spread**2)))
#     return np.array(dist_fn)

def dist_func_zOnly(vz, px_over_px_peak : float, p_fast_spread : float, p_beam : float):
    pz = vz * 7294.3 * constants.electron_mass
    return px_over_px_peak * np.exp(-0.5 * ((pz - p_beam) / p_fast_spread)**2)

def dist_func_xOnly(vx, px_peak : float, p_fast_spread : float, p_ring : float):
    px = vx * 7294.3 * constants.electron_mass
    return (px / px_peak) * np.exp(-0.5 * ((px - p_ring) / p_fast_spread)**2)

def plot_velo_dists(directory : Path, file : str, vPerp : bool = False, saveFolder : Path = None, doLog : bool = True):

    # speciess = ["electron", "deuteron", "alpha"]
    speciess = ["alpha"]
    dims = ["Vx", "Vy", "Vz"]

    # Read input deck
    inputDeck = {}
    with open(str(directory / "input.deck")) as id:
        inputDeck = epydeck.loads(id.read())
    temp = inputDeck['constant']['background_temp'] * u.K

    for species in speciess:
        vTh = pps.thermal_speed(temp, "e-" if species == "electron" else "p+")
        velocities = {dim : {} for dim in dims}

        # Read dataset
        ds = xr.open_dataset(
            str(directory / f"{file}.sdf"),
            keep_particles = True
        )

        # Read input deck
        input = {}
        with open(str(directory / "input.deck")) as id:
            input = epydeck.loads(id.read())

        for dim in velocities.keys(): # Runs out of memory here with many particles on my laptop
            data = ds[f"Particles_{dim}_{species}"].load().data
            #print(f"Max  velo: {data.max()}")
            #print(f"Min  velo: {data.min()}")
            #print(f"Mean velo: {np.mean(data)}")
            #print(f"S.D. velo: {np.std(data)}")
            velocities[dim] = data
            del(data)

        if "Vx" in velocities and "Vy" in velocities:
            velocities["Vperp"] = np.sqrt(np.array(velocities["Vx"])**2 + np.array(velocities["Vy"])**2)

        if species == "alpha":
            p_ring_beam = eval_pRingBeam(input['constant']['fast_ion_mass_e'] * constants.electron_mass, input['constant']['fast_ion_charge_e'] * constants.elementary_charge, float(input['constant']['ring_beam_energy']))
            print(f"p_ring_beam = {p_ring_beam}")
            p_beam = eval_pBeam(p_ring_beam, input['constant']['pitch'])
            print(f"p_beam = {p_beam}")
            p_ring = eval_pRing(p_ring_beam, input['constant']['pitch'])
            print(f"p_ring = {p_ring}")
            p_fast_spread = eval_pFastSpread(p_ring)
            print(f"p_fast_spread = {p_fast_spread}")
            px_peak = eval_px_peak(p_ring, p_fast_spread)
            print(f"px_peak = {px_peak}")

        dimHists = dict.fromkeys(velocities.keys())

        dims = ["Vx", "Vy", "Vperp", "Vz"]
        for dimension in dims:
            print(f"Processing {species}: {dimension}....")
            # For bin widths/range:
            dimData = velocities[dimension]
            upper_lim = dimData.max()
            lower_lim = dimData.min()
            fig, ax = plt.subplots(figsize=(10,8))

            print(f"total particles: {len(dimData)}")
            bin_num = round(np.sqrt(len(dimData)))
            bin_num = int(np.min([1000, bin_num]))
            print(f"Using {bin_num} bins")
            mu = r'$\mu$'
            sigma = r'$\sigma$'
            vTh_unit = r'$v_{Th}$'
            vTh2_unit = r'$v_{Th}^2$'
            mean_vTh = (np.mean(dimData)*u.m/u.s) / vTh
            sigma_vTh2 = (np.var(dimData)*u.m**2/u.s**2) / vTh**2
            stats = f"{mu}: {mean_vTh:.3e}{vTh_unit}, {sigma}: {sigma_vTh2:.3e}{vTh2_unit}, Skew: {skew(dimData):.3e} (p-value: {skewtest(dimData).pvalue:.3f}),  Kurtosis: {kurtosis(dimData):.3e} (p-value: {kurtosistest(dimData).pvalue:.3f})"
            
            hist = ax.hist(
                dimData, 
                bins = bin_num, 
                density = True,
                range=[lower_lim, upper_lim], 
                log=doLog, 
                histtype='bar', 
                alpha = 1.0, 
                ls = "solid", 
                label = stats
            )

            dimHists[dimension] = hist

            box = ax.get_position()
            if species == "alpha":
                # skip_n = 100
                # Vx = velocities["Vperp"][0::skip_n]
                # Vz = np.array(velocities["Vz"][0::skip_n])
                # dist_fn = dist_func_orig((Vx, Vz), px_peak, p_ring, p_fast_spread, p_beam)
                # dist_fn_2 = dist_func_forFitting((Vx, Vz), px_peak, p_ring, p_fast_spread, p_beam)
                # assert (np.allclose(dist_fn,dist_fn_2))

                # if dimension == "Vperp":
                #     ax2 = ax.twinx()
                #     # Fit dist_func
                #     Vx_bins = dimHists["Vperp"][1][:-1]
                #     Vx_vals = dimHists["Vperp"][0]
                #     parameters_xOnly, pcovX = curve_fit(dist_func_xOnly, Vx_bins, Vx_vals, p0=(px_peak, p_fast_spread, p_ring))
                #     print(f"Original px_peak:       {px_peak}, Fit px_peak:       {parameters_xOnly[0]} ({(parameters_xOnly[0]*100.0/px_peak):.4f}%)")
                #     print(f"Original p_fast_spread: {p_fast_spread},  Fit p_fast_spread: {parameters_xOnly[1]} ({(parameters_xOnly[1]*100.0/p_fast_spread):.4f}%)")
                #     print(f"Original p_ring:        {p_ring},  Fit p_ring:        {parameters_xOnly[2]} ({(parameters_xOnly[2]*100.0/p_ring):.4f}%)")
                #     print(f"Errors: {np.sqrt(np.diag(pcovX))}")
                #     ax2.plot(Vx_bins, dist_func_xOnly(Vx_bins, *parameters_xOnly), color="g", label = "Dist func in Vperp only")
                #     ax2.set_ylabel('dist_func')
                #     ax2.set_ylim(bottom=0.0)

                if dimension == "Vz":
                    ax2 = ax.twinx()
                    # Fit dist_func
                    # parameters, pcov = curve_fit(dist_func_zOnly, bins[:-1], particle_count, p0=(2e6, 9e2, 1e6, 9e2))
                    Vx_bins = dimHists["Vperp"][1][:-1]
                    Vz_bins = dimHists["Vz"][1][:-1]
                    Vz_vals = dimHists["Vz"][0]

                    parameters, pcov = curve_fit(dist_func_orig, (Vx_bins, Vz_bins), Vz_vals, p0=(px_peak, p_ring, p_fast_spread, p_beam))
                    print(f"Original px_peak:       {px_peak}, Fit px_peak:       {parameters[0]} ({(parameters[0]*100.0/px_peak):.4f}%)")
                    print(f"Original p_ring:        {p_ring},  Fit p_ring:        {parameters[1]} ({(parameters[1]*100.0/p_ring):.4f}%)")
                    print(f"Original p_fast_spread: {p_fast_spread},  Fit p_fast_spread: {parameters[2]} ({(parameters[2]*100.0/p_fast_spread):.4f}%)")
                    print(f"Original p_beam:        {p_beam}, Fit p_beam:        {parameters[3]}  ({(parameters[3]*100.0/p_beam):.4f}%)")
                    print(f"Errors: {np.sqrt(np.diag(pcov))}")
                    ax2.plot(Vz_bins, dist_func_orig((Vx_bins, Vz_bins), *parameters), color="r", label = "Fit dist func")

                    # parameters_zOnly, pcovZ = curve_fit(dist_func_zOnly, Vz_bins, Vz_vals, p0=(px_peak, p_fast_spread, p_beam))
                    # print(f"Original px_peak:       {px_peak}, Vz Fit px_over_px_peak:       {parameters[0]} ({(parameters[0]*100.0/px_peak):.4f}%)")
                    # print(f"Original p_fast_spread: {p_fast_spread},  Vz Fit p_fast_spread: {parameters[1]} ({(parameters[1]*100.0/p_fast_spread):.4f}%)")
                    # print(f"Original p_beam:        {p_beam}, Vz Fit p_beam:        {parameters[2]}  ({(parameters[1]*100.0/p_beam):.4f}%)")
                    # print(f"Errors: {np.sqrt(np.diag(pcovZ))}")
                    # ax2.plot(Vz_bins, dist_func_zOnly(Vz_bins, *parameters_zOnly), color="g", label = "Dist func in Vz only")
                    ax2.set_ylabel('dist_func')
                    ax2.set_ylim(bottom=0.0)

                fig.tight_layout()
            
            ax.set_position([box.x0, box.y0 + (0.2 * box.height), box.width, (box.height*0.8)])
            ax.legend(bbox_to_anchor=(0.5, -0.1), loc = "upper center")
            ax.set_xlabel("Velocity [m/s]")
            ax.set_ylabel("Particle probability density")
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
        "--file",
        action="store",
        help="Index of SDF file to process.",
        required = True,
        type=str
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
        plot_velo_dists(args.dir, args.file, args.vperp, args.saveFolder, args.log)