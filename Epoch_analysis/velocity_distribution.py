from pathlib import Path
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy import constants
from decimal import Decimal
import xarray as xr
import numpy as np
import epydeck
import argparse

def gaussian(x, a, mean, sigma):
    return a * np.exp(-((x - mean)**2 / (2 * sigma**2)))

def plot_velo_dist(directory : Path):

    # Read dataset
    ds = xr.open_dataset(
        str(directory / "0000.sdf"),
        keep_particles = True
    )

    # Read input deck
    input = {}
    with open(str(directory / "input.deck")) as id:
        input = epydeck.loads(id.read())
    
    B0 = input['constant']['b0_strength']
    electron_mass = input['species']['electron']['mass'] * constants.electron_mass
    ion_bkgd_mass = input['constant']['ion_mass_e'] * constants.electron_mass
    ion_ring_frac = input['constant']['frac_beam']
    fast_ion_mass_number_string = str(input['species']['ion_ring_beam']['mass']).replace("ion_mass_e", "").replace("*", "").strip()
    fast_ion_mass = ion_bkgd_mass if fast_ion_mass_number_string == '' else float(fast_ion_mass_number_string) * constants.electron_mass
    mass_density = input['constant']['background_density'] * (electron_mass + (fast_ion_mass * ion_ring_frac) + (ion_bkgd_mass * (1.0 - ion_ring_frac)))
    alfven_velo = B0 / (np.sqrt(constants.mu_0 * mass_density)) #m/s
    #print(input)

    vx = ds["Particles_Vx_ion_ring_beam"].load()
    vy = ds["Particles_Vy_ion_ring_beam"].load()

    # plt.hist(vx.data, bins = round(np.sqrt(len(vx))))
    # plt.title("Vx")
    # plt.show()

    # plt.hist(vy.data, bins = round(np.sqrt(len(vy))))
    # plt.title("Vy")
    # plt.show()

    v_perp = np.sqrt(vx**2 + vy**2)

    yhist, xhist, _ = plt.hist(v_perp.data, bins = round(np.sqrt(len(v_perp))))
    bin_centres = []
    for b in range(len(xhist) - 1):
        bin_centres.append(np.mean([xhist[b], xhist[b + 1]]))
    
    guesses = [np.max(yhist), np.mean(v_perp.data), np.std(v_perp.data)]
    popt, pcov = curve_fit(gaussian, bin_centres, yhist, guesses)
    #plt.hist(v_perp.data, bins=round(np.sqrt(len(v_perp))))
    print(*popt)
    plt.plot(bin_centres, gaussian(bin_centres, *popt),'r--')
    plt.title("V_perp")
    plt.show()

    alfven_velo = B0 / (np.sqrt(constants.mu_0 * mass_density))
    mean_vperp = np.mean(v_perp.data)
    ring_beam_energy = float(str(input['constant']['ring_beam_energy']))
    p_ring_beam = np.sqrt(2.0 * fast_ion_mass * constants.e * ring_beam_energy)
    p_beam = p_ring_beam * np.sin(-np.pi/3.0)
    p_ring = p_ring_beam * np.cos(-np.pi/3.0)
    v_beam = p_beam / fast_ion_mass
    v_ring = p_ring / fast_ion_mass
    effective_E_rb = fast_ion_mass * (v_beam**2 + v_ring**2) / (2.0 * constants.e)

    print(f"Alfven velo: {'%.2e' % Decimal(alfven_velo)}m/s")
    print(f"Set Ring beam E  = {'%.2e' % Decimal(ring_beam_energy)}")
    print(f"Calc Ring beam E = {'%.2e' % Decimal(effective_E_rb)}")
    print(f"Mean v_perp: {'%.2e' % Decimal(mean_vperp)}m/s")
    print(f"v_ring (v_perp): {'%.2e' % Decimal(v_ring)}")
    print(f"v_beam (v_para): {'%.2e' % Decimal(v_beam)}")
    
    velo_ratio = mean_vperp / alfven_velo
    print(f"Velocity ratio, v_perp/v_alfven: {'%.2e' % Decimal(velo_ratio)}")

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
        "--field",
        action="store",
        help="Simulation field to use in Epoch output format, e.g. \'Derived_Charge_Density\', \'Electric Field_Ex\', \'Magnetic Field_Bz\'",
        required = False,
        type=str
    )
    args = parser.parse_args()

    plot_velo_dist(args.dir)

    # B0 = 5.0
    # b_angle_deg = 80
    # b_ang_rad = b_angle_deg * np.pi/180.0
    # Bx = B0 * np.cos(b_ang_rad)
    # By = 0.0
    # Bz = B0 * np.sin(b_ang_rad)
    # vx = 3
    # vy = 5
    # vz = 9
    # v = np.array([vx, vy, vz])
    # B = np.array([Bx, By, Bz])
    # angle_v_to_B_rad = np.arccos(v.dot(B) / (np.sqrt(vx**2 + vy**2 + vz**2) * np.sqrt(Bx**2 + By**2 + Bz**2)))
    # print(f"angle_v_to_B_rad = {angle_v_to_B_rad}")
    # angle_v_to_B_deg = angle_v_to_B_rad * 180.0 / np.pi
    # print(f"angle_v_to_B_deg = {angle_v_to_B_deg}")

    # new_v_par = v * np.cos(angle_v_to_B_rad)
    # new_v_perp = v * np.sin(angle_v_to_B_rad)
    # print(f"new_V_par  = {new_v_par}")
    # print(f"new_V_perp = {new_v_perp}")
    # print(f"new_V_par mag  = {np.sqrt(new_v_par.dot(new_v_par))}")
    # print(f"new_V_perp mag = {np.sqrt(new_v_perp.dot(new_v_perp))}")

    # v_perp_x = vx - (vx * np.cos(b_ang_rad)**2 + vz * np.cos(b_ang_rad) * np.sin(b_ang_rad))
    # v_perp_y = vy
    # v_perp_z = vz - (vx * np.cos(b_ang_rad) * np.sin(b_ang_rad) + vz * np.sin(b_ang_rad)**2)
    # print(f"my vperp_x = {v_perp_x}")
    # print(f"my vperp_y = {v_perp_y}")
    # print(f"my vperp_z = {v_perp_z}")
    # print(f"my vperp   = {np.sqrt(v_perp_x**2 + v_perp_y**2 + v_perp_z**2)}")