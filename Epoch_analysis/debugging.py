import argparse
from decimal import Decimal
from matplotlib import pyplot as plt
from scipy import constants
import numpy as np
import epydeck
import glob
from pathlib import Path
from plasmapy.formulary import lengths as ppl
from plasmapy.formulary import speeds as pps
from plasmapy.formulary import frequencies as ppf
from astropy import units as u

import warnings
warnings.filterwarnings("ignore")

def debug_input_deck(directory : Path):

    # Read input deck
    input = {}
    with open(glob.glob(str(directory / "input.deck"))[0]) as id:
        input = epydeck.loads(id.read())

    input_vals = input['constant']
    background_density = float(input_vals['background_density']) / u.m**3
    background_temp = float(input_vals['background_temp']) * u.K
    b0_strength = float(input_vals['b0_strength']) * u.T

    # pp_vThe_rms = pps.thermal_speed(background_temp, "e-", "rms")
    # pp_vThe_mp = pps.thermal_speed(background_temp, "e-", "most_probable")
    # pp_rLe_rel = ppl.gyroradius(b0_strength, "e-", T = background_temp)
    # pp_rLe_nonRel = ppl.gyroradius(b0_strength, "e-", T = background_temp, relativistic=False)

    # my_vThe_rms = np.sqrt(3.0) * np.sqrt(constants.k * (u.J/u.K) * background_temp / constants.electron_mass * u.kg)
    # my_vThe_mp = np.sqrt(2.0) * np.sqrt(constants.k * (u.J/u.K) * background_temp / constants.electron_mass * u.kg)
    # my_gyrofrequeny_e = (constants.elementary_charge * u.C * b0_strength) / constants.electron_mass * u.kg
    # my_vPerpe_rms = my_vThe_rms
    # my_vPerpe_mp = my_vThe_mp
    # my_rLe_rms =  my_vPerpe_rms/ my_gyrofrequeny_e
    # my_rLe_mp = my_vPerpe_mp / my_gyrofrequeny_e

    # combined_rLe = np.sqrt(2.0 * constants.k * background_temp * constants.electron_mass) / (constants.elementary_charge * b0_strength)

    # print(f"Plasmapy vThe rms: {pp_vThe_rms:.6e}")
    # print(f"My vThe rms: {my_vThe_rms:.6e}")
    # print(f"Plasmapy vThe mp: {pp_vThe_mp:.6e}")
    # print(f"My vThe mp: {my_vThe_mp:.6e}")
    # print(f"My vPerpe rms: {my_vPerpe_rms:.6e}")
    # print(f"My vPerpe mp: {my_vPerpe_mp:.6e}")
    # print(f"My gyrofrequency: {my_gyrofrequeny_e:.6e}")
    # print(f"Plasmapy rLe (rel): {pp_rLe_rel:.6e}")
    # print(f"Plasmapy rLe (non-rel): {pp_rLe_nonRel:.6e}")
    # print(f"My rLe rms: {my_rLe_rms:.6e}")
    # print(f"My rLe mp: {my_rLe_mp:.6e}")
    # print(f"Combined rLe: {combined_rLe:.6e}")

    # frac_beam = input_vals['frac_beam']
    # ion_mass_e = input_vals['background_ion_mass_e']
    # mass_background_ion = constants.electron_mass * ion_mass_e
    # mass_fast_ion = constants.electron_mass * ion_mass_e
    # background_mass_density = background_density * (mass_background_ion * (1.0 - frac_beam))
    # b0_strength = input_vals['b0_strength']
    # # Angle between B and x (predominantly in Z)
    # alfven_velo = b0_strength / np.sqrt(constants.mu_0 * background_mass_density)
    # #v_perp_ratio = input_vals['v_perp_ratio']
    # v_perp_ratio = 1.0
    # v_perp = v_perp_ratio * alfven_velo
    # #ring_beam_energy = ((v_perp_ratio * alfven_velo)**2 * mass_fast_ion) / 2.0 * constants.e
    # #pring_beam = np.sqrt(2 * mass_fast_ion * constants.e * ring_beam_energy)
    # pring_beam = mass_fast_ion * v_perp * np.sqrt(1.0 / np.cos(-np.pi/3)**2)
    # pbeam = pring_beam * np.sin(-np.pi/3)
    # pring = pring_beam * np.cos(-np.pi/3)
    # vring = pring / mass_fast_ion
    # vbeam = pbeam / mass_fast_ion
    # E_rb_si = mass_fast_ion * (vring**2 + vbeam**2) / 2.0
    # E_rb_ev = E_rb_si / constants.e

    # print(f"Alfven velo      = {'%.2e' % Decimal(alfven_velo.value)}")
    # print(f"Ring beam E (si) = {'%.2e' % Decimal(E_rb_si.value)}")
    # print(f"Ring beam E (ev) = {'%.2e' % Decimal(E_rb_ev.value)}")
    # print(f"Ring v (vperp)   = {'%.2e' % Decimal(vring.value)}")
    # print(f"Beam v (vpara)   = {'%.2e' % Decimal(vbeam.value)}")
    # print(f"Intended v_perp/v_A = {'%.2e' % Decimal(v_perp_ratio)}")
    # print(f"Actual v_perp/v_A   = {'%.2e' % Decimal(vring.value / alfven_velo.value)}")

    # my_Debye_length = np.sqrt(constants.epsilon_0 * constants.k * background_temp.value / background_density.value / constants.elementary_charge**2) * u.m
    # pp_Debye_length = ppl.Debye_length(background_temp, background_density)
    # print(f"My Debye length: {my_Debye_length}")
    # print(f"PP Debye length: {pp_Debye_length}")

    # cell_width = 0.0
    # if my_Debye_length.value <= combined_rLe.value:
    #     print("Debye length is smaller, using this for cell width.")
    #     cell_width = my_Debye_length
    # else:
    #     print("Electron gyroradius is smaller, using this for cell width.")
    #     cell_width = combined_rLe
    # print(f"Cell width: {cell_width} ({cell_width/my_Debye_length} Debye lengths) ({cell_width/combined_rLe} e- Larmor radii)")
    
    # ion_gyroperiod_s = (2.0 * np.pi * u.rad) / ppf.gyrofrequency(b0_strength * u.T, "p+")
    # print(f"Ion gyroperiod = {1.0 / ppf.gyrofrequency(b0_strength * u.T, 'p+')} or {(2.0 * np.pi * u.rad) / ppf.gyrofrequency(b0_strength * u.T, 'p+')}")
    # needed_ppk = 8.0
    # needed_num_cells = np.ceil((needed_ppk * ion_gyroperiod_s * alfven_velo) / cell_width).astype(int)
    # print(f"Minimum number of cells for ppk = {needed_ppk} is {needed_num_cells}")
    # num_cells = needed_num_cells
    # print(f"Num Cells : {num_cells}")
    # sim_L = num_cells * cell_width * u.m
    # print(f"Sim length : {sim_L}")
    # sim_L_vA_Tci = sim_L / (ion_gyroperiod_s * alfven_velo * (u.m / u.s))
    # print(f"Sim length in k units: {sim_L_vA_Tci}")
    # spatial_Nyquist = num_cells/(2.0 * sim_L_vA_Tci)
    # print(f"Spatial Nyquist frequency: {spatial_Nyquist}")
    # pixels_per_wavenumber = num_cells / (2.0 * spatial_Nyquist)
    # print(f"Pixels per wavenumber: {pixels_per_wavenumber}")
    

    # all_variables = dir() 
  
    # # Iterate over the whole list where dir( ) 
    # # is stored. 
    # for name in all_variables: 
        
    #     # Print the item if it doesn't start with '__' 
    #     if not name.startswith('__'): 
    #         myvalue = eval(name) 
    #         print(name, " : ", type(myvalue), " = ", myvalue)

def plot_particle_push_scaling(file : Path):

    # Read input deck
    input = {}
    with open(file) as deck:
        input = epydeck.loads(deck.read())

    input_vals = input['constant']
    background_density = float(input_vals['background_density']) / u.m**3
    background_temp = float(input_vals['background_temp']) * u.K
    b0_strength = float(input_vals['b0_strength']) * u.T
    fast_ion_charge_e = input_vals["fast_ion_charge_e"]
    pixels_per_k = input_vals["pixels_per_k"]
    background_ion_mass_e = input_vals["background_ion_mass_e"]
    background_ion_charge_e = input_vals["background_ion_charge_e"]
    fast_ion_mass_e = input_vals["fast_ion_mass_e"]
    frac_beam = input_vals["frac_beam"]

    print("JET case:")
    jet_case = calculate_number_of_particle_pushes(background_temp, background_density, b0_strength, frac_beam, doPrint=True)
    print("Worst previous case:")
    worst_old_case = calculate_number_of_particle_pushes(1e8 * u.K, 1e18 / u.m**3, 5 * u.T, 1e-2, fast_ion_charge_e=1.0, fast_ion_mass_e=3674.8, nyquist_omega_factor=100.0, num_time_samples=3500.0, doPrint=True)
    print("Best previous case:")
    best_old_case = calculate_number_of_particle_pushes(1e8 * u.K, 4e19 / u.m**3, 1.4 * u.T, 1e-4, fast_ion_charge_e=1.0, fast_ion_mass_e=3674.8, nyquist_omega_factor=100.0, num_time_samples=3500.0, doPrint=True)

    # Scan vals
    num_samples = 1000
    temp_vals_keV_dimless = np.linspace(0.8, 10, num_samples)
    temp_vals_K_dimless = np.array([t.value for t in (temp_vals_keV_dimless*u.keV).to(u.J) / constants.k])
    temp_vals_K = temp_vals_K_dimless * u.K
    density_vals_dimless = [10**v for v in np.linspace(17.9, 20.1, num_samples)]
    density_vals = np.array(density_vals_dimless) / u.m**3
    b0_vals_dimless = np.linspace(0.5, 5, num_samples)
    b0_vals = b0_vals_dimless * u.T
    beam_frac_vals = [10**v for v in np.linspace(-4.1, -1.9, num_samples)]
    nyquist_factors = np.linspace(10, 100, num_samples)
    time_samples_vals = np.linspace(1000, 5000, num_samples)
    ppc_vals = np.linspace(100, 1000, num_samples)

    # Mean vals
    temp_median = np.median(temp_vals_K_dimless) * u.K
    dens_median = np.median(density_vals_dimless) / u.m**3
    b0_median = np.median(b0_vals_dimless) * u.T
    bf_median = np.median(beam_frac_vals)
    nyq_median = np.median(nyquist_factors)
    time_samp_median = np.median(time_samples_vals)
    ppc_median = np.median(ppc_vals)

    # Temperature
    temp_ppushes = [calculate_number_of_particle_pushes(t, dens_median, b0_median, bf_median, nyquist_omega_factor=nyq_median, num_time_samples=time_samp_median, n_part_factor=ppc_median) for t in temp_vals_K]
    plt.plot(temp_vals_keV_dimless, temp_ppushes)
    plt.xlabel("temperature/keV")
    plt.ylabel("Total particle pushes")
    plt.title("Background temperature")
    plt.show()

    # Density
    dens_ppushes = [calculate_number_of_particle_pushes(temp_median, d, b0_median, bf_median, nyquist_omega_factor=nyq_median, num_time_samples=time_samp_median, n_part_factor=ppc_median) for d in density_vals]
    plt.plot(density_vals_dimless, dens_ppushes)
    plt.xlabel(r"density/$m^{-3}$")
    plt.xscale("log")
    plt.ylabel("Total particle pushes")
    plt.title("Background density")
    plt.show()

    # B0
    b0_ppushes = [calculate_number_of_particle_pushes(temp_median, dens_median, b0, bf_median, nyquist_omega_factor=nyq_median, num_time_samples=time_samp_median, n_part_factor=ppc_median) for b0 in b0_vals]
    plt.plot(b0_vals_dimless, b0_ppushes)
    plt.xlabel("B0/T")
    plt.ylabel("Total particle pushes")
    plt.title("B0")
    plt.show()

    # Beam fraction
    bf_ppushes = [calculate_number_of_particle_pushes(temp_median, dens_median, b0_median, bf, nyquist_omega_factor=nyq_median, num_time_samples=time_samp_median, n_part_factor=ppc_median) for bf in beam_frac_vals]
    plt.plot(beam_frac_vals, bf_ppushes)
    plt.xlabel("beam fraction")
    plt.xscale("log")
    plt.ylabel("Total particle pushes")
    plt.title("Beam fraction")
    plt.show()

    # Nyquist factors
    nyq_ppushes = [calculate_number_of_particle_pushes(temp_median, dens_median, b0_median, bf_median, nyquist_omega_factor=n, num_time_samples=time_samp_median, n_part_factor=ppc_median) for n in nyquist_factors]
    plt.plot(nyquist_factors, nyq_ppushes)
    plt.xlabel(r"Nyquist frequency/$\Omega_{c\alpha}$")
    plt.ylabel("Total particle pushes")
    plt.title("Nyquist factor")
    plt.show()

    # Num time samples
    n_ts_ppushes = [calculate_number_of_particle_pushes(temp_median, dens_median, b0_median, bf_median, nyquist_omega_factor=nyq_median, num_time_samples=ts, n_part_factor=ppc_median) for ts in time_samples_vals]
    plt.plot(time_samples_vals, n_ts_ppushes)
    plt.xlabel("Num time samples")
    plt.ylabel("Total particle pushes")
    plt.title("Number of output time samples")
    plt.show()

    # PPC
    ppc_ppushes = [calculate_number_of_particle_pushes(temp_median, dens_median, b0_median, bf_median, nyquist_omega_factor=nyq_median, num_time_samples=time_samp_median, n_part_factor=ppc) for ppc in ppc_vals]
    plt.plot(ppc_vals, ppc_ppushes)
    plt.xlabel("Particles per cell")
    plt.ylabel("Total particle pushes")
    plt.title("Particles per cell")
    plt.show()

    # B0 vs density heatmap
    b0s, denss = np.meshgrid(b0_vals, density_vals)
    push = calculate_number_of_particle_pushes(temp_median, denss, b0s, bf_median).astype(float)
    plt.imshow(push, cmap="plasma", origin="lower", interpolation='none', extent=[b0_vals_dimless[0], b0_vals_dimless[-1], density_vals_dimless[0], density_vals_dimless[-1]], aspect="auto")
    plt.colorbar().set_label("Total particle pushes")
    plt.xlabel(r"B0/$T$")
    plt.ylabel(r"Density/$m^{-3}$")
    plt.show()
    # B0 vs temperature
    b0s, temps = np.meshgrid(b0_vals, temp_vals_K)
    push = calculate_number_of_particle_pushes(temps, dens_median, b0s, bf_median).astype(float)
    plt.imshow(push, cmap="plasma", origin="lower", interpolation='none', extent=[b0_vals_dimless[0], b0_vals_dimless[-1], temp_vals_keV_dimless[0], temp_vals_keV_dimless[-1]], aspect="auto")
    plt.colorbar().set_label("Total particle pushes")
    plt.xlabel("B0/T")
    plt.ylabel(r"Temperature/$keV$")
    plt.show()

    # Density vs temperature
    denss, temps = np.meshgrid(density_vals, temp_vals_K)
    push = calculate_number_of_particle_pushes(temps, denss, b0_median, bf_median).astype(float)
    plt.imshow(push, cmap="plasma", origin="lower", interpolation='none', extent=[density_vals_dimless[0], density_vals_dimless[-1], temp_vals_keV_dimless[0], temp_vals_keV_dimless[-1]], aspect="auto")
    plt.colorbar().set_label("Total particle pushes")
    plt.xlabel(r"Density/$m^{-3}$")
    plt.ylabel(r"Temperature/$keV$")
    plt.show()

    # Comparison
    plt.plot(temp_ppushes, color = "r", label = "temperature")
    plt.plot(dens_ppushes, color = "b", label = "density")
    plt.plot(b0_ppushes, color = "purple", label = "B0")
    plt.plot(bf_ppushes, color = "g", label = "beam fraction")
    plt.plot(nyq_ppushes, color = "orange", label = "Nyquist factors")
    plt.plot(n_ts_ppushes, color = "turquoise", label = "time samples")
    plt.plot(ppc_ppushes, color = "black", label = "particles per cell")
    plt.title("Comparison")
    plt.legend()
    plt.xlabel("Typical simulation range")
    plt.ylabel("Total particle pushes")
    plt.show()

def calculate_number_of_particle_pushes(
        background_temp, 
        background_density, 
        b0_strength,
        frac_beam,
        background_ion_mass_e = 3674.8,
        background_ion_charge_e = 1.0,
        fast_ion_charge_e = 2.0,
        fast_ion_mass_e = 7294.3,
        pixels_per_k = 8,
        nyquist_omega_factor = 20.0,
        num_time_samples = 1000.0,
        n_part_factor = 900.0,
        doPrint : bool = False):
    
    kb = constants.k * (u.J / u.K)
    qe = constants.elementary_charge * u.C
    me = constants.electron_mass * u.kg

    my_Debye_length = np.sqrt(constants.epsilon_0 * (u.F / u.m) * kb * background_temp / background_density / qe**2).to(u.m)
    pp_Debye_length = ppl.Debye_length(background_temp, background_density) # Confirmed equal to above
    my_e_gyroradius = (np.sqrt(2.0 * kb * background_temp * constants.electron_mass * u.kg) / (qe * b0_strength)).to(u.m)
    pp_e_gyroradius = ppl.gyroradius(b0_strength, particle="e-", T=background_temp) # Confirmed equal to above
    #grid_spacing = my_Debye_length if my_Debye_length < my_e_gyroradius else my_e_gyroradius
    grid_spacing = np.minimum(my_Debye_length, my_e_gyroradius)
    dt = 0.95 * grid_spacing / (constants.c * u.m / u.s)
    fast_ion_charge = fast_ion_charge_e * qe
    fast_ion_mass = fast_ion_mass_e * me
    fast_ion_gyrofrequency = ((fast_ion_charge * b0_strength) / (2.0 * np.pi * fast_ion_mass)).to(u.Hz)
    pp_fast_ion_gyrofrequency = ppf.gyrofrequency(B = b0_strength, particle = "alpha") / (2.0 * np.pi * u.rad) # Confirmed equal to above
    fast_ion_gyroperiod = (1.0 / fast_ion_gyrofrequency).to(u.s)
    nyquist_omega = nyquist_omega_factor * fast_ion_gyrofrequency
    sim_time = (num_time_samples / (2.0 * nyquist_omega)).to(u.s)
    num_time_steps = np.ceil(sim_time / dt).astype(int)
    background_ion_mass = me * background_ion_mass_e
    background_ion_density = (background_density - (frac_beam * background_density * fast_ion_charge_e)) / background_ion_charge_e
    background_ion_mass_density = background_ion_mass * background_ion_density
    alfven_velo = b0_strength / np.sqrt(constants.mu_0 * (u.H / u.m) * background_ion_mass_density)
    pp_alfven_velo = pps.Alfven_speed(B = b0_strength, density=background_ion_density, ion="D+") # Confirmed close to above
    num_cells = np.ceil((pixels_per_k * fast_ion_gyroperiod * alfven_velo) / grid_spacing).astype(int)
    num_particles = n_part_factor * num_cells
    num_particle_pushes = num_particles * num_time_steps
    
    if doPrint:
        print("--------------------------------------------------------------------")
        print(f"Debye length = {my_Debye_length} ({pp_Debye_length} (pp))")
        print(f"Electron gyroradius = {my_e_gyroradius} ({pp_e_gyroradius} (pp))")
        print(f"Grid spacing = {grid_spacing}")
        print(f"dt = {dt}")
        print(f"num_time_samples = {num_time_samples}")
        print(f"fast_ion_charge = {fast_ion_charge}")
        print(f"fast_ion_mass = {fast_ion_mass}")
        print(f"fast_ion_gyrofrequency = {fast_ion_gyrofrequency}")
        print(f"PlasmaPy Alpha particle gyrofrequency = {pp_fast_ion_gyrofrequency}")
        print(f"fast_ion_gyroperiod = {fast_ion_gyroperiod}")
        print(f"sim_time = {sim_time}")
        print(f"n_time_steps = {num_time_steps}")
        print(f"Electron density = {background_density}")
        print(f"Fast ion density = {background_density * frac_beam}")
        print(f"Background ion density = {background_ion_density}")
        print(f"alfven_velo = {alfven_velo}")
        print(f"PlasmaPy Alfven velo = {pp_alfven_velo}")
        print(f"num_cells = {num_cells}")
        print(f"num_particles = {num_particles}")
        print(f"Total number of particle pushes = {num_particle_pushes:.3e}")
        print("--------------------------------------------------------------------")

    return num_particle_pushes.value

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--file",
        action="store",
        help="Directory containing input deck.",
        required = True,
        type=Path
    )
    args = parser.parse_args()

    plot_particle_push_scaling(args.file)
    # debug_input_deck(args.dir)