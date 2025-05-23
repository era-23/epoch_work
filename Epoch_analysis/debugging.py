import argparse
from decimal import Decimal
from scipy import constants
import numpy as np
import epydeck
from pathlib import Path
from plasmapy.formulary import lengths as ppl
from plasmapy.formulary import speeds as pps
from plasmapy.formulary import frequencies as ppf
from astropy import units as u

def debug_input_deck(directory : Path):

    # Read input deck
    input = {}
    with open(str(directory / "input.deck")) as id:
        input = epydeck.loads(id.read())

    input_vals = input['constant']
    background_density = input_vals['background_density']
    background_temp = input_vals['background_temp'] * u.K
    b0_strength = input_vals['b0_strength'] * u.T

    pp_vThe_rms = pps.thermal_speed(background_temp, "e-", "rms")
    pp_vThe_mp = pps.thermal_speed(background_temp, "e-", "most_probable")
    pp_rLe_rel = ppl.gyroradius(b0_strength, "e-", T = background_temp)
    pp_rLe_nonRel = ppl.gyroradius(b0_strength, "e-", T = background_temp, relativistic=False)

    my_vThe_rms = np.sqrt(3.0) * np.sqrt(constants.k * (u.J/u.K) * background_temp / constants.electron_mass * u.kg)
    my_vThe_mp = np.sqrt(2.0) * np.sqrt(constants.k * (u.J/u.K) * background_temp / constants.electron_mass * u.kg)
    my_gyrofrequeny_e = (constants.elementary_charge * u.C * b0_strength) / constants.electron_mass * u.kg
    my_vPerpe_rms = my_vThe_rms
    my_vPerpe_mp = my_vThe_mp
    my_rLe_rms =  my_vPerpe_rms/ my_gyrofrequeny_e
    my_rLe_mp = my_vPerpe_mp / my_gyrofrequeny_e

    combined_rLe = np.sqrt(2.0 * constants.k * background_temp * constants.electron_mass) / (constants.elementary_charge * b0_strength)

    print(f"Plasmapy vThe rms: {pp_vThe_rms:.6e}")
    print(f"My vThe rms: {my_vThe_rms:.6e}")
    print(f"Plasmapy vThe mp: {pp_vThe_mp:.6e}")
    print(f"My vThe mp: {my_vThe_mp:.6e}")
    print(f"My vPerpe rms: {my_vPerpe_rms:.6e}")
    print(f"My vPerpe mp: {my_vPerpe_mp:.6e}")
    print(f"My gyrofrequency: {my_gyrofrequeny_e:.6e}")
    print(f"Plasmapy rLe (rel): {pp_rLe_rel:.6e}")
    print(f"Plasmapy rLe (non-rel): {pp_rLe_nonRel:.6e}")
    print(f"My rLe rms: {my_rLe_rms:.6e}")
    print(f"My rLe mp: {my_rLe_mp:.6e}")
    print(f"Combined rLe: {combined_rLe:.6e}")

    frac_beam = input_vals['frac_beam']
    ion_mass_e = input_vals['ion_mass_e']
    mass_background_ion = constants.electron_mass * ion_mass_e
    mass_fast_ion = constants.electron_mass * ion_mass_e
    background_mass_density = background_density * (mass_background_ion * (1.0 - frac_beam))
    b0_strength = input_vals['b0_strength']
    # Angle between B and x (predominantly in Z)
    alfven_velo = b0_strength / np.sqrt(constants.mu_0 * background_mass_density)
    #v_perp_ratio = input_vals['v_perp_ratio']
    v_perp_ratio = 0.634
    v_perp = v_perp_ratio * alfven_velo
    #ring_beam_energy = ((v_perp_ratio * alfven_velo)**2 * mass_fast_ion) / 2.0 * constants.e
    #pring_beam = np.sqrt(2 * mass_fast_ion * constants.e * ring_beam_energy)
    pring_beam = mass_fast_ion * v_perp * np.sqrt(1.0 / np.cos(-np.pi/3)**2)
    pbeam = pring_beam * np.sin(-np.pi/3)
    pring = pring_beam * np.cos(-np.pi/3)
    vring = pring / mass_fast_ion
    vbeam = pbeam / mass_fast_ion
    E_rb_si = mass_fast_ion * (vring**2 + vbeam**2) / 2.0
    E_rb_ev = E_rb_si / constants.e

    print(f"Alfven velo      = {'%.2e' % Decimal(alfven_velo)}")
    print(f"Ring beam E (si) = {'%.2e' % Decimal(E_rb_si)}")
    print(f"Ring beam E (ev) = {'%.2e' % Decimal(E_rb_ev)}")
    print(f"Ring v (vperp)   = {'%.2e' % Decimal(vring)}")
    print(f"Beam v (vpara)   = {'%.2e' % Decimal(vbeam)}")
    print(f"Intended v_perp/v_A = {'%.2e' % Decimal(v_perp_ratio)}")
    print(f"Actual v_perp/v_A   = {'%.2e' % Decimal(vring / alfven_velo)}")

    my_Debye_length = np.sqrt(constants.epsilon_0 * constants.k * background_temp / background_density / constants.elementary_charge**2) * u.m
    pp_Debye_length = ppl.Debye_length(background_temp, background_density / u.m**3)
    print(f"My Debye length: {my_Debye_length}")
    print(f"PP Debye length: {pp_Debye_length}")

    cell_width = 0.0
    if my_Debye_length.value <= combined_rLe.value:
        print("Debye length is smaller, using this for cell width.")
        cell_width = my_Debye_length
    else:
        print("Electron gyroradius is smaller, using this for cell width.")
        cell_width = combined_rLe
    print(f"Cell width: {cell_width}")
    
    num_cells = input_vals["num_cells"]
    sim_L = num_cells * cell_width * u.m
    print(f"Sim length : {sim_L}")
    ion_gyroperiod_s = (2.0 * np.pi * u.rad) / ppf.gyrofrequency(b0_strength * u.T, "p+")
    sim_L_vA_Tci = sim_L / (ion_gyroperiod_s * alfven_velo * (u.m / u.s))
    print(f"Sim length in k units: {sim_L_vA_Tci}")
    spatial_Nyquist = num_cells/(2.0 * sim_L_vA_Tci)
    print(f"Spatial Nyquist frequency: {spatial_Nyquist}")
    pixels_per_wavenumber = num_cells / (2.0 * spatial_Nyquist)
    print(f"Pixels per wavenumber: {pixels_per_wavenumber}")

    needed_ppk = 8.0
    needed_num_cells = np.ceil((needed_ppk * ion_gyroperiod_s * alfven_velo) / cell_width).astype(int)
    print(f"Minimum number of cells for ppk = {needed_ppk} is {needed_num_cells}")

    # all_variables = dir() 
  
    # # Iterate over the whole list where dir( ) 
    # # is stored. 
    # for name in all_variables: 
        
    #     # Print the item if it doesn't start with '__' 
    #     if not name.startswith('__'): 
    #         myvalue = eval(name) 
    #         print(name, " : ", type(myvalue), " = ", myvalue)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--dir",
        action="store",
        help="Directory containing input deck.",
        required = True,
        type=Path
    )
    args = parser.parse_args()

    debug_input_deck(args.dir)