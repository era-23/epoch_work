import argparse
from decimal import Decimal
from scipy import constants
import numpy as np
import epydeck
from pathlib import Path

def debug_vperp_ratio(directory : Path):

    # Read input deck
    input = {}
    with open(str(directory / "input.deck")) as id:
        input = epydeck.loads(id.read())

    input_vals = input['constant']
    background_density = input_vals['background_density']
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

    debug_vperp_ratio(args.dir)