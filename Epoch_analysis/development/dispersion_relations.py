from sdf_xarray import SDFPreprocess
from pathlib import Path
from scipy import constants, optimize
from matplotlib import colors
from plasmapy.formulary import frequencies as ppf
from plasmapy.formulary import speeds as pps
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import argparse
import xrft
import epydeck

class MySdfPreprocess(SDFPreprocess):
    def __init__(self):
        SDFPreprocess.__init__(self)

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        if self.job_id is None:
            self.job_id = ds.attrs["jobid1"]

        if self.job_id != ds.attrs["jobid1"]:
            raise ValueError(
                f"Mismatching job ids (got {ds.attrs['jobid1']}, expected {self.job_id})"
            )
        
        # ds_clean = ds.drop_dims([d for d in ds.dims if ("px" in d or "py" in d or "pz" in d)])

        # new_ds = xr.combine_by_coords(ds_clean, data_vars="minimal", combine_attrs="drop_conflicts")

        #new_ds = ds_clean.expand_dims(time=[ds_clean.attrs["time"]])
        new_ds = ds.expand_dims(time=[ds.attrs["time"]])

        # Particles' spartial coordinates also evolve in time
        for coord, value in new_ds.coords.items():
            if "Particles" in coord:
                new_ds.coords[coord] = value.expand_dims(time=[new_ds.attrs["time"]])

        return new_ds.drop_dims([c for c in new_ds.coords if ("px" in c or "py" in c or "pz" in c)])

#def calculate_CPDR_perp():

def R(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e):
    ion_component = plasma_freq_i**2 / (omega * (omega + cyclo_freq_i))
    electron_component = plasma_freq_e**2 / (omega * (omega + cyclo_freq_e))
    # electron_component = 0.0
    return 1.0 - (ion_component + electron_component)

def L(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e):
    ion_component = plasma_freq_i**2 / (omega * (omega - cyclo_freq_i))
    electron_component = plasma_freq_e**2 / (omega * (omega - cyclo_freq_e))
    # electron_component = 0.0
    return 1.0 - (ion_component + electron_component)

def P(omega, plasma_freq_i, plasma_freq_e):
    ion_component = plasma_freq_i**2 / omega**2
    electron_component = plasma_freq_e**2 / omega**2
    # electron_component = 0.0
    return 1.0 - (ion_component + electron_component)

def S(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e):
    return 0.5 * (R(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e) + L(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e))

def D(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e):
    return 0.5 * (R(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e) - L(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e))

def B(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e, theta):
    b1 = R(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e) * L(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e) * np.sin(theta)**2
    b2 = P(omega, plasma_freq_i, plasma_freq_e) * S(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e) * (1.0 + np.cos(theta)**2)
    return b1 + b2

def F_squared(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e, theta):
    f1 = (
        ((R(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e) * L(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e))
        - (P(omega, plasma_freq_i, plasma_freq_e) * S(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e)))**2
        * np.sin(theta)**4
    )
    f2 = (
        4.0 * P(omega, plasma_freq_i, plasma_freq_e)**2 * D(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e)**2 * np.cos(theta)**2
    )
    return f1 + f2

def A(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e, theta):
    a1 = S(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e) * np.sin(theta)**2
    a2 = P(omega, plasma_freq_i, plasma_freq_e) * np.cos(theta)**2
    return a1 + a2

def n_squared_solvable(omega, k, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e, theta, positive_root : bool):
    lhs = k**2 * constants.c**2 / omega**2
    rhs_numerator = B(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e, theta) + ((1.0 if positive_root else -1.0) * np.sqrt(F_squared(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e, theta)))
    rhs_denominator = 2.0 * A(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e, theta)
    rhs = rhs_numerator / rhs_denominator
    return lhs-rhs

def n_squared_perp_1(omega, k, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e):
    lhs = k**2 * constants.c**2 / omega**2
    RL = R(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e) * L(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e)
    rhs = RL / S(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e)
    return lhs - rhs

def n_squared_perp_2(omega, k, plasma_freq_i, plasma_freq_e):
    lhs = k**2 * constants.c**2 / omega**2
    rhs = P(omega, plasma_freq_i, plasma_freq_e)
    return lhs - rhs

# n^2 = R
def k_squared_para_1(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e):
    return (omega**2.0 / constants.c**2.0) * R(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e)

# n^2 = L
def k_squared_para_2(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e):
    return (omega**2.0 / constants.c**2.0) * L(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e)

# n^2 = RL/S
def k_squared_perp_1(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e):
    RL = R(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e) * L(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e)
    k_squared = (omega**2.0 / constants.c**2.0) * (RL / S(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e))
    return k_squared

# n^2 = P
def k_squared_perp_2(omega, plasma_freq_i, plasma_freq_e):
    return (omega**2.0 / constants.c**2.0) * P(omega, plasma_freq_i, plasma_freq_e)

def k_squared(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e, theta):
    rhs_numerator_posF = B(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e, theta) + np.sqrt(F_squared(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e, theta))
    rhs_numerator_negF = B(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e, theta) - np.sqrt(F_squared(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e, theta))
    rhs_denominator = 2.0 * A(omega, plasma_freq_i, plasma_freq_e, cyclo_freq_i, cyclo_freq_e, theta)
    k_squared_posF = (omega**2.0 / constants.c**2.0) * (rhs_numerator_posF / rhs_denominator)
    k_squared_negF = (omega**2.0 / constants.c**2.0) * (rhs_numerator_negF / rhs_denominator)
    return k_squared_posF, k_squared_negF

def eval_CPDR_rootFinding(species = "D+", theta = np.pi / 2.0, density = 1e19, B = 2.0, positive_root = False):
    plasma_freq_i = ppf.plasma_frequency(n = density / u.m**3, particle = species)
    plasma_freq_e = ppf.plasma_frequency(n = density / u.m**3, particle = "e-")
    cyclo_freq_i = ppf.gyrofrequency(B = B * u.T, particle = species)
    cyclo_freq_e = ppf.gyrofrequency(B = B * u.T, particle = "e-")
    alfven_velo = pps.Alfven_speed(B = B * u.T, density = density / u.m**3, ion = species)
    k_vals = np.linspace(0.01 * (alfven_velo.value / cyclo_freq_i.value), 200.0 * (alfven_velo.value / cyclo_freq_i.value), 1000)
    # w_sols = []
    w_sols_pos = []
    w_sols_neg = []
    bracket = [0.00001 * cyclo_freq_i.value, 10000.0 * cyclo_freq_i.value]
    for k in k_vals:
        fw_pos = lambda w : n_squared_solvable(w, k, plasma_freq_i.value, plasma_freq_e.value, cyclo_freq_i.value, cyclo_freq_e.value, theta, positive_root = True)
        fw_neg = lambda w : n_squared_solvable(w, k, plasma_freq_i.value, plasma_freq_e.value, cyclo_freq_i.value, cyclo_freq_e.value, theta, positive_root = False)
        sol_pos = optimize.root_scalar(fw_pos, bracket=bracket)
        sol_neg = optimize.root_scalar(fw_neg, bracket=bracket)
        w_sols_pos.append(sol_pos)
        w_sols_neg.append(sol_neg)
        # fw_1 = lambda w : n_squared_perp_1(w, k, plasma_freq_i.value, plasma_freq_e.value, cyclo_freq_i.value, cyclo_freq_e.value)
        # sol_1 = optimize.root_scalar(fw_1, bracket=bracket)
        # w_sols_1.append(sol_1)
        # fw_2 = lambda w : n_squared_perp_2(w, k, plasma_freq_i.value, plasma_freq_e.value)
        # sol_2 = optimize.root_scalar(fw_2, bracket=bracket)
        # w_sols_2.append(sol_2)
    w_cyclos_pos = [w.root for w in w_sols_pos] / cyclo_freq_i.value
    w_cyclos_neg = [w.root for w in w_sols_neg] / cyclo_freq_i.value
    # w_cyclos_1 = [w.root for w in w_sols_1] / cyclo_freq_i.value
    # w_cyclos_2 = [w.root for w in w_sols_2] / cyclo_freq_i.value
    k_vA_cyclos = k_vals / (alfven_velo.value / cyclo_freq_i.value)
    omega_lh = ppf.lower_hybrid_frequency(B = B * u.T, n_i = density / u.m**3, ion = species)
    omega_uh = ppf.upper_hybrid_frequency(B = B * u.T, n_e = density / u.m**3)
    omega_ec = ppf.gyrofrequency(B = B * u.T, particle = "e-")
    lh_cyclos = omega_lh.value / cyclo_freq_i.value
    uh_cyclos = omega_uh.value / cyclo_freq_i.value
    ec_cyclos = omega_ec / cyclo_freq_i
    print(f"LH in gyrofrequencies: {lh_cyclos}")
    print(f"UH in gyrofrequencies: {uh_cyclos}")
    print(f"e- gyrofrequency in ion gyrofrequencies: {ec_cyclos}")
    plt.plot(k_vA_cyclos, w_cyclos_pos, label = f"theta = {theta:.3f}, +ve root F")
    # plt.plot(k_vA_cyclos, w_cyclos_neg, label = f"theta = {theta:.3f}, -ve root F")
    # plt.plot(k_vA_cyclos, w_cyclos_1, label = "RL/S")
    # plt.plot(k_vA_cyclos, w_cyclos_2, label = "P")
    # plt.plot(np.linspace(0.0, w_cyclos.max(), 100), np.linspace(0.0, w_cyclos.max(), 100), color = "black", linestyle = "dotted", label = "omega = kVa")
    # plt.plot(k_vA_cyclos, k_vA_cyclos * (constants.c / alfven_velo.value), color = "black", linestyle = "dashdot", label = "omega = kc")
    # plt.axhline(lh_cyclos, color = "black", linestyle = "dashed", label = "Lower hybrid frequency")
    # plt.axhline(plasma_freq_i.value / cyclo_freq_i.value, color = "black", linestyle = "dashdot", label = "Ion plasma frequency")
    #plt.ylim(0.0, 50.0)
    #plt.xlim(0.0, 200.0)
    # plt.yscale("log")
    plt.legend()
    plt.show()
    print("woh")

def eval_CDPR_scanning(species = "p+", density = 2.5e19, B = 2.5):
    plasma_freq_i = ppf.plasma_frequency(n = density / u.m**3, particle = species)
    plasma_freq_e = ppf.plasma_frequency(n = density / u.m**3, particle = "e-")
    cyclo_freq_i = ppf.gyrofrequency(B = B * u.T, particle = species)
    cyclo_freq_e = ppf.gyrofrequency(B = B * u.T, particle = "e-")
    alfven_velo = pps.Alfven_speed(B = B * u.T, density = density / u.m**3, ion = species)
    omega_lh = ppf.lower_hybrid_frequency(B = B * u.T, n_i = density / u.m**3, ion = species)
    omega_uh = ppf.upper_hybrid_frequency(B = B * u.T, n_e = density / u.m**3)
    print(f"Ion plasma freq at {plasma_freq_i} or {plasma_freq_i/cyclo_freq_i}omega_ci")
    print(f"Electron plasma freq at {plasma_freq_e} or {plasma_freq_e/cyclo_freq_i}omega_ci")
    print(f"Lower hybrid at {omega_lh} or {omega_lh/cyclo_freq_i}omega_ci")
    print(f"Upper hybrid at {omega_uh} or {omega_uh/cyclo_freq_i}omega_ci")

    min_omega = 0.0001 * cyclo_freq_i.value
    max_omega = 2.0 * omega_lh.value
    omega_vals = np.linspace(min_omega, max_omega, 2000)

    # Parallel
    k_squared_1 = k_squared_para_1(omega_vals, plasma_freq_i.value, plasma_freq_e.value, cyclo_freq_i.value, cyclo_freq_e.value)
    k_squared_2 = k_squared_para_2(omega_vals, plasma_freq_i.value, plasma_freq_e.value, cyclo_freq_i.value, cyclo_freq_e.value)
    k_1 = np.sqrt(np.abs(k_squared_1)) / (cyclo_freq_i.value / alfven_velo.value)
    k_2 = np.sqrt(np.abs(k_squared_2)) / (cyclo_freq_i.value / alfven_velo.value)
    plt.title("Parallel")
    plt.plot(k_1, omega_vals / cyclo_freq_i.value, label = "R", color = "red", alpha = 0.75)
    plt.plot(k_2, omega_vals / cyclo_freq_i.value, label = "L", color = "blue", alpha = 0.75)
    plt.plot(omega_vals / cyclo_freq_i.value, omega_vals / cyclo_freq_i.value, label = r'$\Omega = kv_A$', color = "black", linestyle = "dotted")
    plt.axhline(omega_lh.value / cyclo_freq_i.value, color = "black", linestyle = "dashed", label = "LH")
    plt.axhline(plasma_freq_i.value / cyclo_freq_i.value, color = "black", linestyle = "dashdot", label = r"$\Omega_{pi}$")
    plt.axhline(1.0, color = "black", linestyle = "dotted", label = r"1.0$\Omega_{ci}$")
    plt.xlabel(r"Wavenumber [$\Omega_{ci}/v_A$]")
    plt.ylabel(r"Frequency [$\Omega_{ci}$]")
    plt.xlim(0.0, np.max([k_1, k_2]))
    plt.legend()
    plt.show()

    # k_squared_plusF, k_squared_minusF = k_squared(omega_vals, plasma_freq_i.value, plasma_freq_e.value, cyclo_freq_i.value, cyclo_freq_e.value, theta = 0.0)
    # k_squared_plusF = np.sqrt(np.abs(k_squared_plusF)) / (cyclo_freq_i.value / alfven_velo.value) # Order of abs and sqrt is significant!
    # k_squared_minusF = np.sqrt(np.abs(k_squared_minusF)) / (cyclo_freq_i.value / alfven_velo.value)
    # plt.title("Parallel")
    # plt.plot(k_squared_plusF, omega_vals / cyclo_freq_i.value, label = "+F", color = "red", alpha = 0.75)
    # plt.plot(k_squared_minusF, omega_vals / cyclo_freq_i.value, label = "-F", color = "blue", alpha = 0.75)
    # plt.plot(omega_vals / cyclo_freq_i.value, omega_vals / cyclo_freq_i.value, label = r'$\Omega = kv_A$', color = "black", linestyle = "dashdot")
    # plt.axhline(omega_lh.value / cyclo_freq_i.value, color = "black", linestyle = "dashed", label = "LH")
    # plt.axhline(1.0, color = "black", linestyle = "dotted", label = r"1.0$\Omega_{ci}$")
    # plt.xlabel(r"Wavenumber [$\Omega_{ci}/v_A$]")
    # plt.ylabel(r"Frequency [$\Omega_{ci}$]")
    # #plt.xlim(0.0, 100.0)
    # plt.legend()
    # plt.show()

    # Perpendicular
    k_squared_1 = k_squared_perp_1(omega_vals, plasma_freq_i.value, plasma_freq_e.value, cyclo_freq_i.value, cyclo_freq_e.value)
    k_squared_2 = k_squared_perp_2(omega_vals, plasma_freq_i.value, plasma_freq_e.value)
    k_1 = np.sqrt(np.abs(k_squared_1)) / (cyclo_freq_i.value / alfven_velo.value)
    k_2 = np.sqrt(np.abs(k_squared_2)) / (cyclo_freq_i.value / alfven_velo.value)
    plt.title("Perpendicular")
    plt.plot(k_1, omega_vals / cyclo_freq_i.value, label = "RL/S", color = "red", alpha = 0.75)
    plt.plot(k_2, omega_vals / cyclo_freq_i.value, label = "P", color = "blue", alpha = 0.75)
    plt.plot(omega_vals / cyclo_freq_i.value, omega_vals / cyclo_freq_i.value, label = r'$\Omega = kv_A$', color = "black", linestyle = "dotted")
    plt.axhline(omega_lh.value / cyclo_freq_i.value, color = "black", linestyle = "dashed", label = "LH")
    plt.axhline(plasma_freq_i.value / cyclo_freq_i.value, color = "black", linestyle = "dashdot", label = r"$\Omega_{pi}$")
    plt.axhline(1.0, color = "black", linestyle = "dotted", label = r"1.0$\Omega_{ci}$")
    plt.xlabel(r"Wavenumber [$\Omega_{ci}/v_A$]")
    plt.ylabel(r"Frequency [$\Omega_{ci}$]")
    plt.xlim(0.0, 100.0)
    plt.legend()
    plt.show()

    # k_squared_plusF, k_squared_minusF = k_squared(omega_vals, plasma_freq_i.value, plasma_freq_e.value, cyclo_freq_i.value, cyclo_freq_e.value, theta = np.pi / 2.0)
    # k_squared_plusF = np.sqrt(np.abs(k_squared_plusF)) / (cyclo_freq_i.value / alfven_velo.value)
    # k_squared_minusF = np.sqrt(np.abs(k_squared_minusF)) / (cyclo_freq_i.value / alfven_velo.value)
    # plt.title("Perpendicular")
    # plt.plot(k_squared_plusF, omega_vals / cyclo_freq_i.value, label = "+F", color = "red", alpha = 0.75)
    # plt.plot(k_squared_minusF, omega_vals / cyclo_freq_i.value, label = "-F", color = "blue", alpha = 0.75)
    # plt.plot(omega_vals / cyclo_freq_i.value, omega_vals / cyclo_freq_i.value, label = r'$\Omega = kv_A$', color = "black", linestyle = "dashdot")
    # plt.axhline(omega_lh.value / cyclo_freq_i.value, color = "black", linestyle = "dashed", label = "LH")
    # plt.axhline(1.0, color = "black", linestyle = "dotted", label = r"1.0$\Omega_{ci}$")
    # plt.xlabel(r"Wavenumber [$\Omega_{ci}/v_A$]")
    # plt.ylabel(r"Frequency [$\Omega_{ci}$]")
    # plt.xlim(0.0, 100.0)
    # plt.legend()
    # plt.show()


if __name__ == "__main__":

    #eval_CPDR_rootFinding()
    eval_CDPR_scanning()

    # parser = argparse.ArgumentParser("parser")
    # parser.add_argument(
    #     "--dir",
    #     action="store",
    #     help="Directory containing all sdf files from simulation.",
    #     required = True,
    #     type=Path
    # )
    # parser.add_argument(
    #     "--maxK",
    #     action="store",
    #     help="Max value of k to plot on the x-axis.",
    #     required = True,
    #     type=float
    # )
    # args = parser.parse_args()

    # directory = args.dir

    # ds = xr.open_mfdataset(
    #     str(directory / "*.sdf"),
    #     data_vars='minimal', 
    #     coords='minimal', 
    #     compat='override', 
    #     preprocess=MySdfPreprocess()
    # )

    # # Load input.deck content
    # deck = dict()
    # with open(args.dir / "input.deck") as inputDeck:
    #     deck = epydeck.load(inputDeck)

    # data_to_plot = []
    # # electric_field_ex : xr.DataArray = ds["Electric Field/Ex"]
    # # electric_field_ey : xr.DataArray = ds["Electric Field/Ey"]
    # # electric_field_ez : xr.DataArray = ds["Electric Field/Ez"]
    # magnetic_field_by : xr.DataArray = ds["Magnetic Field/By"]
    # magnetic_field_bz : xr.DataArray = ds["Magnetic Field/Bz"]
    # # charge_density_all : xr.DataArray = ds["Derived/Charge_Density"]
    # # charge_density_irb : xr.DataArray = ds["Derived/Charge_Density/ion_ring_beam"]
    # # charge_density_p : xr.DataArray = ds["Derived/Charge_Density/proton"]
    # # charge_density_e : xr.DataArray = ds["Derived/Charge_Density/electron"]

    # # data_to_plot.append(magnetic_field_by)
    # # data_to_plot.append(magnetic_field_bz)
    # # data_to_plot.append(electric_field_ex)
    # # data_to_plot.append(electric_field_ey)
    # # data_to_plot.append(electric_field_ez)
    # # data_to_plot.append(charge_density_irb)
    # # data_to_plot.append(charge_density_p)
    # # data_to_plot.append(charge_density_e)
    # # data_to_plot.append(charge_density_all)
    
    # # data_to_plot.append(pos_charge)
    # # data_to_plot.append(all_charge)

    # # for data in data_to_plot:
    # #     x = plt.figure()
    # #     data.plot()
    # #     x.show()

    # # plt.show()

    # #magnetic_field_by_c = magnetic_field_by.chunk({'time': magnetic_field_by.sizes['time']})
    # data_load = magnetic_field_bz.load()
    
    # B0 = 0.0
    # for direction in deck["fields"]:
    #     B0 += deck["fields"][direction]**2
    # B0 = np.sqrt(B0)

    # bkgd_density = deck["constant"]["background_density"]
    # bkgd_temp = deck["constant"]["background_temp"]
    # ion_mass = constants.proton_mass
    # ion_charge = constants.elementary_charge

    # if "deuteron" in deck["species"]:
    #     ion_mass = constants.proton_mass + constants.neutron_mass
    #     ion_charge *= deck["species"]["deuteron"]["charge"]
    
    # mass_density = (bkgd_density * ion_mass) + (bkgd_density * constants.electron_mass)
    # alfven_velo = B0 / (np.sqrt(constants.mu_0 * mass_density))
    # thermal_velo = np.sqrt((2.0 * constants.k * bkgd_temp) / ion_mass)
    
    # omega_pe_squared = (bkgd_density * constants.elementary_charge**2)/(constants.electron_mass * constants.epsilon_0)
    # omega_pi_squared = (bkgd_density * constants.elementary_charge**2)/(ion_mass * constants.epsilon_0)
    # omega_p_squared = omega_pe_squared + omega_pi_squared
    # TWO_PI = 2.0 * np.pi
    # ion_gyroperiod = (TWO_PI * ion_mass) / (ion_charge * B0)
    # Tci = data_load.coords["time"] / ion_gyroperiod
    # vA_Tci = data_load.coords["X_Grid_mid"] / (ion_gyroperiod * alfven_velo)
    # #vTh_over_Wci = ds.coords["X_Grid_mid"] * (TWO_PI / B0)
    # #Tci = np.array(ds.coords["time"]*B0/TWO_PI)
    # data_scaled = xr.DataArray(data_load, coords=[Tci, vA_Tci], dims=["time", "X_Grid_mid"])
    # orig_spec : xr.DataArray = xrft.xrft.fft(data_scaled, true_amplitude=True, true_phase=True)
    # spec = abs(orig_spec)
    # spec = spec.sel(freq_time=spec.freq_time>=0.0)
    # #spec = spec.sel(freq_time=spec.freq_time<=30.0)
    # spec = spec.sel(freq_X_Grid_mid=spec.freq_X_Grid_mid>=0.0)
    # spec = spec.sel(freq_X_Grid_mid=spec.freq_X_Grid_mid<=args.maxK)
    # print(f"Max: {np.max(spec)}")
    # print(f"Min: {np.min(spec)}")
    # spec.plot(norm=colors.LogNorm())
    # #spec.plot()
    # omega_A = spec.freq_X_Grid_mid

    # # Broken, fix
    # #omega_magnetosonic = ((spec.freq_X_Grid_mid * (ion_gyroperiod * alfven_velo))**2 * constants.speed_of_light**2 * ((thermal_velo**2 + alfven_velo**2)/(constants.speed_of_light**2 + alfven_velo**2))) / ion_gyroperiod
    # cVa = constants.speed_of_light / alfven_velo
    # plt.plot(spec.freq_X_Grid_mid, omega_A, "k--", label="w = kVa")
    # plt.plot(spec.freq_X_Grid_mid, spec.freq_X_Grid_mid * cVa, "r--", label="w = kc")
    # # plt.plot(spec.freq_X_Grid_mid, omega_magnetosonic, "k:")
    # plt.xlabel("Wavenumber [Wcp/Va]")
    # plt.ylabel("Frequency [Wcp]")
    # plt.title("Warm plasma dispersion relation")
    # plt.legend()
    # plt.show()