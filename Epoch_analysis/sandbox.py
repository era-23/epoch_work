import numpy as np
import matplotlib.pyplot as plt
from plasmapy.formulary import frequencies as ppf
from plasmapy.formulary import speeds as pps
from plasmapy.formulary import lengths as ppl
from scipy import constants as constants
import astropy.units as u

def cold_plasma_dispersion_relation():
    # Constants
    B0 = 0.5  * u.T  # Background magnetic field (T)
    n_e = 1e19 * u.m**-3  # Electron density (m^-3)
    n_i = n_e  # Assume quasi-neutrality
    TWO_PI_RAD = 2.0 * np.pi * u.rad

    # Cyclotron frequencies
    Omega_i = ppf.gyrofrequency(B0, "p+")   # Ion cyclotron frequency rad/s
    Omega_e = ppf.gyrofrequency(B0, "e-")  # Electron cyclotron frequency rad/s
    normalised_frequency = Omega_e
    Omega_i_norm = Omega_i / normalised_frequency
    Omega_e_norm = Omega_e / normalised_frequency
    #Tau_i = (1.0 / Omega_i)
    v_A = pps.Alfven_speed(B0, n_i, "p+")  # Alfvén speed m/s

    c = constants.c * (u.m/u.s)

    # Plasma frequencies
    omega_pe = ppf.plasma_frequency(n_e, "e-")  # Electron plasma frequency rad/s
    omega_pe_norm = omega_pe / normalised_frequency
    omega_pi = ppf.plasma_frequency(n_e, "p+")  # Ion plasma frequency rad/s
    omega_pi_norm = omega_pi / normalised_frequency

    # Hybrid frequencies
    omega_uh = ppf.upper_hybrid_frequency(B0, n_e)  # Upper hybrid frequency rad/s
    omega_uh_norm = omega_uh / normalised_frequency
    omega_lh = ppf.lower_hybrid_frequency(B0, n_i, "p+")  # Lower hybrid frequency rad/s
    omega_lh_norm = omega_lh / normalised_frequency

    # Wavenumber range
    simLength = 0.0001 * u.m
    #normalised_wavenumber = normalised_frequency / c
    k_vals = np.linspace(0.0, TWO_PI_RAD/simLength, 100000)# Wavenumber range
    k_vals_norm = (k_vals * c) / normalised_frequency

    # O-mode dispersion (unmagnetized plasma mode)
    #omega_o_mode = np.sqrt(omega_pe**2 + (c * k_vals)**2)

    # X-mode dispersion
    #omega_x_mode = np.sqrt(0.5 * (omega_pe**2 + Omega_e**2) +
    #                        0.5 * np.sqrt((omega_pe**2 + Omega_e**2)**2 - 4 * Omega_e**2 * omega_pe**2))

    # Whistler mode (parallel propagation)
    omega_whistler = (k_vals**2 * c**2 * Omega_e) / omega_pe**2
    omega_whistler_norm = omega_whistler / normalised_frequency

    # Ion cyclotron mode
    #omega_ion_cyclotron = Omega_i / Omega_i  # Ion cyclotron resonance (constant)

    # Lower hybrid wave mode
    #omega_lower_hybrid = np.ones_like(k_vals) * omega_lh  # Constant lower hybrid frequency

    # Alfvén wave dispersion (low k approximation: ω = k * v_A)
    omega_alfven = k_vals * v_A  # Alfvén wave linear dispersion
    omega_alfven_norm = omega_alfven / normalised_frequency  # Alfvén wave linear dispersion

    omega_light = k_vals * c
    omega_light_norm = omega_light / normalised_frequency

    # Plot dispersion relations
    plt.figure(figsize=(8, 6))

    #plt.plot(k_vals, omega_o_mode, label="O-mode (Ordinary)", color="blue")
    #plt.axhline(y=omega_pe_norm.value, color="blue", linestyle="dashed", label=r"$\omega_{pe}$")
    #plt.axhline(y=omega_pi_norm.value, color="cyan", linestyle="dashed", label=r"$\omega_{pi}$")

    #plt.plot(k_vals, omega_x_mode * np.ones_like(k_vals), label="X-mode (Extraordinary)", color="red")
    #plt.axhline(y=omega_uh_norm.value, color="red", linestyle="dashdot", label=r"$\omega_{UH}$")
    #plt.axhline(y=omega_lh_norm.value, color="orange", linestyle="dashdot", label=r"$\omega_{LH}$ (Lower Hybrid)")

    plt.axhline(y=Omega_i_norm.value, color="purple", linestyle="dashed", label=r"$\Omega_i$ (Ion Cyclotron)")
    plt.axhline(y=Omega_e_norm.value, color="brown", linestyle="dashed", label=r"$\Omega_e$ (Electron Cyclotron)")

    plt.plot(k_vals_norm, omega_whistler_norm, label="Whistler Mode", color="green")
    plt.plot(k_vals_norm, omega_alfven_norm, label="Alfvén Wave", color="black", linestyle="dotted")
    plt.plot(k_vals_norm, omega_light_norm, label="Light Wave", color="black", linestyle="dotted")

    plt.xlabel("Wavenumber k (ω_ce/c)")
    plt.ylabel("Frequency ω (ω_ce)")
    plt.title("Cold Plasma Dispersion Relations with Ion Modes")
    plt.legend()
    #plt.yscale("log")
    #plt.xscale("log")
    plt.grid()
    plt.ylim(1E-4, 100)
    plt.xlim(1E-2, 100)
    plt.show()

def X(omega, omega_pe):
    return omega_pe**2 / omega**2

def Y(omega, omega_ce):
    return omega_ce / omega

def appleton_hartree():
    # https://en.wikipedia.org/wiki/Appleton%E2%80%93Hartree_equation

    # Constants
    B0 = 0.5 * u.T  # Background magnetic field (T)
    n_e = 1e18 * u.m**-3  # Electron density (m^-3)
    n_i = n_e  # Assume quasi-neutrality

    omega_pe = ppf.plasma_frequency(n_e, "e-").value
    omega_ce = ppf.gyrofrequency(B0, "e-").value
    omega_ci = ppf.gyrofrequency(B0, "p-").value
    #theta = 0.0 # Angle between k and B
    theta = np.pi / 2.0 # Angle between k and B

    omega_uh = ppf.upper_hybrid_frequency(B0, n_e).value
    omega_lh = ppf.lower_hybrid_frequency(B0, n_i, 'p+').value
    max_omega = 1.5 * omega_uh
    omega_vals = np.linspace(omega_lh, max_omega, 100000)

    # omega_pe^2 * (1 - X)
    numerator = 1.0 * X(omega_vals, omega_pe)

    # Part before +/-: 1 - X - (0.5Y^2 * sin^2(theta))
    denominator_1 = 1.0 - X(omega_vals, omega_pe) - (0.5 * Y(omega_vals, omega_ce)**2 * np.sin(theta)**2)

    # Part after +/-: sqrt((0.5Y^2 * sin^2(theta))^2 + ((1-X)^2 * Y^2 * cos^2(theta)))
    denominator_2 = np.sqrt(
        (0.5 * Y(omega_vals, omega_ce)**2 * np.sin(theta)**2)**2 
        + ((1.0 - X(omega_vals, omega_pe))**2 * Y(omega_vals, omega_ce)**2 * np.cos(theta)**2)
    )

    n_squared_plus = 1.0 - (numerator / (denominator_1 + denominator_2))
    n_squared_minus = 1.0 - (numerator / (denominator_1 - denominator_2))

    plt.axvline(omega_uh, color = 'black', linestyle = 'dotted', label = "Upper hybrid")
    plt.axvline(omega_lh, color = 'black', linestyle = 'dashed', label = "Lower hybrid")
    plt.axvline(omega_ce, color = 'blue', linestyle = 'dotted', label = "Electron gyrofrequency")
    plt.axvline(omega_pe, color = 'blue', linestyle = 'dashed', label = "Electron plasma frequency")
    plt.title(f"theta = {theta * 180.0 / np.pi} degrees")
    plt.plot(omega_vals, n_squared_plus, color = 'red', label="n^2 +")
    plt.plot(omega_vals, n_squared_minus, color = 'green', label="n^2 -")
    plt.xlabel("omega")
    plt.ylabel("refractive index^2 [(kc/omega)^2]")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    #plt.ylim(0.0, max_omega)
    #plt.xlim(np.min([np.min(k_squared_plus), np.min(k_squared_minus)]), np.max([np.max(k_squared_plus), np.max(k_squared_minus)]))
    #plt.xscale('log')
    #plt.yscale('log')
    plt.show()

def trigonometry():

    theta = np.linspace(0.0, 2.0 * np.pi, 10000)

    me_over_mi = constants.electron_mass / constants.proton_mass
    trig_func = np.cos(theta)**2

    crosses = []
    below = False
    index = 0

    for y in trig_func:
        if below:
            if y >= me_over_mi:
                crosses.append(theta[index])
                below = False
        else:
            if y <= me_over_mi:
                crosses.append(theta[index])
                below = True
        index += 1

    print(f"Crossing points: {crosses}")
    print(f"cos^2 (theta) is <= me/mi from {crosses[0]} ({crosses[0] * 180/np.pi}deg) to {crosses[1]} ({crosses[1] * 180/np.pi}deg) and {crosses[2]} ({crosses[2] * 180/np.pi}deg) to {crosses[3]} ({crosses[3] * 180/np.pi}deg)")

    plt.plot(theta, trig_func, "r")
    plt.axhline(me_over_mi, linestyle="dashed", color="blue", label = "me/mi")
    plt.xlabel("Theta/rad")
    plt.legend()
    plt.ylabel("cos^2 (theta)")
    plt.show()

    # Function is <= me/mi from 1.5477rad - 1.5942rad (88.677 - 91.341)

def verdon_et_al():
    # Follows equ 2.1 and 2.2 in: https://www.cambridge.org/core/services/aop-cambridge-core/content/view/S1743921309029871

    # Constants
    B0 = 0.5 * u.T  # Background magnetic field (T)
    n_e = 1e18 * u.m**-3  # Electron density (m^-3)
    n_i = n_e  # Assume quasi-neutrality
    thetaDeg_all = np.linspace(0.0, 180.0, 2000)
    theta_all = thetaDeg_all * (np.pi / 180.0)
    k_max = 2000.0
    k = np.linspace(0.00001, k_max, 100000)
    k_norm_factor =  ppf.gyrofrequency(B0, "p+").value / pps.Alfven_speed(B0, n_i, "p+").value
    normalised_k = k / k_norm_factor
    all_intersections = []

    for theta in theta_all:
        # 2.1: plasma is cold but incluidng EM effects
        omega_pe = ppf.plasma_frequency(n_e, "e-").value
        w_squared_over_wLH_squared = ((1.0 / (1.0 + (omega_pe**2 / (k**2 * constants.c**2))))
                                    * (1.0 + ((constants.proton_mass / constants.electron_mass) * (np.cos(theta)**2 / (1.0 + (omega_pe**2 / (k**2 * constants.c**2))))))
                                    )
        
        diff = abs(w_squared_over_wLH_squared - 1.0)
        index = np.argmin(diff)
        all_intersections.append(normalised_k[index])

    plt.plot(thetaDeg_all, all_intersections)
    plt.xlabel("Theta/degrees")
    plt.ylabel(r"k at $\omega = \pm\omega_{LH}$")
    plt.title(r"Intersections of k and $\omega = \pm\omega_{LH}$, by angle")
    plt.show()

    midpoint_index = int(len(thetaDeg_all)/2)
    plt.plot(thetaDeg_all[midpoint_index-29:midpoint_index+30], all_intersections[midpoint_index-29:midpoint_index+30])
    plt.xlabel("Theta/degrees")
    plt.grid()
    plt.ylabel(r"k at $\omega = \pm\omega_{LH}$")
    plt.title(r"Intersections of k and $\omega = \pm\omega_{LH}$, near theta = 90")
    plt.show()

    # plt.title(f"Theta = {thetaDeg}deg")
    # plt.axhline(1.0, color = "gray", linestyle="dotted", label = r"$\omega = \pm\omega_{LH}$")
    # plt.axvline(intersection_1, color = "green", linestyle="dashed", label = f"k = {intersection_1:.3f}")
    # plt.axvline(intersection_2, color = "green", linestyle="dashed", label = f"k = {intersection_2:.3f}")
    # plt.plot(normalised_k, w_squared_over_wLH_squared)
    # plt.legend()
    # plt.xlabel(r"$kV_A/\omega_i$")
    # plt.ylabel(r"$\omega^2/\omega_{LH}^2$")
    # plt.show()

    # 2.2: plasma is warm but assumes wave is electrostatic/longitudinal


if __name__ == "__main__":

    #cold_plasma_dispersion_relation()
    #appleton_hartree()
    #trigonometry()
    verdon_et_al()