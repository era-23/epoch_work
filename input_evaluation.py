import numpy as np
from scipy import constants

#Conversions
kb = constants.physical_constants["Boltzmann constant in eV/K"][0]

##### Constants block
background_density = 1e19
background_temp = 1e8 # K not eV
frac_beam = 1e-3
ion_mass_e = 1836.2
mass_ion = constants.electron_mass * ion_mass_e
ring_beam_energy = 1.0e6
pring_beam = np.sqrt(2.0 * mass_ion * constants.elementary_charge * ring_beam_energy)
pbeam = pring_beam * np.sin(-np.pi/3.0)
pring = pring_beam * np.cos(-np.pi/3.0)

p_ring_th = pring_beam / 100.0
p_beam_th = pring_beam / 1000.0

px_min_limit = np.max([pring - 6.0*p_ring_th, 0]) * (pring - 6.0*p_ring_th)
px_peak = pring / 2.0 * (1.0 + np.sqrt(1.0 + (2.0*p_ring_th/pring)**2.0))

vel_ion = np.sqrt(2.0 * constants.k * background_temp / mass_ion)

CPDR_bkgd_temp = 1.16e7
CPDR_bkgd_dens = 1e19
CPDR_lambda_db = np.sqrt(constants.epsilon_0 * constants.k * CPDR_bkgd_temp / CPDR_bkgd_dens / constants.elementary_charge**2.0)
CPDR_grid_spac = CPDR_lambda_db * 0.95
CPDR_nxgrid = 28500
CPDR_x_length = CPDR_grid_spac * CPDR_nxgrid

b0_strength = 3.0
# Angle between B and x (predominantly in Z)
b0_angle = 90

# Check resolutions:
lambda_db = np.sqrt((constants.epsilon_0 * constants.k * background_temp) / (background_density * constants.elementary_charge**2)) # 0.000218
grid_spacing = lambda_db * 0.5 # 0.000109

ion_gyrofrequency = (constants.elementary_charge * b0_strength) / (2.0 * np.pi * mass_ion) # 45734380.5778
ion_gyroperiod = 1.0 / ion_gyrofrequency # 2.186538851878323e-08
bkgd_mass_density = mass_ion * background_density * (1.0 - frac_beam) # 1.6709923702341605e-08
alfven_velo = b0_strength / np.sqrt(constants.mu_0 * bkgd_mass_density) # 20702800.119

#    Space
resolution_pixels = 10
minimum_num_cells_for_resolution = 2.0 * np.pi * resolution_pixels * (alfven_velo / (lambda_db * ion_gyrofrequency)) # 130334.846
minimum_nyquist_k = 40.0 * (ion_gyrofrequency / alfven_velo) # 88.36
nyquist_k = 1.0 / lambda_db # 4582.414
# Check nyquist_k > minimum_nyquist_k
print(nyquist_k > minimum_nyquist_k) # True
num_cells = np.ceil(minimum_num_cells_for_resolution) # 130335
domain_length = num_cells * grid_spacing # 14.2212

#    Time
simtime = 20.0 * ion_gyroperiod # 4.373077703756646e-07
minimum_timepoints_for_resolution = resolution_pixels * (simtime / ion_gyroperiod) # 200.0
nyquist_omega = 40.0 * ion_gyrofrequency # 1829375223.1129313
minimum_timepoints_for_nyquist = 2.0 * nyquist_omega * simtime # 1600.0
num_time_samples = np.ceil(np.max([minimum_timepoints_for_resolution, minimum_timepoints_for_nyquist])) # 1600.0

diagtime = simtime / num_time_samples # 2.733173564847904e-10

##### Control block
nx = num_cells
npart = 100 * nx

# final time of simulation
t_end = simtime

# size of domain
x_min = 0.0
x_max = domain_length

##### Fields block
bx = b0_strength * np.cos(b0_angle * np.pi/180)
by = 0.0
bz = b0_strength * np.sin(b0_angle * np.pi/180)

##### Species: proton
p_name = "proton"
p_charge = 1.0
p_mass = ion_mass_e
p_frac = 0.3
p_number_density = background_density * (1.0 - frac_beam)
p_temp = background_temp

##### Species: electron
e_name = "electron"
e_charge = -1.0
e_mass = 1.0
e_frac = 0.3
e_temp = background_temp / 100
e_number_density = background_density

##### Species: ion ring beam
irb_name = "ion_ring_beam"
irb_charge = 1.0
irb_mass = ion_mass_e
irb_frac = 0.4
irb_number_density = background_density * frac_beam

# px not defined, bespoke handling in EPOCH
#dist_fn = px / px_peak * np.exp(-0.5 * ((px - pring) / p_ring_th)**2) \
#        * np.exp(-0.5 * ((pz - pbeam) / p_beam_th)**2)
dist_fn_px_range = (px_min_limit, pring + 6.0*p_ring_th)
# deliberately ignore this dist_fn_py_range
# dist_fn_py_range = (0, 0)
dist_fn_pz_range = (pbeam - 6*p_ring_th, pbeam + 6.0*p_ring_th)
field_aligned_initialisation = True
x_perp_y_ignored_z_para = True

##### Output block
# number of timesteps between output dumps
dt_snapshot = diagtime
# Number of dt_snapshot between full dumps
full_dump_every = 100
force_final_to_be_restartable = True

# Properties at particle positions
# bespoke handling in EPOCH
# particles = full
# particle_weight = full
# vx = full
# vy = full
# vz = full

# Properties on grid
# bespoke handling in EPOCH
# grid = always
# ex = always
# ey = always
# ez = always
# by = always
# bz = always
# ekbar = always
# charge_density = full
# number_density = always + species

# Extended IO
# distribution_functions = always

##### Dist_fn block
# All bespoke handling in EPOCH
# pypz_name = py_pz
# pypz_ndims = 2

# pypz_direction1 = dir_py
# pypz_direction2 = dir_pz

# # range is ignored for spatial coordinates
# range1 = (1, 1)
# range2 = (-3.0e-22, 3.0e-22)

# # resolution is ignored for spatial coordinates
# resolution1 = 100
# resolution2 = 100

# include_species:electron
# include_species:proton
# include_species:ion_ring_beam

# ##### Dist_fn block
# name = px_py
# ndims = 2

# direction1 = dir_px
# direction2 = dir_py

# # range is ignored for spatial coordinates
# range1 = (1, 1)
# range2 = (-3.0e-22, 3.0e-22)

# # resolution is ignored for spatial coordinates
# resolution1 = 100
# resolution2 = 100

# include_species:electron
# include_species:proton
# include_species:ion_ring_beam

# ##### Dist_fn block
# name = px
# ndims = 1

# direction1 = dir_px

# # range is ignored for spatial coordinates
# range1 = (-3.0e-22, 3.0e-22)

# # resolution is ignored for spatial coordinates
# resolution1 = 100

# include_species:electron
# include_species:proton
# include_species:ion_ring_beam

for v in dir():
    if not v.startswith("__"): 
        print(f"{v}: {eval(v)}")