from sdf_xarray import sdf
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import argparse

# Run python setup.py -h for list of possible arguments
parser = argparse.ArgumentParser("parser")
parser.add_argument(
    "--dir",
    action="store",
    help="Directory containing all sdf files from simulation.",
    required = True
)
args = parser.parse_args()

wd = args.dir
# strvx = "Particles/Vx/ion_ring_beam"
# strvy = "Particles/Vy/ion_ring_beam"
# strvz = "Particles/Vz/ion_ring_beam"
# data_b1 = sdf.read(wd + "/by_1.0/0000.sdf", dict=True)
# data_b7z3 = sdf.read(wd + "/by_0.7_bz_0.3/0000.sdf", dict=True)

# df = xr.open_dataset(wd + "/by_1.0/0099.sdf")
# print(df.keys())
# print(df[strvx])

# ds = xr.open_mfdataset(
#     wd + "/by_1.0/*.sdf",
#     concat_dim="time",
#     combine="nested",
#     data_vars='minimal', 
#     coords='minimal', 
#     compat='override', 
#     preprocess=SDFPreprocess()
# )

ds = xr.open_mfdataset(
    "*.sdf",
    combine="nested",
    data_vars='minimal', 
    coords='minimal', 
    compat='override', 
    preprocess=SDFPreprocess()
)

print(ds)

# Randomly sample some points to reduce plot size
n_samples = 1000
indices = np.random.choice(data_b7z3[strvx].data.shape[0], n_samples, replace=False) 
vx_sample_b7z3 = data_b7z3[strvx].data[indices]
vy_sample_b7z3 = data_b7z3[strvy].data[indices]
vz_sample_b7z3 = data_b7z3[strvz].data[indices]

vx_sample_b1 = data_b1[strvx].data[indices]
vy_sample_b1 = data_b1[strvy].data[indices]
vz_sample_b1 = data_b1[strvz].data[indices]

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(vx_sample_b7z3, vy_sample_b7z3, vz_sample_b7z3, label="By = 0.7, Bz = 0.3")
ax.scatter(vx_sample_b1, vy_sample_b1, vz_sample_b1, label="By = 1.0")
ax.axis('equal')
ax.set_xlabel('Vx')
ax.set_ylabel('Vy')
ax.set_zlabel('Vz')
plt.legend()
plt.show()

#data=sh.getdata('/home/era536/epoch/epoch1d/Data/ring_test/by_1.0/0000.sdf')
#sh.list_variables(data)

#sh.plot_auto(data.)

# variable = data.Electric_Field_Ex
# raw = variable.data
# print(raw)
# print(type(raw))
# print(np.mean(raw))

# sh.plot_auto(data.Electric_Field_Ex)
# plt.show()
