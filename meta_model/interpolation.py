import pandas as pd
from scipy.interpolate import RBFInterpolator
import numpy as np

xt = pd.read_csv('../piston_engine/surrogate_data/backup/h2_validated_woschni_18/xt_cleaned.csv', index_col=0)
xval = pd.read_csv('../piston_engine/surrogate_data/backup/h2_validated_woschni_18/xval_cleaned.csv', index_col=0)
yt = pd.read_csv('../piston_engine/surrogate_data/backup/h2_validated_woschni_18/yt_cleaned.csv', index_col=0)
yval = pd.read_csv('../piston_engine/surrogate_data/backup/h2_validated_woschni_18/yval_cleaned.csv', index_col=0)

xt = pd.DataFrame.to_numpy(xt)
xval = pd.DataFrame.to_numpy(xval)
yt = pd.DataFrame.to_numpy(yt)
yval = pd.DataFrame.to_numpy(yval)

# take less training points
xt = xt[::16, :]
yt = yt[::16, :]
xval = xval[::16, :]
yval = yval[::16, :]

"""
print(yval[:, 0])  # T_out [K]
print(yval[:, 1])  # eta_th [-]
print(yval[:, 2])  # air_flow [kg/s]
print(yval[:, 3])  # p_max [bar]
print(yval[:, 4])  # T_max [K]
print(yval[:, 5])  # indicated power [kW]
print(yval[:, 6])  # heat loss [kW]
print(yval[:, 7])  # p_tdc [bar]
"""


nr = 1000
x0_min = np.min(xt[:, 0])
x0_max = np.max(xt[:, 0])
x1_min = np.min(xt[:, 1])
x1_max = np.max(xt[:, 1])
x2_min = np.min(xt[:, 2])
x2_max = np.max(xt[:, 2])
x3_min = np.min(xt[:, 3])
x3_max = np.max(xt[:, 3])
x4_min = np.min(xt[:, 4])
x4_max = np.max(xt[:, 4])
x5_min = np.min(xt[:, 5])
x5_max = np.max(xt[:, 5])
x6_min = np.min(xt[:, 6])
x6_max = np.max(xt[:, 6])

x0_points = np.linspace(x0_min, x0_max, nr)  # p_in
x1_points = np.linspace(x1_min, x1_max, nr)  # T_in
x2_points = np.linspace(x2_min, x2_max, nr)  # cr
x3_points = np.linspace(x3_min, x3_max, nr)  # bore
x4_points = np.linspace(x4_min, x4_max, nr)  # far
x5_points = np.linspace(x5_min, x5_max, nr)  # p_ratio
x6_points = np.linspace(x6_min, x6_max, nr)  # v_mean
interp_points = (x0_points, x1_points, x2_points, x3_points, x4_points, x5_points, x6_points)
xt_points = (xt[:, 0], xt[:, 1], xt[:, 2], xt[:, 3], xt[:, 4], xt[:, 5], xt[:, 6])
yt_1 = yt[:, 0]

# should rescale data to 0-1


#xgrid = np.mgrid[x0_min:x0_max:50j, x1_min:x1_max:50j, x2_min:x2_max:50j, x3_min:x3_max:50j, x4_min:x4_max:50j,
#        x5_min:x5_max:50j, x6_min:x6_max:50j]
#xflat = xgrid.reshape(2, -1).T
#print(xflat)
yflat = RBFInterpolator(xt, yt_1)
xtest = np.array([3e5, 500, 5, 0.1, 0.02, 1.0, 10]).T
print(yflat(xtest))
#ygrid = yflat.reshape(50, 50)

# validate

