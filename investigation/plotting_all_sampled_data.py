import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
from scipy import stats


X = pd.read_csv("./../neural_network/input_data/H2/x_cleaned.csv", index_col=0)
y = pd.read_csv("./../neural_network/input_data/H2/y_cleaned.csv", index_col=0)

# convert to numpy arrays
X = pd.DataFrame.to_numpy(X)
y = pd.DataFrame.to_numpy(y)


data = np.concatenate((X, y), axis=1)


labels = [
    "p_in",
    "T_in",
    "cr",
    "bore",
    "far",
    "pressure ratio",
    "v_mean",
    "T_fuel",
    "T_out",
    "eff",
    "air_flow",
    "p_max",
    "T_max",
    "power",
    "heat loss",
    "p_tdc",
]
# 0 = p_in
# 1 = T_in
# 2 = cr
# 3 = bore
# 4 = far
# 5 = pressure ratio
# 6 = v_mean
# 7 = T_fuel
# 8 = T_out
# 9 = eff
# 10 = air_flow
# 11 = p_max
# 12 = T_max
# 13 = power
# 14 = heat loss
# 15 = p_tdc
independent = 4
dependent = 13

data = data[data[:, independent].argsort()]


sigma = 8


fig, ax = plt.subplots()
fig.suptitle("All sampled data points")


ax.plot(data[:, independent], data[:, dependent], "k.", markersize=1, label="raw data")
bin_means, bin_edges, binnumber = stats.binned_statistic(
    data[:, independent], data[:, dependent], statistic="mean", bins=100
)
ax.hlines(
    bin_means, bin_edges[:-1], bin_edges[1:], colors="g", lw=5, label="binned average"
)
ax.set_xlabel(labels[independent])
ax.set_ylabel(labels[dependent])
ax.set_title("Scatter plot")


plt.legend()
plt.show()
