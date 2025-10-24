import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    QuantileTransformer,
)
import pandas as pd
from torcheval.metrics import R2Score
import numpy as np
import math

import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt

from src import load_ANN

## NN-PP: Neural Network Post Processing

folder = "jetA"
# Load the trained model
hidden_dim = 256

layers = 2
model = load_ANN(f"./models/{folder}_{hidden_dim}_{layers}.pth")
model = model.double()
print(model)

# import data

data = pd.read_csv("./input_data/" + folder + "/data.csv", index_col=0)

data = data[data.eff != 0]
data = data[pd.notna(data.eff)]


# Decide which data points should be input and output
X = data[['p_in', 'T_in', 'PI', 'cr', 'bore',  'v_mean', 'T_fuel', 'far_goal']]
#y = data[['power', 'heat_loss', 'nox', 'p_tdc', 'p_max', 'T_max', 'T_out', 'air_flow']]
y = data[['power', 'nox', 'p_tdc', 'p_max', 'T_out', 'T_max', 'air_flow']]



X_labels = X.columns.tolist()
y_labels = y.columns.tolist()

# convert to numpy arrays
X = pd.DataFrame.to_numpy(X)
y = pd.DataFrame.to_numpy(y)




# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=42
)


train_data = np.concatenate((X_train, y_train), axis=1)
test_data = np.concatenate((X_test, y_test), axis=1)

train_data = np.concatenate((X_train, y_train), axis=1)
test_data = np.concatenate((X_test, y_test), axis=1)


y_test_hat = np.zeros((X_test.shape[0], y_test.shape[1]))
y_train_hat = np.zeros((X_train.shape[0], y_train.shape[1]))

for i in range(X_test.shape[0]):
    # y_test_hat.append(model(X_test[i, :]).detach().numpy()) #for tensor
    y_test_hat[i, :] = model.inference(X_test[i, :])


for i in range(X_train.shape[0]):
    # y_train_hat.append(model(X_train[i, :]).detach().numpy())
    y_train_hat[i, :] = model.inference(X_train[i, :])


plt.plot(y_test[:, 5], y_test[:, 5])
plt.scatter(y_test[:, 5], y_test_hat[:, 5], s=4)
plt.show()


# root square error
RSE_test = np.sqrt(np.square(np.subtract(y_test, y_test_hat)))
RSE_train = np.sqrt(np.square(np.subtract(y_train, y_train_hat)))

# relative error
rel_test = np.divide(RSE_test, y_test)
rel_train = np.divide(RSE_train, y_train)

print(np.argwhere(np.isnan(rel_train)))

# mean absolute relative error
MRE_test = np.mean(np.abs(rel_test), axis=0)
MRE_train = np.mean(np.abs(rel_train), axis=0)

print("Test data %")
for i, label in enumerate(y_labels):
    print(f"MRE {label}: {MRE_test[i] * 100:.2f} %")

print("Train data %")
for i, label in enumerate(y_labels):
    print(f"MRE {label}: {MRE_train[i] * 100:.2f} %")


# Create data arrays dynamically
n_outputs = y_test.shape[1]
data_arrays = {}
colour_arrays = {}

# Stack actual and predicted values for each output
for i in range(n_outputs):
    var_name = y_labels[i] if i < len(y_labels) else f"var_{i}"
    data_arrays[var_name] = np.vstack((y_test[:, i], y_test_hat[:, i])).T

# Create colour mapping (using efficiency as reference - index 1)
eff_data = data_arrays[y_labels[1]]  # Assuming efficiency is at index 1
for var_name, data in data_arrays.items():
    colour_arrays[var_name] = eff_data[data[:, 0].argsort()]

# Sort arrays by actual values
for var_name in data_arrays.keys():
    data_arrays[var_name] = data_arrays[var_name][data_arrays[var_name][:, 0].argsort()]

# Look at data points that perform poorly
test_data = np.concatenate((X_test, y_test), axis=1)
problems_idx = np.abs(rel_test) > 0.5
problems_idx = np.any(problems_idx, axis=1)
problems = test_data[problems_idx]

# Plot settings
fs = 24
figsize = (12, 8)
res = 50
dotsize = 2

# Define plot configurations for each variable
plot_configs = {
    y_labels[0]: {'xlabel': r"Actual $T2$ [K]", 'title': r"T2 relative error", 'scale': 1, 'use_colorbar': True},
    y_labels[1]: {'xlabel': r"Actual $\eta$ [-]", 'title': r"$\eta$ relative error", 'scale': 1, 'use_colorbar': False,
                  'color': 'r'},
    'airflow': {'xlabel': r"Actual airflow [kg/s]", 'title': r"Airflow relative error", 'scale': 1,
                'use_colorbar': True},
    'pmax': {'xlabel': r"Actual $p_{max}$ [bar]", 'title': r"$p_{max}$ relative error", 'scale': 1e-5,
             'use_colorbar': True},
    'tmax': {'xlabel': r"Actual $T_{max}$ [K]", 'title': r"$T_{max}$ relative error", 'scale': 1, 'use_colorbar': True},
    'P': {'xlabel': r"Actual $P_i$ [kW]", 'title': r"$P_i$ relative error", 'scale': 1e-3, 'use_colorbar': True},
    'Q': {'xlabel': r"Actual $Q$ [kW]", 'title': r"$Q$ relative error", 'scale': 1e-3, 'use_colorbar': True},
    'ptdc': {'xlabel': r"Actual $p_{tdc}$ [bar]", 'title': r"$p_{tdc}$ relative error", 'scale': 1e-5,
             'use_colorbar': True},
    'nox': {'xlabel': r"Actual NOx [ppm]", 'title': r"NOx relative error", 'scale': 1, 'use_colorbar': True}
}

# Create plots dynamically
for i, (var_name, data) in enumerate(data_arrays.items()):
    # Use variable name if it exists in plot_configs, otherwise use a default
    if var_name in plot_configs:
        config = plot_configs[var_name]
    else:
        config = {'xlabel': f"Actual {var_name}", 'title': f"{var_name} relative error", 'scale': 1,
                  'use_colorbar': True}

    fig, ax = plt.subplots(figsize=figsize)

    # Calculate relative error
    x_data = data[:, 0] * config['scale']
    rel_error = (data[:, 1] - data[:, 0]) / data[:, 0] * 100

    # Create scatter plot
    if config.get('use_colorbar', True) and var_name in colour_arrays:
        scatter = ax.scatter(
            x_data,
            rel_error,
            label="Prediction",
            c=colour_arrays[var_name][:, 0],
            s=dotsize,
        )
        fig.colorbar(scatter)
    else:
        color = config.get('color', 'blue')
        ax.scatter(
            x_data,
            rel_error,
            label="Prediction",
            color=color,
            s=dotsize,
        )

    # Set labels and title
    ax.set_xlabel(config['xlabel'], fontsize=fs)
    ax.set_ylabel(r"Relative error [%]", fontsize=fs)
    ax.set_title(config['title'], fontsize=fs)
    ax.tick_params(labelsize=fs)
    plt.legend(loc="best", frameon=True, fontsize=fs)

    # Add grid for efficiency plot (index 1)
    if i == 1:  # Efficiency plot
        ax.grid()

plt.show()
