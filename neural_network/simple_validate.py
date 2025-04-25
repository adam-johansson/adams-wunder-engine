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

folder = "H2"
# Load the trained model
hidden_dim = 128
layers = 2
model = load_ANN(f"./models/{folder}_{hidden_dim}_{layers}.pth")
model = model.double()
print(model)

# import data

data = pd.read_csv("./input_data/" + folder + "/data.csv", index_col=0)
# X = pd.read_csv('./input_data/' + folder + '/x_cleaned.csv', index_col=0)
# y = pd.read_csv('./input_data/' + folder + '/y_cleaned.csv', index_col=0)

data = data[data.eff != 0]

# convert to numpy arrays
# X = pd.DataFrame.to_numpy(X)
# y = pd.DataFrame.to_numpy(y)
data = pd.DataFrame.to_numpy(data)


X = data[:, :8]
y = data[:, 8:]


# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=42
)


train_data = np.concatenate((X_train, y_train), axis=1)
test_data = np.concatenate((X_test, y_test), axis=1)

# convert to tensor
# X_test = torch.tensor(X_test, dtype=torch.float32)
# X_train = torch.tensor(X_train, dtype=torch.float32)

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

# mean absolute relative error
MRE_test = np.mean(np.abs(rel_test), axis=0)
MRE_train = np.mean(np.abs(rel_train), axis=0)

print(
    f"Test data % \n"
    f"MRE T_2: {MRE_test[0] * 100:.2f} % \n"
    f"MRE eff: {MRE_test[1] * 100:.2f} % \n"
    f"MRE airflow: {MRE_test[2]*100:.2f} % \n"
    f"MRE pmax: {MRE_test[3]*100:.2f} % \n"
    f"MRE T_max: {MRE_test[4]*100:.2f} % \n"
    f"MRE P: {MRE_test[5]*100:.2f} % \n"
    f"MRE Q : {MRE_test[6]*10:.2f} % \n"
    f"MRE p_tdc: {MRE_test[7]*100:.2f} %"
)

print(
    f"Training data % \n"
    f"MRE T_2: {MRE_train[0] * 100:.2f} % \n"
    f"MRE eff: {MRE_train[1] * 100:.2f} % \n"
    f"MRE airflow: {MRE_train[2]*100:.2f} % \n"
    f"MRE pmax: {MRE_train[3]*100:.2f} % \n"
    f"MRE T_max: {MRE_train[4]*100:.2f} % \n"
    f"MRE P: {MRE_train[5]*100:.2f} % \n"
    f"MRE Q : {MRE_train[6]*10:.2f} % \n"
    f"MRE p_tdc: {MRE_train[7]*100:.2f} %"
)


t2 = np.vstack((y_test[:, 0], y_test_hat[:, 0])).T
eff = np.vstack((y_test[:, 1], y_test_hat[:, 1])).T
airflow = np.vstack((y_test[:, 2], y_test_hat[:, 2])).T
pmax = np.vstack((y_test[:, 3], y_test_hat[:, 3])).T
tmax = np.vstack((y_test[:, 4], y_test_hat[:, 4])).T
P = np.vstack((y_test[:, 5], y_test_hat[:, 5])).T
Q = np.vstack((y_test[:, 6], y_test_hat[:, 6])).T
ptdc = np.vstack((y_test[:, 7], y_test_hat[:, 7])).T

# colour mapping for the plots
t2_colour = eff[t2[:, 0].argsort()]
airflow_colour = eff[airflow[:, 0].argsort()]
pmax_colour = eff[pmax[:, 0].argsort()]
tmax_colour = eff[tmax[:, 0].argsort()]
P_colour = eff[P[:, 0].argsort()]
Q_colour = eff[Q[:, 0].argsort()]
ptdc_colour = eff[ptdc[:, 0].argsort()]



t2 = t2[t2[:, 0].argsort()]
eff = eff[eff[:, 0].argsort()]
airflow = airflow[airflow[:, 0].argsort()]
pmax = pmax[pmax[:, 0].argsort()]
tmax = tmax[tmax[:, 0].argsort()]
P = P[P[:, 0].argsort()]
Q = Q[Q[:, 0].argsort()]
ptdc = ptdc[ptdc[:, 0].argsort()]


# look at data points that perform poorly
test_data = np.concatenate((X_test, y_test), axis=1)
problems_idx = np.abs(rel_test) > 0.5
problems_idx = np.any(problems_idx, axis=1)

problems = test_data[problems_idx]


# sweep of far and inlet temp
p_in = 10.5e5
T_in = 570
cr = 7
bore = 0.17
far = 0.02923 / 1.3
PI = 1.0
v_mean = 10.5
T_fuel = 430

tins = np.linspace(300, 1000, 1000)

powers1 = []
for T_in1 in tins:
    temp = model.inference(np.array([p_in, T_in1, cr, bore, far, PI, v_mean, T_fuel]))
    powers1.append(temp[0][5])

fars = np.linspace(0.02923 / 3.0, 0.02923 / 1.1, 1000)

powers2 = []
for far in fars:
    temp = model.inference(np.array([p_in, T_in, cr, bore, far, PI, v_mean, T_fuel]))
    powers2.append(temp[0][5])

powers1 = np.array(powers1)
powers2 = np.array(powers2)


fs = 24
figsize = (12, 8)
res = 50
dotsize = 2


fig, ax1 = plt.subplots(figsize=figsize)
cax = ax1.scatter(
    t2[:, 0],
    (t2[:, 1] - t2[:, 0]) / t2[:, 0] * 100,
    label="Prediction",
    c=t2_colour[:, 0],
    s=dotsize,
)
ax1.set_xlabel(r"Actual $T2$ [K]", fontsize=fs)
ax1.set_ylabel(r"Relative error [%]", fontsize=fs)
ax1.set_title(r"T2 relative error", fontsize=fs)
cbar = fig.colorbar(cax)

# ax2.set_xlim(660, 810)
# ax2.set_xticks([690, 720, 750, 780, 810])
# ax2.set_ylim(0, 55)
# ax2.set_yticks([0, 10, 20, 30, 40, 50])
ax1.tick_params(labelsize=fs)
plt.legend(loc="best", frameon=True, fontsize=fs)
# ax2.grid()
# plt.savefig('power_validation.pdf', dpi=res, bbox_inches='tight')

fig, ax2 = plt.subplots(figsize=figsize)
ax2.scatter(
    eff[:, 0],
    (eff[:, 1] - eff[:, 0]) / eff[:, 0] * 100,
    label="Prediction",
    color="r",
    s=dotsize,
)
ax2.set_xlabel(r"Actual $\eta$ [-]", fontsize=fs)
ax2.set_ylabel(r"Relative error [%]", fontsize=fs)
ax2.set_title(r"$\eta$ relative error", fontsize=fs)
ax2.tick_params(labelsize=fs)
plt.legend(loc="best", frameon=True, fontsize=fs)
ax2.grid()


fig, ax3 = plt.subplots(figsize=figsize)
cax3 = ax3.scatter(
    airflow[:, 0],
    (airflow[:, 1] - airflow[:, 0]) / airflow[:, 0] * 100,
    label="Prediction",
    c=airflow_colour[:, 0],
    s=dotsize,
)
ax3.set_xlabel(r"Actual airflow [kg/s]", fontsize=fs)
ax3.set_ylabel(r"Relative error [%]", fontsize=fs)
ax3.set_title(r"Airflow relative error", fontsize=fs)
ax3.tick_params(labelsize=fs)
plt.legend(loc="best", frameon=True, fontsize=fs)
cbar3 = fig.colorbar(cax3)

fig, ax4 = plt.subplots(figsize=figsize)
cax4 = ax4.scatter(
    pmax[:, 0] * 1e-5,
    (pmax[:, 1] - pmax[:, 0]) / pmax[:, 0] * 100,
    label="Prediction",
    c=pmax_colour[:, 0],
    s=dotsize,
)
ax4.set_xlabel(r"Actual $p_{max}$ [bar]", fontsize=fs)
ax4.set_ylabel(r"Relative error [%]", fontsize=fs)
ax4.set_title(r"$p_{max}$ relative error", fontsize=fs)
ax4.tick_params(labelsize=fs)
cbar4 = fig.colorbar(cax4)
plt.legend(loc="best", frameon=True, fontsize=fs)


fig, ax5 = plt.subplots(figsize=figsize)
cax5 = ax5.scatter(
    tmax[:, 0],
    (tmax[:, 1] - tmax[:, 0]) / tmax[:, 0] * 100,
    label="Prediction",
    c=tmax_colour[:, 0],
    s=dotsize,
)

ax5.set_xlabel(r"Actual $T_{max}$ [K]", fontsize=fs)
ax5.set_ylabel(r"Relative error [%]", fontsize=fs)
ax5.set_title(r"$T_{max}$ relative error", fontsize=fs)
ax5.tick_params(labelsize=fs)
cbar5 = fig.colorbar(cax5)
plt.legend(loc="best", frameon=True, fontsize=fs)

fig, ax6 = plt.subplots(figsize=figsize)
cax6 = ax6.scatter(
    P[:, 0] * 1e-3,
    (P[:, 1] - P[:, 0]) / P[:, 0] * 100,
    label="Prediction",
    c=P_colour[:, 0],
    s=dotsize,
)

ax6.set_xlabel(r"Actual $P_i$ [kW]", fontsize=fs)
ax6.set_ylabel(r"Relative error [%]", fontsize=fs)
ax6.set_title(r"$P_i$ relative error", fontsize=fs)
ax6.tick_params(labelsize=fs)
cbar6 = fig.colorbar(cax6)
plt.legend(loc="best", frameon=True, fontsize=fs)

fig, ax7 = plt.subplots(figsize=figsize)
cax7 = ax7.scatter(
    Q[:, 0] * 1e-3,
    (Q[:, 1] - Q[:, 0]) / Q[:, 0] * 100,
    label="Prediction",
    c=Q_colour[:, 0],
    s=dotsize,
)
ax7.set_xlabel(r"Actual $Q$ [kW]", fontsize=fs)
ax7.set_ylabel(r"Relative error [%]", fontsize=fs)
ax7.set_title(r"$Q$ relative error", fontsize=fs)
ax7.tick_params(labelsize=fs)
cbar7 = fig.colorbar(cax7)
plt.legend(loc="best", frameon=True, fontsize=fs)

fig, ax8 = plt.subplots(figsize=figsize)
cax8 = ax8.scatter(
    ptdc[:, 0] * 1e-5,
    (ptdc[:, 1] - ptdc[:, 0]) / ptdc[:, 0] * 100,
    label="Prediction",
    c=ptdc_colour[:,0],
    s=dotsize,
)
ax8.set_xlabel(r"Actual $p_{tdc}$ [bar]", fontsize=fs)
ax8.set_ylabel(r"Relative error [%]", fontsize=fs)
ax8.set_title(r"$p_{tdc}$ relative error", fontsize=fs)
ax8.tick_params(labelsize=fs)
cbar8 = fig.colorbar(cax8)
plt.legend(loc="best", frameon=True, fontsize=fs)


plt.show()

# plot the real values for power and the predicted one
fig, ax5 = plt.subplots()
ax5.plot(tins, powers1 * 1e-3)
ax5.set_xlabel(r"T_in [K]")
ax5.set_ylabel(r"P [kW]")
ax5.set_title(rf"Layers: {layers + 1}, neurons: {hidden_dim}, Training set")
# ax2.set_ylim(-10, 10)
# ax1.set_xlim(100, epochs)

# plot the real values for power and the predicted one
fig, ax6 = plt.subplots()
ax6.plot(fars, powers2 * 1e-3)
ax6.set_xlabel(r"far [-]")
ax6.set_ylabel(r"P [kW]")
ax6.set_title(rf"Layers: {layers + 1}, neurons: {hidden_dim}, Training set")
# ax2.set_ylim(-10, 10)
# ax1.set_xlim(100, epochs)


plt.show()
