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

# Load the trained model
hidden_dim = 128
layers = 1
model = load_ANN(f"./models/straight_{hidden_dim}_{layers}.pth")
print(model)

X = pd.read_csv("./input_data/H2_mediumnarrow/x.csv", index_col=0)
y = pd.read_csv("./input_data/H2_mediumnarrow/y.csv", index_col=0)

# convert to numpy arrays
X = pd.DataFrame.to_numpy(X)
y = pd.DataFrame.to_numpy(y)


# only look at power
y = y[:, [5]]

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=42
)


train_data = np.concatenate((X_train, y_train), axis=1)
test_data = np.concatenate((X_test, y_test), axis=1)

# normalise the data, fit the normalisation on training data
# we dont have to scale since it is done in the inference model
scaler = "3"

if scaler == "1":

    X_mean = np.mean(X_train, 0)
    X_std = np.std(X_train, 0)

    y_mean = np.mean(y_train, 0)
    y_std = np.std(y_train, 0)

    # normalise the training data
    X_train = (X_train - X_mean) / X_std
    y_train = (y_train - y_mean) / y_std

    # normalise the test data
    X_test = (X_test - X_mean) / X_std
    y_test = (y_test - y_mean) / y_std

elif scaler == "2":
    # scale between 0 and 1
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)

    y_min = y_train.min(axis=0)
    y_max = y_train.max(axis=0)

    X_train = (X_train - x_min) / (x_max - x_min)
    y_train = (y_train - y_min) / (y_max - y_min)

    X_test = (X_test - x_min) / (x_max - x_min)
    y_test = (y_test - y_min) / (y_max - y_min)


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


if scaler == "1":
    # rescale from std 1 mean 0
    y_test = y_test * y_std + y_mean
    y_test_hat = y_test_hat * y_std + y_mean
    y_train = y_train * y_std + y_mean
    y_train_hat = y_train_hat * y_std + y_mean

elif scaler == "2":
    # rescale from 0 to 1
    # X_scaled = X_std * (X_train.max(axis=0) - X_train.min(axis=0)) + X_train.min(axis=0)
    y_test = y_test * (y_max - y_min) + y_min
    y_test_hat = y_test_hat * (y_max - y_min) + y_min
    y_train = y_train * (y_max - y_min) + y_min
    y_train_hat = y_train_hat * (y_max - y_min) + y_min


# root square error
RSE_test = np.sqrt(np.square(np.subtract(y_test, y_test_hat)))
RSE_train = np.sqrt(np.square(np.subtract(y_train, y_train_hat)))

# relative error
rel_test = np.divide(RSE_test, y_test)
rel_train = np.divide(RSE_train, y_train)

# mean absolute relative error
MRE_test = np.mean(np.abs(rel_test), axis=0)
MRE_train = np.mean(np.abs(rel_train), axis=0)


print(f"Training data % \n" f"MRE power: {MRE_train[0] * 100:.2f} %")

print(f"Testdata % \n" f"MRE power: {MRE_test[0] * 100:.2f} %")


y_both = np.hstack((y_test, y_test_hat))
y_both_train = np.hstack((y_train, y_train_hat))

rel_error = (y_both[:, 1] - y_both[:, 0]) / y_both[:, 0]
rel_error_train = (y_both_train[:, 1] - y_both_train[:, 0]) / y_both_train[:, 0]

problems_idx = np.abs(rel_error) > 0.1
problems_idx_train = np.abs(rel_error_train) > 0.1

problems = test_data[problems_idx]
problems_train = train_data[problems_idx_train]


# sort from smallest to biggest power, for plotting
y_both = y_both[y_both[:, 0].argsort()]

y_both_train = y_both_train[y_both_train[:, 0].argsort()]

# relative error, sorted
rel_error = (y_both[:, 1] - y_both[:, 0]) / y_both[:, 0]
rel_error_train = (y_both_train[:, 1] - y_both_train[:, 0]) / y_both_train[:, 0]

print(f"Average rel error: {np.mean(np.abs(rel_error))} \n")
print(f"Average rel error training: {np.mean(np.abs(rel_error_train))}")

points = np.arange(1, y_test.shape[0] + 1, 1)
points_train = np.arange(1, y_train.shape[0] + 1, 1)


# sweep of far and inlet temp
p_in = 10.5e5
T_in = 570
cr = 7
bore = 0.17
far = 0.02923 / 1.3
PI = 1.0
v_mean = 10.5
T_fuel = 430

tins = np.linspace(550, 600, 1000)

powers1 = []
for T_in1 in tins:
    temp = model.inference(np.array([p_in, T_in1, cr, bore, far, PI, v_mean, T_fuel]))
    powers1.append(temp[0][0])

fars = np.linspace(0.02923 / 1.5, 0.02923 / 1.1, 1000)

powers2 = []
for far in fars:
    temp = model.inference(np.array([p_in, T_in, cr, bore, far, PI, v_mean, T_fuel]))
    powers2.append(temp[0][0])


s = 1

# plot the real values for power and the predicted one
fig, ax1 = plt.subplots()
ax1.scatter(points, y_both[:, 0], label="Validation", s=s)
ax1.scatter(points, y_both[:, 1], label="Predicted", s=s)
ax1.set_xlabel(r"Data point")
ax1.set_ylabel(r"Power [kW]")
ax1.set_title(rf"Straight. Layers: {layers + 1}, neurons: {hidden_dim}, Test set")
ax1.set_ylim(0, 1000)
# ax1.set_xlim(100, epochs)
plt.legend()


# plot the real values for power and the predicted one
fig, ax2 = plt.subplots()
ax2.scatter(points, rel_error, s=s)
ax2.set_xlabel(r"Data point")
ax2.set_ylabel(r"Relative error")
ax2.set_title(rf"Straight. Layers: {layers + 1}, neurons: {hidden_dim}, Test set")
# ax2.set_ylim(-10, 10)
# ax1.set_xlim(100, epochs)


fig, ax3 = plt.subplots()
ax3.scatter(points_train, y_both_train[:, 0], label="Validation", s=s)
ax3.scatter(points_train, y_both_train[:, 1], label="Predicted", s=s)
ax3.set_xlabel(r"Data point")
ax3.set_ylabel(r"Power [kW]")
ax3.set_title(rf"Straight. Layers: {layers + 1}, neurons: {hidden_dim}, Training set")
# ax3.set_ylim(0, 1000)
# ax1.set_xlim(100, epochs)
plt.legend()

# plot the real values for power and the predicted one
fig, ax4 = plt.subplots()
ax4.scatter(points_train, rel_error_train, s=s)
ax4.set_xlabel(r"Data point")
ax4.set_ylabel(r"Relative error")
ax4.set_title(rf"Straight. Layers: {layers + 1}, neurons: {hidden_dim}, Training set")
# ax2.set_ylim(-10, 10)
# ax1.set_xlim(100, epochs)


# plot the real values for power and the predicted one
fig, ax5 = plt.subplots()
ax5.plot(tins, powers1)
ax5.set_xlabel(r"T_in [K]")
ax5.set_ylabel(r"P [kW]")
ax5.set_title(rf"Layers: {layers + 1}, neurons: {hidden_dim}, Training set")
# ax2.set_ylim(-10, 10)
# ax1.set_xlim(100, epochs)

# plot the real values for power and the predicted one
fig, ax6 = plt.subplots()
ax6.plot(fars, powers2)
ax6.set_xlabel(r"far [-]")
ax6.set_ylabel(r"P [kW]")
ax6.set_title(rf"Layers: {layers + 1}, neurons: {hidden_dim}, Training set")
# ax2.set_ylim(-10, 10)
# ax1.set_xlim(100, epochs)


plt.show()
