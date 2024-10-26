import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
import pandas as pd
from torcheval.metrics import R2Score
import numpy as np
import math

import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt

from src import load_inference, load

## NN-PP: Neural Network Post Processing


# Load the trained model
hidden_dim = 512
layers = 1
model = load(f'./models/straight_{hidden_dim}_{layers}.pth')
print(model)

X = pd.read_csv('./input_data/H2_narrow/x.csv', index_col=0)
y = pd.read_csv('./input_data/H2_narrow/y.csv', index_col=0)

# convert to numpy arrays
X = pd.DataFrame.to_numpy(X)
y = pd.DataFrame.to_numpy(y)

# only look at power
y = y[:, [5]]

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=42)


mean_far = np.mean(X_test[:,4])

# normalise the data, fit the normalisation on training data
X_mean = np.mean(X_train,0)
X_std = np.std(X_train,0)

y_mean = np.mean(y_train,0)
y_std = np.std(y_train,0)

# normalise the training data
X_train = (X_train - X_mean) / X_std
y_train = (y_train - y_mean) / y_std

# normalise the test data
X_test = (X_test - X_mean) / X_std
y_test = (y_test - y_mean) / y_std

# convert to tensor
X_test = torch.tensor(X_test, dtype=torch.float32)
X_train = torch.tensor(X_train, dtype=torch.float32)

y_test_hat = []
y_train_hat = []

for i in range(X_test.shape[0]):
    y_test_hat.append(model(X_test[i, :]).detach().numpy())


for i in range(X_train.shape[0]):
    y_train_hat.append(model(X_train[i, :]).detach().numpy())


# rescale
y_test = y_test * y_std + y_mean
y_test_hat = y_test_hat * y_std + y_mean
y_train = y_train * y_std + y_mean
y_train_hat = y_train_hat * y_std + y_mean

X_test = X_test * X_std + X_mean



# sort from smallest to biggest power
y_both = np.hstack((y_test, y_test_hat))
#y_both = y_both[y_both[:,0].argsort()]

y_both_train = np.hstack((y_train, y_train_hat))
#y_both_train = y_both_train[y_both_train[:,0].argsort()]

# relative error
rel_error = (y_both[:, 1] - y_both[:, 0]) / y_both[:, 0]
rel_error_train = (y_both_train[:, 1] - y_both_train[:, 0]) / y_both_train[:, 0]

print(f"Average rel error: {np.mean(np.abs(rel_error))} \n")
print(f"Average rel error training: {np.mean(np.abs(rel_error_train))}")

big_error_idx = np.abs(rel_error) > 0.1

y_big_error = y_test[big_error_idx]
X_big_error = X_test[big_error_idx].detach().numpy()

mean_y = np.mean(y_test)
mean_y_big_error = np.mean(y_big_error)
mean_far_big_error = np.mean(X_big_error[:,4])

print(f"Mean power test data: {mean_y}. mean power big error: {mean_y_big_error}")
print(f"Mean far test data: {mean_far}. mean far big error: {mean_far_big_error}")

plt.plot(X_big_error[:,4])
plt.show()


points = np.arange(1,y_test.shape[0] + 1, 1)
points_train = np.arange(1,y_train.shape[0] + 1, 1)

# plot the real values for power and the predicted one
fig, ax1 = plt.subplots()
ax1.scatter(points, y_both[:, 0], label='Validation')
ax1.scatter(points, y_both[:, 1], label='Predicted')
ax1.set_xlabel(r'Data point')
ax1.set_ylabel(r'Power [kW]')
ax1.set_title(fr'Straight. Layers: {layers}, Neurons 1st hidden layer: {hidden_dim}')
ax1.set_ylim(0, 1000)
#ax1.set_xlim(100, epochs)
plt.legend()


# plot the real values for power and the predicted one
fig, ax2 = plt.subplots()
ax2.scatter(points, rel_error)
ax2.set_xlabel(r'Data point')
ax2.set_ylabel(r'Relative error')
ax2.set_title(fr'Straight. Layers: {layers}, Neurons 1st hidden layer: {hidden_dim}')
#ax2.set_ylim(-10, 10)
#ax1.set_xlim(100, epochs)

#plt.show()