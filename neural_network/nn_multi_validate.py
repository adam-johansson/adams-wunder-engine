import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from torcheval.metrics import R2Score
import numpy as np
import math

import matplotlib.pyplot as plt

## NN-PP: Neural Network Post Processing


# import data  IN FUTURE JUST IMPORT ONE CSV FILE
X = pd.read_csv('../piston_engine/sampled_data/h2/X_cleaned.csv', index_col=0)
y = pd.read_csv('../piston_engine/sampled_data/h2/y_cleaned.csv', index_col=0)

# convert to numpy arrays
X = pd.DataFrame.to_numpy(X)
y = pd.DataFrame.to_numpy(y)


# convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Normalize the data
# Which scaler to use???
X_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y)

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.33, random_state=42)


#
#
#

class NET(nn.Module):
    '''Regression Model
    '''

    def __init__(self, n_layers: int, input_dim: int, hidden_dim: int, output_dim: int) -> None:

        super(NET, self).__init__()
        self.input_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.hidden_to_output = nn.Linear(hidden_dim, output_dim)
        self.ReLu = nn.ReLU()  # activation function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_to_hidden(x)
        x = self.ReLu(x)
        for layer in self.hidden:
            x = layer(x)
            x = self.ReLu(x)
        x = self.hidden_to_output(x)

        return x


# Create new model and load states
newmodel = NET(3, 8, 128, 8)
newmodel.load_state_dict(torch.load("model_multi_4_128_relu.pth"))


# convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# test with new model using copied tensor
y_pred_test = newmodel(X_test)
y_pred_train = newmodel(X_train)


# R2 score
metric = R2Score()


# calculating R2 score between prediction and validation on training data
for i in range(y_pred_train.shape[1]):
    metric.update(y_pred_train[:, i], y_train[:, i])
    R2_test = metric.compute()
    print(f'R2 score training data{i}: {R2_test}')


# calculating R2 score between prediction and validation on test data
for i in range(y_pred_train.shape[1]):
    metric.update(y_pred_test[:, i], y_test[:, i])
    R2_test = metric.compute()
    print(f'R2 score test data{i}: {R2_test}')


# overall R2 score
metric.update(y_pred_test, y_test)
R2_test = metric.compute()
print(f'R2 score test data overall: {R2_test}')

# overall R2 score
metric.update(y_pred_train, y_train)
R2_test = metric.compute()
print(f'R2 score training data overall: {R2_test}')



# convert back to actual values
y_pred_train = y_scaler.inverse_transform(y_pred_train.detach().numpy())
y_pred_test = y_scaler.inverse_transform(y_pred_test.detach().numpy())
y_test = y_scaler.inverse_transform(y_test.detach().numpy())
y_train = y_scaler.inverse_transform(y_train.detach().numpy())

fig = plt.figure()

plt.subplot(1,2,1)
plt.plot(y_test[:, 0], y_test[:, 0], 'b', lw=2, label='Actual')
plt.plot(y_test[:, 0], y_pred_test[:, 0], 'ro', ms=2, label='Prediction')
plt.xlabel(r'Actual')
plt.ylabel(r'Prediction')
plt.title('Test data T_out')
plt.legend()

plt.subplot(1,2, 2)
plt.plot(y_train[:, 0], y_train[: ,0], 'b', lw=2, label='Actual')
plt.plot(y_train[:, 0], y_pred_train[:, 0], 'ro', ms=2, label='Prediction')
plt.xlabel(r'Actual')
plt.ylabel(r'Prediction')
plt.title(f'Training data T_out')
plt.legend()

fig = plt.figure()

plt.subplot(1, 2, 1)
plt.plot(y_test[:, 1], y_test[:, 1], 'b', lw=2, label='Actual')
plt.plot(y_test[:, 1], y_pred_test[:, 1], 'ro', ms=2, label='Prediction')
plt.xlabel(r'Actual')
plt.ylabel(r'Prediction')
plt.title('Test data eff')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y_train[:, 1], y_train[:, 1], 'b', lw=2, label='Actual')
plt.plot(y_train[:, 1], y_pred_train[:, 1], 'ro', ms=2, label='Prediction')
plt.xlabel(r'Actual')
plt.ylabel(r'Prediction')
plt.title(f'Training data eff')
plt.legend()

fig = plt.figure()

plt.subplot(1, 2, 1)
plt.plot(y_test[:, 2], y_test[:, 2], 'b', lw=2, label='Actual')
plt.plot(y_test[:, 2], y_pred_test[:, 2], 'ro', ms=2, label='Prediction')
plt.xlabel(r'Actual')
plt.ylabel(r'Prediction')
plt.title('Test data air_flow')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y_train[:, 2], y_train[:, 2], 'b', lw=2, label='Actual')
plt.plot(y_train[:, 2], y_pred_train[:, 2], 'ro', ms=2, label='Prediction')
plt.xlabel(r'Actual')
plt.ylabel(r'Prediction')
plt.title(f'Training data air_flow')
plt.legend()


fig = plt.figure()

plt.subplot(1, 2, 1)
plt.plot(y_test[:, 3], y_test[:, 3], 'b', lw=2, label='Actual')
plt.plot(y_test[:, 3], y_pred_test[:, 3], 'ro', ms=2, label='Prediction')
plt.xlabel(r'Actual')
plt.ylabel(r'Prediction')
plt.title('Test data p_max')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y_train[:, 3], y_train[:, 3], 'b', lw=2, label='Actual')
plt.plot(y_train[:, 3], y_pred_train[:, 3], 'ro', ms=2, label='Prediction')
plt.xlabel(r'Actual')
plt.ylabel(r'Prediction')
plt.title(f'Training data p_max')
plt.legend()


fig = plt.figure()

# T_max
output = 4

plt.subplot(1, 2, 1)
plt.plot(y_test[:, output], y_test[:, output], 'b', lw=2, label='Actual')
plt.plot(y_test[:, output], y_pred_test[:, output], 'ro', ms=2, label='Prediction')
plt.xlabel(r'Actual')
plt.ylabel(r'Prediction')
plt.title('Test data T_max')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y_train[:, output], y_train[:, output], 'b', lw=2, label='Actual')
plt.plot(y_train[:, output], y_pred_train[:, output], 'ro', ms=2, label='Prediction')
plt.xlabel(r'Actual')
plt.ylabel(r'Prediction')
plt.title(f'Training data T_max')
plt.legend()


fig = plt.figure()

# indicated power
output = 5

plt.subplot(1, 2, 1)
plt.plot(y_test[:, output], y_test[:, output], 'b', lw=2, label='Actual')
plt.plot(y_test[:, output], y_pred_test[:, output], 'ro', ms=2, label='Prediction')
plt.xlabel(r'Actual')
plt.ylabel(r'Prediction')
plt.title('Test data power')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y_train[:, output], y_train[:, output], 'b', lw=2, label='Actual')
plt.plot(y_train[:, output], y_pred_train[:, output], 'ro', ms=2, label='Prediction')
plt.xlabel(r'Actual')
plt.ylabel(r'Prediction')
plt.title(f'Training data power')
plt.legend()

fig = plt.figure()

# heat_loss
output = 6

plt.subplot(1, 2, 1)
plt.plot(y_test[:, output], y_test[:, output], 'b', lw=2, label='Actual')
plt.plot(y_test[:, output], y_pred_test[:, output], 'ro', ms=2, label='Prediction')
plt.xlabel(r'Actual')
plt.ylabel(r'Prediction')
plt.title('Test data heatloss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y_train[:, output], y_train[:, output], 'b', lw=2, label='Actual')
plt.plot(y_train[:, output], y_pred_train[:, output], 'ro', ms=2, label='Prediction')
plt.xlabel(r'Actual')
plt.ylabel(r'Prediction')
plt.title(f'Training data heatloss')
plt.legend()

fig = plt.figure()

# p_tdc
output = 7

plt.subplot(1, 2, 1)
plt.plot(y_test[:, output], y_test[:, output], 'b', lw=2, label='Actual')
plt.plot(y_test[:, output], y_pred_test[:, output], 'ro', ms=2, label='Prediction')
plt.xlabel(r'Actual')
plt.ylabel(r'Prediction')
plt.title('Test data p_tdc')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y_train[:, output], y_train[:, output], 'b', lw=2, label='Actual')
plt.plot(y_train[:, output], y_pred_train[:, output], 'ro', ms=2, label='Prediction')
plt.xlabel(r'Actual')
plt.ylabel(r'Prediction')
plt.title(f'Training data p_tdc')
plt.legend()

#plt.show()

for i in range(8):

    # root square error
    RSE = np.sqrt(np.square(np.subtract(y_test[:, i], y_pred_test[:, i]) ) )

    # relative error
    rel_error = np.divide(RSE, y_test[:, i])

    # mean relative error
    MRE_ptdc = rel_error.mean()


    print(f'MRE: {MRE_ptdc} for output {i}')


# go from 12 cylinders to 1
power_pred_trimmed = y_pred_test[:, 5] / 12
power_val_trimmed = y_test[:,5] / 12
power_pred_trimmed = np.atleast_2d(power_pred_trimmed).T
power_val_trimmed = np.atleast_2d(power_val_trimmed).T

power_trimmed = np.concatenate((power_val_trimmed, power_pred_trimmed), axis=1)

power_trimmed = power_trimmed[power_trimmed[:, 0].argsort()]

#mask = power_trimmed[:, 0] < 1500

#power_trimmed = power_trimmed[mask]

mask = power_trimmed[:, 0] > 1

power_trimmed = power_trimmed[mask]

# root square error
RSE = np.sqrt(np.square(np.subtract(power_trimmed[:, 0], power_trimmed[:, 1])))

# relative error
rel_error = np.divide(RSE, power_trimmed[:, 0])

# mean relative error
MRE_ptdc = rel_error.mean()

print(f'MRE: {MRE_ptdc} for power over 5 kW')







fs = 52
figsize = (24, 16)
res = 50




fig, ax1 = plt.subplots(figsize=figsize)
ax1.plot(y_test[:, 0], y_test[:, 0], 'b', lw=2, label='Actual')
ax1.scatter(y_test[:, 0], y_pred_test[:, 0], label='Prediction', color="r", s=32)

ax1.set_xlabel(r'Actual $T_2$ [K]', fontsize=fs)
ax1.set_ylabel(r'Predicted $T_2$ [K]', fontsize=fs)
ax1.set_title(r'$T_2$ prediction on validation data set', fontsize=fs)
#ax1.set_xlim(710, 770)
#ax1.set_xticks([720, 730, 740, 750, 760, 770])
#ax1.set_ylim(0, 14)
#ax1.set_yticks([0, 2, 4, 6, 8, 10, 12, 14])
ax1.tick_params(labelsize=fs)
plt.legend(loc='best', frameon=True, fontsize=fs)
#ax1.grid()
#plt.savefig('T_out_validation.pdf', dpi=res, bbox_inches='tight')



fig, ax2 = plt.subplots(figsize=figsize)
ax2.plot(y_test[:, 5] / 12, y_test[:, 5] / 12, 'b', lw=2, label='Actual')
ax2.scatter(y_test[:, 5] / 12, y_pred_test[:, 5] / 12, label='Prediction', color="r", s=32)
ax2.set_xlabel(r'Actual $P_i$ [kW]', fontsize=fs)
ax2.set_ylabel(r'Predicted $P_i$ [kW]', fontsize=fs)
ax2.set_title(r'$P_i$ prediction on validation data set', fontsize=fs)
#ax2.set_xlim(660, 810)
#ax2.set_xticks([690, 720, 750, 780, 810])
#ax2.set_ylim(0, 55)
#ax2.set_yticks([0, 10, 20, 30, 40, 50])
ax2.tick_params(labelsize=fs)
plt.legend(loc='best', frameon=True, fontsize=fs)
#ax2.grid()
#plt.savefig('power_validation.pdf', dpi=res, bbox_inches='tight')

#plt.show()



power_transpose = np.atleast_2d(y_test[:, 5] / 12).T
power_pred_transpose = np.atleast_2d(y_pred_test[:, 5] / 12).T

power_true = np.concatenate((power_transpose, power_transpose), axis=1)
power_pred = np.concatenate((power_transpose, power_pred_transpose), axis=1)
power_true = power_true[power_true[:, 0].argsort()]
power_pred = power_pred[power_pred[:, 0].argsort()]

T2_transpose = np.atleast_2d(y_test[:, 0]).T
T2_pred_transpose = np.atleast_2d(y_pred_test[:, 0]).T


T2_true = np.concatenate((T2_transpose, T2_transpose), axis=1)
T2_pred = np.concatenate((T2_transpose, T2_pred_transpose), axis=1)
T2_true = T2_true[T2_true[:, 0].argsort()]
T2_pred = T2_pred[T2_pred[:, 0].argsort()]

#np.savetxt("nn_output_data/power_true.dat", power_true, fmt="%s")
#np.savetxt("nn_output_data/power_pred.dat", power_pred, fmt="%s")
#np.savetxt("nn_output_data/t2_true.dat", T2_true, fmt="%s")
#np.savetxt("nn_output_data/t2_pred.dat", T2_pred, fmt="%s")




