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
X = pd.read_csv("../piston_engine/sampled_data/h2/X_cleaned.csv", index_col=0)
y = pd.read_csv("../piston_engine/sampled_data/h2/y_cleaned.csv", index_col=0)

# convert to numpy arrays
X = pd.DataFrame.to_numpy(X)
y = pd.DataFrame.to_numpy(y)
y = y[:, 5]


# convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# Normalize the data
# Which scaler to use???
X_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y)

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.33, random_state=42
)


#
#
#


class NET(nn.Module):
    """Regression Model"""

    def __init__(
        self, n_layers: int, input_dim: int, hidden_dim: int, output_dim: int
    ) -> None:

        super(NET, self).__init__()
        self.input_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.ModuleList(
            [
                nn.Linear(hidden_dim // (2**i), hidden_dim // (2 ** (i + 1)))
                for i in range(n_layers)
            ]
        )
        self.hidden_to_output = nn.Linear(hidden_dim // (2 ** (n_layers)), output_dim)
        self.ReLu = nn.ReLU()  # activation function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_to_hidden(x)
        # x = self.ReLu(x)
        for layer in self.hidden:
            x = layer(x)
            x = self.ReLu(x)
        x = self.hidden_to_output(x)

        return x


# Create new model and load states
newmodel = NET(3, 8, 256, 1)
newmodel.load_state_dict(torch.load("model_power.pth"))


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

# overall R2 score
metric.update(y_pred_test, y_test)
R2_test = metric.compute()
print(f"R2 score test data overall: {R2_test}")

# overall R2 score
metric.update(y_pred_train, y_train)
R2_test = metric.compute()
print(f"R2 score training data overall: {R2_test}")


# convert back to actual values
y_pred_train = y_scaler.inverse_transform(y_pred_train.detach().numpy())
y_pred_test = y_scaler.inverse_transform(y_pred_test.detach().numpy())
y_test = y_scaler.inverse_transform(y_test.detach().numpy())
y_train = y_scaler.inverse_transform(y_train.detach().numpy())


# root square error
RSE = np.sqrt(np.square(np.subtract(y_test, y_pred_test)))

# relative error
rel_error = np.divide(RSE, y_test)

# mean relative error
MRE = rel_error.mean()


print(f"MRE: {MRE}")


# go from 12 cylinders to 1
power_pred_trimmed = y_pred_test / 12
power_val_trimmed = y_test / 12
# power_pred_trimmed = np.atleast_2d(power_pred_trimmed).T
# power_val_trimmed = np.atleast_2d(power_val_trimmed).T
# power_pred_trimmed = power_pred_trimmed.T
# power_val_trimmed = power_val_trimmed.T

power_trimmed = np.concatenate((power_val_trimmed, power_pred_trimmed), axis=1)

power_trimmed = power_trimmed[power_trimmed[:, 0].argsort()]

mask = power_trimmed[:, 0] < 10000

power_trimmed = power_trimmed[mask]

mask = power_trimmed[:, 0] > 5

power_trimmed = power_trimmed[mask]

# root square error
RSE = np.sqrt(np.square(np.subtract(power_trimmed[:, 1], power_trimmed[:, 0])))

# relative error
rel_error = np.divide(RSE, power_trimmed[:, 0])

# mean relative error
MRE_ptdc = rel_error.mean()

print(f"MRE: {MRE_ptdc} for power over x kW")


fs = 24
figsize = (12, 8)
res = 50


fig, ax2 = plt.subplots(figsize=figsize)
# ax2.plot(y_test[:, 5] / 12, y_test[:, 5] / 12, 'b', lw=2, label='Actual')
ax2.scatter(
    power_trimmed[:, 0],
    (power_trimmed[:, 1] - power_trimmed[:, 0]) / power_trimmed[:, 0],
    label="Prediction",
    color="r",
    s=32,
)
ax2.set_xlabel(r"Actual $P_i$ [kW]", fontsize=fs)
ax2.set_ylabel(r"Predicted $P_i$ [kW]", fontsize=fs)
ax2.set_title(r"$P_i$ prediction on validation data set", fontsize=fs)
# ax2.set_xlim(660, 810)
# ax2.set_xticks([690, 720, 750, 780, 810])
# ax2.set_ylim(0, 55)
# ax2.set_yticks([0, 10, 20, 30, 40, 50])
ax2.tick_params(labelsize=fs)
plt.legend(loc="best", frameon=True, fontsize=fs)
# ax2.grid()
# plt.savefig('power_validation.pdf', dpi=res, bbox_inches='tight')

plt.show()


# create data for tikx
# power_trimmed[:,0], (power_trimmed[:,1] - power_trimmed[:,0]) / power_trimmed[:,0]

# power_diff = 100 * (power_trimmed[:,1] - power_trimmed[:,0]) / power_trimmed[:,0]
# power_diff = np.atleast_2d()
# power_true = power_trimmed[:,0]

# power_data = np.concatenate((power_true, power_diff), axis=1)


np.savetxt("nn_output_data/power_diff.dat", power_data, fmt="%.5f")
