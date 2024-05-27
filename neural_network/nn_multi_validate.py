import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from torcheval.metrics import R2Score

import matplotlib.pyplot as plt

## NN-PP: Neural Network Post Processing


# import data  IN FUTURE JUST IMPORT ONE CSV FILE
X = pd.read_csv('../piston_engine/sampled_data/h2/x_cleaned.csv', index_col=0)
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

# NOTES:
# 2 hidden layers with 32 neurons each seems good fit: R2 99.71 on test data
#
#
#
class NET(nn.Module):
    '''Regression Model
    '''

    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2: int, output_dim: int) -> None:
        '''The network has 4 layers
             - input layer
             - hidden layer
             - hidden layer
             - output layer
        '''
        super(NET, self).__init__()
        self.input_to_hidden = nn.Linear(input_dim, hidden_dim1)
        self.hidden_layer_1 = nn.Linear(hidden_dim1, hidden_dim2)
        self.hidden_layer_2 = nn.Linear(hidden_dim2, hidden_dim2)
        self.hidden_to_output = nn.Linear(hidden_dim2, output_dim)
        self.ReLu = nn.ReLU()  # activation function

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.input_to_hidden(x)
        x = self.ReLu(x)
        x = self.hidden_layer_1(x)
        x = self.ReLu(x)
        x = self.hidden_layer_2(x)
        x = self.ReLu(x)
        x = self.hidden_to_output(x)


        #x = F.relu(self.input_to_hidden(x))
        #x = F.relu(self.hidden_layer_1(x))
        #x = F.relu(self.hidden_layer_2(x))
        #x = self.hidden_to_output(x)
        return x


# Create new model and load states
newmodel = NET(7, 64, 64, 8)
newmodel.load_state_dict(torch.load("./model_multi.pth"))


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

# heat_loss
output = 6

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

plt.show()