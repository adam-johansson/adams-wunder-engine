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
X = pd.read_csv('../piston_engine/surrogate_data/backup/h2_validated_woschni_18/x_cleaned.csv', index_col=0)
y = pd.read_csv('../piston_engine/surrogate_data/backup/h2_validated_woschni_18/y_cleaned.csv', index_col=0)

# convert to numpy arrays
X = pd.DataFrame.to_numpy(X)
y = pd.DataFrame.to_numpy(y)


# choose correct output for each model
y = y[:, 2]

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
newmodel = NET(7, 32, 32, 1)
newmodel.load_state_dict(torch.load("./model_airflow.pth"))


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

# calculating R2 score between prediction and validation on test data
metric.update(y_pred_test, y_test)
R2_test = metric.compute()
print(f'R2 score test data: {R2_test}')


# calculating R2 score between prediction and validation on training data
metric.update(y_pred_train, y_train)
R2_test = metric.compute()
print(f'R2 score training data: {R2_test}')

#convert back to actual values
y_pred_train = y_scaler.inverse_transform(y_pred_train.detach().numpy())
y_pred_test = y_scaler.inverse_transform(y_pred_test.detach().numpy())
y_test = y_scaler.inverse_transform(y_test.detach().numpy())
y_train = y_scaler.inverse_transform(y_train.detach().numpy())

fig, ax1 = plt.subplots()
ax1.plot(y_test, y_test, 'b', lw=2, label='Actual')
ax1.plot(y_test, y_pred_test, 'ro', ms=2, label='Prediction')
ax1.set_xlabel(r'Actual')
ax1.set_ylabel(r'Prediction')
ax1.set_title('Test data')
plt.legend()

fig, ax2 = plt.subplots()
ax2.plot(y_train, y_train, 'b', lw=2, label='Actual')
ax2.plot(y_train, y_pred_train, 'ro', ms=2, label='Prediction')
ax2.set_xlabel(r'Actual')
ax2.set_ylabel(r'Prediction')
ax2.set_title(f'Training data')
plt.legend()

plt.show()
