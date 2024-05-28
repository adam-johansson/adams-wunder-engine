import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from torcheval.metrics import R2Score

import matplotlib.pyplot as plt



def load_nn():

    # import data just to know the scale
    X = pd.read_csv('../piston_engine/sampled_data/h2/x_cleaned.csv', index_col=0)
    y = pd.read_csv('../piston_engine/sampled_data/h2/y_cleaned.csv', index_col=0)

    # convert to numpy arrays
    X = pd.DataFrame.to_numpy(X)
    y = pd.DataFrame.to_numpy(y)

    # convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Normalize the data just to get the scaler
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)



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

            # x = F.relu(self.input_to_hidden(x))
            # x = F.relu(self.hidden_layer_1(x))
            # x = F.relu(self.hidden_layer_2(x))
            # x = self.hidden_to_output(x)
            return x

    # Create new model and load states
    model = NET(7, 64, 64, 8)
    model.load_state_dict(torch.load("../neural_network/model_multi.pth"))

    bundle = [model, X_scaler, y_scaler]

    return bundle