import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from torcheval.metrics import R2Score

import matplotlib.pyplot as plt



def load_nn():

    # import data just to now the scale
    X = pd.read_csv('../piston_engine/surrogate_data/backup/h2_validated_woschni_18/x_cleaned.csv', index_col=0)
    y = pd.read_csv('../piston_engine/surrogate_data/backup/h2_validated_woschni_18/y_cleaned.csv', index_col=0)

    # convert to numpy arrays
    X = pd.DataFrame.to_numpy(X)
    y = pd.DataFrame.to_numpy(y)

    # need one scaler for each output
    y0 = y[:, 0]
    y2 = y[:, 2]
    y3 = y[:, 3]
    y4 = y[:, 4]
    y5 = y[:, 5]
    y6 = y[:, 6]
    y7 = y[:, 7]

    # convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y0 = torch.tensor(y0, dtype=torch.float32).reshape(-1, 1)
    y2 = torch.tensor(y2, dtype=torch.float32).reshape(-1, 1)
    y3 = torch.tensor(y3, dtype=torch.float32).reshape(-1, 1)
    y4 = torch.tensor(y4, dtype=torch.float32).reshape(-1, 1)
    y5 = torch.tensor(y5, dtype=torch.float32).reshape(-1, 1)
    y6 = torch.tensor(y6, dtype=torch.float32).reshape(-1, 1)
    y7 = torch.tensor(y7, dtype=torch.float32).reshape(-1, 1)

    # Normalize the data just to get the scaler
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    y0_scaler = StandardScaler()
    y0_scaled = y0_scaler.fit_transform(y0)
    y2_scaler = StandardScaler()
    y2_scaled = y2_scaler.fit_transform(y2)
    y3_scaler = StandardScaler()
    y3_scaled = y3_scaler.fit_transform(y3)
    y4_scaler = StandardScaler()
    y4_scaled = y4_scaler.fit_transform(y4)
    y5_scaler = StandardScaler()
    y5_scaled = y5_scaler.fit_transform(y5)
    y6_scaler = StandardScaler()
    y6_scaled = y6_scaler.fit_transform(y6)
    y7_scaler = StandardScaler()
    y7_scaled = y7_scaler.fit_transform(y7)


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
    model0 = NET(7, 32, 32, 1)
    model0.load_state_dict(torch.load("../meta_model/model_Tout.pth"))

    model2 = NET(7, 32, 32, 1)
    model2.load_state_dict(torch.load("../meta_model/model_airflow.pth"))

    model3 = NET(7, 32, 32, 1)
    model3.load_state_dict(torch.load("../meta_model/model_pmax.pth"))

    model4 = NET(7, 32, 32, 1)
    model4.load_state_dict(torch.load("../meta_model/model_Tmax.pth"))

    model5 = NET(7, 32, 32, 1)
    model5.load_state_dict(torch.load("../meta_model/model_power.pth"))

    model6 = NET(7, 32, 32, 1)
    model6.load_state_dict(torch.load("../meta_model/model_heat.pth"))

    model7 = NET(7, 32, 32, 1)
    model7.load_state_dict(torch.load("../meta_model/model_ptdc.pth"))

    model = [model0, model2, model3, model4, model5, model6, model7, X_scaler, y0_scaler, y2_scaler, y3_scaler,
             y4_scaler, y5_scaler, y6_scaler, y7_scaler]

    return model