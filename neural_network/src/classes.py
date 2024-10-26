from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn as nn


class Data(Dataset):
    """Dataset Class to store the samples and their corresponding labels,
    and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        # need to convert float64 to float32 else
        # will get the following error
        # RuntimeError: expected scalar type Double but found Float
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, index: int) -> tuple:
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        return self.len

class NET_narrowing(nn.Module):
    '''Regression Model with halve number of neurons for each layer
    '''

    def __init__(self, n_layers: int, input_dim: int, hidden_dim: int, output_dim: int) -> None:

        super(NET_narrowing, self).__init__()
        self.input_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.ModuleList([nn.Linear(hidden_dim // (2**i), hidden_dim // (2**(i+1))) for i in range(n_layers)])
        self.hidden_to_output = nn.Linear(hidden_dim // (2**(n_layers)), output_dim)
        self.ReLu = nn.ReLU()  # activation function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_to_hidden(x)
        x = self.ReLu(x)
        for layer in self.hidden:
            x = layer(x)
            x = self.ReLu(x)
        x = self.hidden_to_output(x)

        return x




class NET_straight(nn.Module):
    '''Regression Model with same number of neurons for each hidden layer
    '''

    def __init__(self, n_layers: int, input_dim: int, hidden_dim: int, output_dim: int) -> None:

        super(NET_straight, self).__init__()
        self.input_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers)])
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



class InferenceModel(NET_narrowing):
    def __init__(self, layers, input_dim, hidden_dim, output_dim, weights, x_scaler, y_scaler):
        super().__init__(layers, input_dim, hidden_dim, output_dim)
        # load weights from saved file
        self.load_state_dict(weights)
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    def inference(self, x: np.array) -> np.array:
        #print(np.shape(np.atleast_2d(x))[0])
        if np.shape(np.atleast_2d(x))[0] == 1:
            x = x.reshape(1, -1)
        x = self.x_scaler.transform(x)
        if np.shape(np.atleast_2d(x))[0] == 1:
            x = torch.tensor(x, dtype=torch.float32).reshape(1, -1)
        else:
            x = torch.tensor(x, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            y = self.forward(x)
        y = self.y_scaler.inverse_transform(y.detach().numpy())

        return y


class InferenceModelStraight(NET_straight):
    def __init__(self, layers, input_dim, hidden_dim, output_dim, weights, x_std, x_mean, y_std, y_mean):
        super().__init__(layers, input_dim, hidden_dim, output_dim)
        # load weights from saved file
        self.load_state_dict(weights)
        self.x_std = x_std
        self.x_mean = x_mean
        self.y_std = y_std
        self.y_mean = y_mean

    def inference(self, x: np.array) -> np.array:
        #print(np.shape(np.atleast_2d(x))[0])
        if np.shape(np.atleast_2d(x))[0] == 1:
            x = x.reshape(1, -1)
        x = (x - self.x_mean) / self.x_std
        if np.shape(np.atleast_2d(x))[0] == 1:
            x = torch.tensor(x, dtype=torch.float32).reshape(1, -1)
        else:
            x = torch.tensor(x, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            y = self.forward(x)
        y = self.y_mean + self.y_std * y.detach().numpy()

        return y






