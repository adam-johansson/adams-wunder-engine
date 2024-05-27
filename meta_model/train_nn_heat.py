import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd

import matplotlib.pyplot as plt

# import data  IN FUTURE JUST IMPORT ONE CSV FILE
X = pd.read_csv('../piston_engine/surrogate_data/backup/h2_validated_woschni_18/x_cleaned.csv', index_col=0)
y = pd.read_csv('../piston_engine/surrogate_data/backup/h2_validated_woschni_18/y_cleaned.csv', index_col=0)

# convert to numpy arrays
X = pd.DataFrame.to_numpy(X)
y = pd.DataFrame.to_numpy(y)


# Seventh output (heat loss)
y = y[:, 6]

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


class Data(Dataset):
    '''Dataset Class to store the samples and their corresponding labels,
    and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.
    '''

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


# Generate the training dataset
traindata = Data(X_train, y_train)

batch_size = 10
epochs = 1000

# Load the training data into data loader with the
# respective batch_size and num_workers
trainloader = DataLoader(traindata, batch_size=batch_size,
                         shuffle=True)

# NOTES:
# 2 hidden layers with 32 neurons each seems good fit: 10 batch size.
# 1000 Epochs: R2 test: 99.92, R2 train: 99.95


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


# number of features (len of X cols)
input_dim = X_train.shape[1]
# number of neurons of hidden layers
hidden_dim1 = 32
hidden_dim2 = 32
# output dimension is 1 because of linear regression
output_dim = 1
# initiate the linear regression model
model = NET(input_dim, hidden_dim1, hidden_dim2, output_dim)
print(model)

# criterion to computes the loss between input and target
criterion = nn.L1Loss()
# optimizer that will be used to update weights and biases
# lr 1e-3 was too large. 1e-4 seems to work well
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# saving losses for each epoch to visualize
training_loss = []


# start training

for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        # this is in each batch

        # inputs, labels = data

        # forward propagation
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # set optimizer to zero grad
        # to remove previous epoch gradients
        optimizer.zero_grad()

        # backward propagation
        loss.backward()

        # optimize
        optimizer.step()
        running_loss += loss.item()

    # display statistics
    if not ((epoch + 1) % (epochs // 10)):
        print(f'Epochs:{epoch + 1:5d} | ' \
              f'Batches per epoch: {i + 1:3d} | ' \
              f'Loss: {running_loss / (i + 1):.10f}')

    training_loss.append(running_loss / (i + 1))

# save the trained model
PATH = './model_heat.pth'
torch.save(model.state_dict(), PATH)


# Validate model on validation data
testdata = Data(X_test, y_test)
testloader = DataLoader(testdata, batch_size=batch_size,
                        shuffle=True)

# Validate trained model using the test dataset
with torch.no_grad():
    loss = 0
    for i, (inputs, labels) in enumerate(testloader):
        # calculate output by running through the network
        predictions = model(inputs)
        labels = torch.from_numpy(y_scaler.inverse_transform(labels))
        predictions = torch.from_numpy(y_scaler.inverse_transform(predictions))
        loss += F.l1_loss(predictions, labels)
    print(f'L1 Loss on test dataset: {loss / (i + 1):.5f}')


fig, ax1 = plt.subplots()
ax1.plot(training_loss)
ax1.set_xlabel(r'Epoch')
ax1.set_ylabel(r'Training loss')
plt.show()


predictions = model(torch.tensor(X_test, dtype=torch.float32))
