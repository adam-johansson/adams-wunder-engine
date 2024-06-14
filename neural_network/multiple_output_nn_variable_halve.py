import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.optim.lr_scheduler as lr_scheduler

import matplotlib.pyplot as plt

# import data
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
    X_scaled, y_scaled, train_size=0.8, test_size=0.2, random_state=42)



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

# Generate the test dataset
testdata = Data(X_test, y_test)


batch_size = 25
epochs = 400

# number of neurons of hidden layers
hidden_dim1 = 75
hidden_dim2 = 50
hidden_dim3 = 25


lr = 1e-2
weight_decay = 0.0

# Load the training data into data loader with the
# respective batch_size
trainloader = DataLoader(traindata, batch_size=batch_size,
                         shuffle=True)

# Validate model on validation data
testloader = DataLoader(testdata, batch_size=batch_size,
                        shuffle=True)


class NET(nn.Module):
    '''Regression Model
    '''

    def __init__(self, n_layers: int, input_dim: int, hidden_dim: int, output_dim: int) -> None:

        super(NET, self).__init__()
        self.input_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.ModuleList([nn.Linear(hidden_dim // (2**i), hidden_dim // (2**(i+1))) for i in range(n_layers)])
        self.hidden_to_output = nn.Linear(hidden_dim // (2**(n_layers)), output_dim)
        self.ReLu = nn.ReLU()  # activation function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_to_hidden(x)
        #x = self.ReLu(x)
        for layer in self.hidden:
            x = layer(x)
            x = self.ReLu(x)
        x = self.hidden_to_output(x)

        return x


# number of inputs
input_dim = X_train.shape[1]

# output dimension
output_dim = y_train.shape[1]

# initiate the regression model
model = NET(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim)


print(model)

# criterion to computes the loss between input and target
criterion = nn.MSELoss()

# optimizer that will be used to update weights and biases

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# learning rate scheduler
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

# saving losses for each epoch to visualize
training_loss = []
test_loss = []


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

    # test loss (only once per epoch)
    outputs_test = model(testdata.X)
    running_loss_test = criterion(outputs_test, testdata.y)

    # update learning rate
    # normally use val loss but for overfitting use training loss
    scheduler.step(running_loss / (i + 1))
    lr = optimizer.param_groups[0]["lr"]

    # display statistics
    if not ((epoch + 1) % (epochs // 100)):
        print(f'Epochs:{epoch + 1:5d} | ' \
              f'Batches per epoch: {i + 1:3d} | ' \
              f'Training loss: {running_loss / (i + 1):.10f} | ' \
              f'Test loss: {running_loss_test:.10f} |' \
              f'Learning rate: {lr}')

    training_loss.append(running_loss / (i + 1))
    test_loss.append(running_loss_test.detach().numpy())

# save the trained model
PATH = 'model_multi_halve.pth'
torch.save(model.state_dict(), PATH)


# Validate trained model using the test dataset
with torch.no_grad():
    loss = 0
    for i, (inputs, labels) in enumerate(testloader):
        # calculate output by running through the network
        predictions = model(inputs)
        loss += F.l1_loss(predictions, labels)
    print(f'L1 Loss on test dataset, scaled data: {loss / (i + 1):.5f}')


# Validate trained model using the test dataset, real values
with torch.no_grad():
    loss = 0
    for i, (inputs, labels) in enumerate(testloader):
        # calculate output by running through the network
        predictions = model(inputs)
        labels = torch.from_numpy(y_scaler.inverse_transform(labels))
        predictions = torch.from_numpy(y_scaler.inverse_transform(predictions))
        loss += F.l1_loss(predictions, labels)
    print(f'L1 Loss on test dataset, real numbers: {loss / (i + 1):.5f}')


epochss = np.arange(20, epochs)
fig, ax1 = plt.subplots()
ax1.plot(epochss, training_loss[20:], label='Training loss')
ax1.plot(epochss, test_loss[20:], label='Test loss')
ax1.set_xlabel(r'Epoch')
ax1.set_ylabel(r'Training loss')
#ax1.set_ylim(0, 0.01)
#ax1.set_xlim(100, epochs)
plt.legend()
plt.show()

