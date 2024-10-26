import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import pandas as pd

from torch.utils.data import DataLoader

from src import Data, NET_straight

import torch.nn as nn
import torch.optim as optim

# import data
X = pd.read_csv('./data/x.csv', index_col=0)
y = pd.read_csv('./data/y.csv', index_col=0)

# convert to numpy arrays
X = pd.DataFrame.to_numpy(X)
y = pd.DataFrame.to_numpy(y)

# only look at power
y = y[:, [5]]

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=42)

# normalise the data, fit the normalisation on training data
X_mean = np.mean(X_train,0)
X_std = np.std(X_train,0)

y_mean = np.mean(y_train,0)
y_std = np.std(y_train,0)

# normalise the training data
X_train_norm = (X_train - X_mean) / X_std
y_train_norm = (y_train - y_mean) / y_std

# normalise the test data
X_test_norm = (X_test - X_mean) / X_std
y_test_norm = (y_test - y_mean) / y_std

data = np.concatenate((X_train_norm, y_train_norm), axis=1)

# check for covariance
corr = np.corrcoef(data, rowvar=False)

# if we want to see linear correlation
#plt.matshow(corr)
#plt.show()

# Generate the training dataset
traindata = Data(X_train_norm, y_train_norm)

# Generate the test dataset
testdata = Data(X_test_norm, y_test_norm)

batch_size = 32
epochs = 100

# number of neurons of hidden layers
hidden_dim = 256

# number of hidden layers
layers = 1

lr = 1e-3
weight_decay = 0


# Load the training data into data loader with the
# respective batch_size
trainloader = DataLoader(traindata, batch_size=batch_size,
                         shuffle=True)

# Validate model on validation data
testloader = DataLoader(testdata, batch_size=batch_size,
                        shuffle=True)

# number of inputs
input_dim = X_train_norm.shape[1]

# output dimension
output_dim = y_train_norm.shape[1]

# initiate the regression model
model = NET_straight(layers, input_dim, hidden_dim, output_dim)


print(model)

# criterion to computes the loss between input and target
criterion = nn.L1Loss()

# optimizer that will be used to update weights and biases
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


# saving losses for each epoch to visualize
training_loss = []
test_loss = []


# initialize best validation loss
best_loss = 1
best_epoch = 0

# early stopping threshold
epoch_threshold = 100

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

    # training loss
    #scheduler.step(running_loss / (i + 1))

    # test loss
    #scheduler.step(running_loss_test)

    lr = optimizer.param_groups[0]["lr"]

    # saving best model based on validation loss
    if running_loss_test < best_loss:
        best_loss = running_loss_test
        best_epoch = epoch
        checkpoint(model, optimizer, X_scaler, y_scaler, f'./models/straight_{hidden_dim}_{layers}.pth')

    elif epoch - best_epoch > epoch_threshold:
        print("Early stopped training at epoch %d" % epoch)
        # final epoch evaluated
        epochs = epoch
        break  # terminate the training loop

    # display statistics
    if not ((epoch + 1) % (epochs // 10)):
        print(f'Epochs:{epoch + 1:5d} | ' \
              f'Batches per epoch: {i + 1:3d} | ' \
              f'Training loss: {running_loss / (i + 1):.10f} | ' \
              f'Test loss: {running_loss_test:.10f} |' \
              f'Learning rate: {lr}')

    # save 10 checkpoints
    if not ((epoch + 1) % (epochs // 10)):
        # checkpoint the model parameters
        checkpoint(model, optimizer, X_scaler, y_scaler, f"./checkpoints/epoch-{epoch}.pth")
        # saving best model found so far based on validation loss


    training_loss.append(running_loss / (i + 1))
    test_loss.append(running_loss_test.detach().numpy())



# load the best model for validation
resume(model, optimizer, f'./models/straight_{hidden_dim}_{layers}.pth')

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

# dont plot first epochs due to very large initial loss
skip = 20

epochss = np.arange(start_epoch, epochs)

fig, ax1 = plt.subplots()
ax1.plot(epochss[skip:], training_loss[skip:], label='Training loss')
ax1.plot(epochss[skip:], test_loss[skip:], label='Test loss')
ax1.set_xlabel(r'Epoch')
ax1.set_ylabel(r'L1 loss')
ax1.set_title(fr'Straight. Layers: {layers}, Neurons 1st hidden layer: {hidden_dim}')
#ax1.set_ylim(0, 0.01)
#ax1.set_xlim(100, epochs)
plt.legend()
plt.show()


print(X_test)
print(y_test)


