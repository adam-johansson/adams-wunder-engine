import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import torch.optim.lr_scheduler as lr_scheduler

import matplotlib.pyplot as plt

from src import clean_folder, checkpoint, resume
from src import Data, NET_straight

# import data
X = pd.read_csv('./input_data/H2/x_cleaned.csv', index_col=0)
y = pd.read_csv('./input_data/H2/y_cleaned.csv', index_col=0)

# convert to numpy arrays
X = pd.DataFrame.to_numpy(X)
y = pd.DataFrame.to_numpy(y)

# convert to PyTorch tensors
#X = torch.tensor(X, dtype=torch.float32)
#y = torch.tensor(y, dtype=torch.float32)

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=42)


# Normalize the data
# Which scaler to use???
X_scaler = StandardScaler()
#X_scaler = QuantileTransformer(output_distribution='normal')
# fit transformer on training data
X_train = X_scaler.fit_transform(X_train)
# just scale test data without fitting
X_test = X_scaler.transform(X_test)
y_scaler = StandardScaler()
#y_scaler = QuantileTransformer(output_distribution='normal')
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)


# Generate the training dataset
traindata = Data(X_train, y_train)

# Generate the test dataset
testdata = Data(X_test, y_test)


batch_size = 128
epochs = 500

# number of neurons of hidden layers
hidden_dim = 512

# number of hidden layers
layers = 2

lr = 1e-3
weight_decay = 0

# allows for starting from a checkpoint
start_epoch = 0

# Load the training data into data loader with the
# respective batch_size
trainloader = DataLoader(traindata, batch_size=batch_size,
                         shuffle=True)

# Validate model on validation data
testloader = DataLoader(testdata, batch_size=batch_size,
                        shuffle=True)



# number of inputs
input_dim = X_train.shape[1]

# output dimension
output_dim = y_train.shape[1]

# initiate the regression model
model = NET_straight(layers, input_dim, hidden_dim, output_dim)


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

# delete all checkpoints if new run from epoch 0 is started
if start_epoch == 0:
    clean_folder('./checkpoints')

# use checkpoint number from the file + 1
if start_epoch > 0:
    resume_epoch = start_epoch - 1
    resume(model, optimizer, f"./checkpoints/epoch-{resume_epoch}.pth")


# initialize best validation loss
best_loss = 1
best_epoch = 0

# early stopping threshold
epoch_threshold = 20

# start training

for epoch in range(start_epoch, epochs):
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
    scheduler.step(running_loss_test)

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



# save the trained model
#PATH = './models/narrowing_256_3.pth'
#torch.save(model.state_dict(), PATH)

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
ax1.set_ylabel(r'MSE loss')
ax1.set_title(fr'Straight. Layers: {layers}, Neurons 1st hidden layer: {hidden_dim}')
#ax1.set_ylim(0, 0.01)
#ax1.set_xlim(100, epochs)
plt.legend()
plt.show()

