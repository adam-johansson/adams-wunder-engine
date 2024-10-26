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

from src import clean_folder, checkpoint, resume, save, load, save_inference, load_ANN
from src import Data, NET_straight

# import data
X = pd.read_csv('./input_data/H2_mediumnarrow/x.csv', index_col=0)
y = pd.read_csv('./input_data/H2_mediumnarrow/y.csv', index_col=0)

# convert to numpy arrays
X = pd.DataFrame.to_numpy(X)
y = pd.DataFrame.to_numpy(y)

# only look at power
y = y[:, [5]]

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=42)


scaler = "1"
if scaler == "1":
    # normalise the data, fit the normalisation on training data (mean 0 std 1)
    X_mean = np.mean(X_train,0)
    X_std = np.std(X_train,0)

    y_mean = np.mean(y_train,0)
    y_std = np.std(y_train,0)

    # normalise the training data
    X_train = (X_train - X_mean) / X_std
    y_train = (y_train - y_mean) / y_std

    # normalise the test data
    X_test = (X_test - X_mean) / X_std
    y_test = (y_test - y_mean) / y_std

elif scaler == "2":
    # scale between 0 and 1
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)

    y_min = y_train.min(axis=0)
    y_max = y_train.max(axis=0)

    X_train = (X_train - x_min) / (x_max - x_min)
    y_train = (y_train - y_min) / (y_max - y_min)

    X_test = (X_test - x_min) / (x_max - x_min)
    y_test = (y_test - y_min) / (y_max - y_min)




# Generate the training dataset
traindata = Data(X_train, y_train)

# Generate the test dataset
testdata = Data(X_test, y_test)


batch_size = 64
epochs = 5000

# number of neurons of hidden layers
hidden_dim = 128

# number of hidden layers - 1
layers = 1

lr = 1e-3
weight_decay = 0.1

# allows for starting from a checkpoint
start_epoch = 0

# early stopping threshold
epoch_threshold = 200

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
#criterion = nn.L1Loss()
criterion = nn.MSELoss()

# optimizer that will be used to update weights and biases

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# learning rate scheduler
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5)

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
        save_inference(model,f'./models/straight_{hidden_dim}_{layers}.pth', X_std, X_mean, y_std, y_mean)

    elif epoch - best_epoch > epoch_threshold:
        print("Early stopped training at epoch %d" % epoch)
        # final epoch evaluated
        epochs = epoch
        break  # terminate the training loop

    # display statistics
    if not ((epoch + 1) % (epochs // 100)):
        print(f'Epochs:{epoch + 1:5d} | ' \
              f'Batches per epoch: {i + 1:3d} | ' \
              f'Training loss: {running_loss / (i + 1):.10f} | ' \
              f'Test loss: {running_loss_test:.10f} |' \
              f'Learning rate: {lr}')

    # save 10 checkpoints
    #if not ((epoch + 1) % (epochs // 10)):
        # checkpoint the model parameters
        #checkpoint(model, optimizer, X_scaler, y_scaler, f"./checkpoints/epoch-{epoch}.pth")
        # saving best model found so far based on validation loss


    training_loss.append(running_loss / (i + 1))
    test_loss.append(running_loss_test.detach().numpy())


# load the best model for validation
#resume(model, optimizer, f'./models/straight_{hidden_dim}_{layers}.pth')

model = load_ANN(f'./models/straight_{hidden_dim}_{layers}.pth')

# Validate trained model using the test dataset
with torch.no_grad():
    loss = 0
    for i, (inputs, labels) in enumerate(testloader):
        # calculate output by running through the network
        predictions = model(inputs)
        loss += F.l1_loss(predictions, labels)
    print(f'L1 Loss on test dataset: {loss / (i + 1):.5f}')




# dont plot first epochs due to very large initial loss
skip = 10

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




