
import sys
sys.path.append("./../")



import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    QuantileTransformer,
)


from sklearn.preprocessing import KBinsDiscretizer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import torch.optim.lr_scheduler as lr_scheduler

import matplotlib.pyplot as plt

from src import clean_folder, checkpoint, resume, save, load, save_inference, load_ANN
from src import Data, NET_straight, scale_data



# This script trains 5 different neural networks on the data that has been sampled by the piston engine simulation
# Each model has 7 inputs and 1 output. 
# There is one model for power prediciton, one for nox, peak pressure, outlet temperature and airflow
# Heat loss is calculated outside the neural network by energy conservation


# import data

folder = "jetA"

data = pd.read_csv("./input_data/" + folder + "/data.csv", index_col=0)

raw_points = data.shape[0]

print(f"Number of datapoints total: {data.shape[0]}")

# remove datapoints that were not sampled (invalid input)
data = data[data.eff != 0]
data = data[pd.notna(data.eff)]

print(f"Number of datapoints removed during sampling: {raw_points - data.shape[0]}")
print(f"Number of datapoints not removed during sampling: {data.shape[0]}")


# settings for the neural network

batch_size = 64
epochs = 1000

# number of neurons of hidden layers
hidden_dim = 32

# number of hidden layers - 1
layers = 1

#weight_decay = 1e-3
weight_decay = 0.0

# allows for starting from a checkpoint
start_epoch = 0

# early stopping threshold
epoch_threshold = 20



# All models have the same inputs
# NOTE: T_fuel was in here before but I think it is unnessecarry to include
X = data[['p_in', 'T_in', 'PI', 'cr', 'bore',  'v_mean', 'far_goal']]

# convert to numpy array
X = pd.DataFrame.to_numpy(X)

# NOTE: removed p_tdc, heatloss and Tmax
#output_list = ['power', 'nox', 'p_max', 'T_out', 'air_flow']
output_list = ['power']

for output in output_list:
    print(output)
    y = data[[f"{output}"]]


    # convert to numpy array
    y = pd.DataFrame.to_numpy(y)


    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=42
    )

    # Scale the data
    scaler = "standard"

    X_train, y_train, X_test, y_test, x1, x2, y1, y2 = scale_data(scaler, X_train, y_train, X_test, y_test)


    # Generate the training dataset
    traindata = Data(X_train, y_train)

    # Generate the test dataset
    testdata = Data(X_test, y_test)



    # Load the training data into data loader with the
    # respective batch_size
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True)

    # Validate model on validation data
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True)


    # number of inputs
    input_dim = X_train.shape[1]

    # output dimension
    output_dim = y_train.shape[1]

    # initiate the regression model
    model = NET_straight(layers, input_dim, hidden_dim, output_dim)

    # convert it to double precision
    model = model.double()


    print(model)

    if scaler == "standard":
        # Get scaling parameters from your scale_data function
        # You'll need to modify scale_data to return the scaler objects or parameters
        # For now, assuming you have access to the fitted scalers:
        x_scaler_params = {
            'mean': x1,  # These should be the scaling parameters from scale_data
            'std': x2
        }
        y_scaler_params = {
            'mean': y1,
            'std': y2
        }
    else:
        # Handle other scaler types (MinMax, Robust, etc.)
        raise NotImplementedError(f"Scaling parameters extraction not implemented for {scaler}")

    # criterion to computes the loss between input and target
    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()

    # optimizer that will be used to update weights and biases
    lr = 1e-3

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # learning rate scheduler (100 patience normal)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=epoch_threshold/2, factor=0.5, min_lr=1e-5
    )

    # saving losses for each epoch to visualize
    training_loss = []
    test_loss = []

    # NOTE: Fix this for multiple models
    # delete all checkpoints if new run from epoch 0 is started
    if start_epoch == 0:
        clean_folder("./checkpoints")

    # use checkpoint number from the file + 1
    if start_epoch > 0:
        resume_epoch = start_epoch - 1
        resume(model, optimizer, f"./checkpoints/epoch-{resume_epoch}.pth")


    # initialize best validation loss
    best_loss = float('inf')
    best_epoch = 0


    # start training

    for epoch in range(start_epoch, epochs):
        model.train()
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
        model.eval()
        running_loss_test = 0.0
        with torch.no_grad():
            for test_inputs, test_labels in testloader:
                test_outputs = model(test_inputs)
                running_loss_test += criterion(test_outputs, test_labels).item()
        running_loss_test /= len(testloader)
        model.train()

        # update learning rate
        # normally use val loss but for overfitting use training loss

        # training loss
        # scheduler.step(running_loss / (i + 1))

        # test loss
        scheduler.step(running_loss_test)

        lr = optimizer.param_groups[0]["lr"]

        # saving best model based on validation loss
        if running_loss_test < best_loss:
            best_loss = running_loss_test
            best_epoch = epoch
            save_inference(
                model, f"models/{folder}_{hidden_dim}_{layers}/{output}.pth", scaler, x1, x2, y1, y2
            )

        elif epoch - best_epoch > epoch_threshold:
            print("Early stopped training at epoch %d" % epoch)
            # final epoch evaluated
            epochs = epoch
            break  # terminate the training loop

        # display statistics
        print_interval = max(1, epochs // 1000)  # Print 1000 times
        if (epoch + 1) % print_interval == 0:
            print(
                f"Epochs:{epoch + 1:5d} | "
                f"Batches per epoch: {i + 1:3d} | "
                f"Training loss: {running_loss / (i + 1):.10f} | "
                f"Test loss: {running_loss_test:.10f} |"
                f"Learning rate: {lr}"
            )

        # save 10 checkpoints
        # if not ((epoch + 1) % (epochs // 10)):
        # checkpoint the model parameters
        # checkpoint(model, optimizer, X_scaler, y_scaler, f"./checkpoints/epoch-{epoch}.pth")
        # saving best model found so far based on validation loss

        training_loss.append(running_loss / (i + 1))
        test_loss.append(running_loss_test)


    # load the best model for validation
    # resume(model, optimizer, f'./models_old/straight_{hidden_dim}_{layers}.pth')

    model = load_ANN(f"models/{folder}_{hidden_dim}_{layers}/{output}.pth")
    model = model.double()

    # Validate trained model using the test dataset
    with torch.no_grad():
        loss = 0
        for i, (inputs, labels) in enumerate(testloader):
            # calculate output by running through the network
            predictions = model(inputs)
            loss += F.l1_loss(predictions, labels)
        print(f"L1 Loss on test dataset: {loss / (i + 1):.5f}")


    # dont plot first epochs due to very large initial loss
    skip = 10

    epochss = np.arange(start_epoch, epochs)

    fig, ax1 = plt.subplots()
    ax1.plot(epochss[skip:], training_loss[skip:], label="Training loss")
    ax1.plot(epochss[skip:], test_loss[skip:], label="Test loss")
    ax1.set_xlabel(r"Epoch")
    ax1.set_ylabel(r"L1 loss")
    ax1.set_title(
        rf"Straight. Layers: {layers + 1}, Neurons 1st hidden layer: {hidden_dim}"
    )
    # ax1.set_ylim(0, 0.01)
    # ax1.set_xlim(100, epochs)
    plt.legend()

    plt.show()
