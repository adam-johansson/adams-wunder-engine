import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    QuantileTransformer,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import torch.optim.lr_scheduler as lr_scheduler

import matplotlib.pyplot as plt

from src import clean_folder, checkpoint, resume, save, load, save_inference, load_ANN
from src import Data, NET_straight

from src import scale_data, PhysicsInformedLoss


# import data

folder = "jetA"

data = pd.read_csv("./input_data/" + folder + "/data.csv", index_col=0)

print(f"Number of datapoints total: {data.shape[0]}")

# remove datapoints that were not sampled (invalid input)
data = data[data.eff != 0]

print(f"Number of datapoints removed during sampling: {data.shape[0]}")

# Decide which data points should be input and output (not using eff for now)

# COULD HAVE POWER AS INPUT AND GEOMETRY AS OUTPUT
X = data[['p_in', 'T_in', 'PI', 'cr', 'bore',  'v_mean', 'T_fuel', 'far_goal']]
y = data[['power', 'heat_loss', 'nox', 'p_tdc', 'p_max', 'T_max', 'T_out', 'air_flow']]


# convert to numpy arrays
X = pd.DataFrame.to_numpy(X)
y = pd.DataFrame.to_numpy(y)

# Split the data into training and testing
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


batch_size = 32
epochs = 3000

# number of neurons of hidden layers
hidden_dim = 128

# number of hidden layers - 1
layers = 2

lr = 1e-3
weight_decay = 1e-1

# allows for starting from a checkpoint
start_epoch = 0

# early stopping threshold
epoch_threshold = 20

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



# FIGURE OUT HOW THIS CAN IMPLEMENT ENERGY CONSERVATION
# criterion to computes the loss between input and target
# criterion = nn.L1Loss()
# criterion = nn.MSELoss()

# Physics-informed loss function
criterion = PhysicsInformedLoss(
    x_scaler_params=x_scaler_params,
    y_scaler_params=y_scaler_params,
    fuel_type=folder,
    data_weight=1.0,
    physics_weight=1.0
)

"""criterion = PhysicsInformedLoss_vectorized(
    x_scaler_params=x_scaler_params,
    y_scaler_params=y_scaler_params,
    fuel_type=folder,
    data_weight=1.0,
    physics_weight=1.0
)"""

# optimizer that will be used to update weights and biases
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# learning rate scheduler (100 patience normal)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", patience=10, factor=0.5, min_lr=1e-4
)

# Storage for losses
training_loss = []
test_loss = []
physics_loss_history = []
data_loss_history = []

# Clean checkpoints if starting fresh
if start_epoch == 0:
    clean_folder("./checkpoints")

if start_epoch > 0:
    resume_epoch = start_epoch - 1
    resume(model, optimizer, f"./checkpoints/epoch-{resume_epoch}.pth")

best_loss = float('inf')
best_epoch = 0

# Training loop
for epoch in range(start_epoch, epochs):
    model.train()
    running_loss = 0.0
    running_data_loss = 0.0
    running_physics_loss = 0.0

    for i, (inputs, labels) in enumerate(trainloader):
        # Forward pass
        outputs = model(inputs)

        # Calculate physics-informed loss
        total_loss, data_loss, physics_loss = criterion(outputs, labels, inputs)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Accumulate losses
        running_loss += total_loss.item()
        running_data_loss += data_loss.item()
        running_physics_loss += physics_loss.item()

    # Validation
    model.eval()
    with torch.no_grad():
        val_total_loss = 0.0
        val_data_loss = 0.0
        val_physics_loss = 0.0
        val_batches = 0

        for inputs, labels in testloader:
            outputs = model(inputs)
            total_loss, data_loss, physics_loss = criterion(outputs, labels, inputs)

            val_total_loss += total_loss.item()
            val_data_loss += data_loss.item()
            val_physics_loss += physics_loss.item()
            val_batches += 1

        avg_val_loss = val_total_loss / val_batches

    # Update learning rate
    scheduler.step(avg_val_loss)
    lr = optimizer.param_groups[0]["lr"]

    # Save best model
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        best_epoch = epoch
        save_inference(
            model, f"models/{folder}_{hidden_dim}_{layers}_pinn.pth", scaler, x1, x2, y1, y2
        )

    # Early stopping
    elif epoch - best_epoch > epoch_threshold:
        print("Early stopped training at epoch %d" % epoch)
        epochs = epoch
        break

    # Store losses for plotting
    avg_train_loss = running_loss / len(trainloader)
    avg_data_loss = running_data_loss / len(trainloader)
    avg_physics_loss = running_physics_loss / len(trainloader)

    training_loss.append(avg_train_loss)
    data_loss_history.append(avg_data_loss)
    physics_loss_history.append(avg_physics_loss)
    test_loss.append(avg_val_loss)

    # Progress reporting
    if not ((epoch + 1) % (epochs // 3000)) or epoch == 0:
        print(
            f"Epoch: {epoch + 1:5d} | "
            f"Total Loss: {avg_train_loss:.6f} | "
            f"Data Loss: {avg_data_loss:.6f} | "
            f"Physics Loss: {avg_physics_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"LR: {lr:.2e}"
        )

# Load best model for final evaluation
model = load_ANN(f"models/{folder}_{hidden_dim}_{layers}_pinn.pth")
model = model.double()

# Final validation
model.eval()
with torch.no_grad():
    total_loss = 0
    total_data_loss = 0
    total_physics_loss = 0
    batches = 0

    for inputs, labels in testloader:
        predictions = model(inputs)
        loss, data_loss, physics_loss = criterion(predictions, labels, inputs)

        total_loss += loss.item()
        total_data_loss += data_loss.item()
        total_physics_loss += physics_loss.item()
        batches += 1

    print(f"\nFinal Test Results:")
    print(f"Total Loss: {total_loss / batches:.6f}")
    print(f"Data Loss: {total_data_loss / batches:.6f}")
    print(f"Physics Loss: {total_physics_loss / batches:.6f}")

# Plotting
skip = 0
epochss = np.arange(start_epoch, len(training_loss) + start_epoch)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot total losses
ax1.plot(epochss[skip:], training_loss[skip:], label="Training loss", alpha=0.7)
ax1.plot(epochss[skip:], test_loss[skip:], label="Validation loss", alpha=0.7)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Total Loss")
ax1.set_title(f"PINN Training - Layers: {layers + 1}, Neurons: {hidden_dim}")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot component losses
ax2.plot(epochss[skip:], data_loss_history[skip:], label="Data Loss", alpha=0.7)
ax2.plot(epochss[skip:], physics_loss_history[skip:], label="Physics Loss", alpha=0.7)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Component Loss")
ax2.set_title("Data vs Physics Loss Components")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()