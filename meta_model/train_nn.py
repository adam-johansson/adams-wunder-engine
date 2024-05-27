import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler



# import data  IN FUTURE JUST IMPORT ONE CSV FILE
xt = pd.read_csv('../piston_engine/surrogate_data/backup/h2_validated_woschni_18/xt_cleaned.csv', index_col=0)
xval = pd.read_csv('../piston_engine/surrogate_data/backup/h2_validated_woschni_18/xval_cleaned.csv', index_col=0)
yt = pd.read_csv('../piston_engine/surrogate_data/backup/h2_validated_woschni_18/yt_cleaned.csv', index_col=0)
yval = pd.read_csv('../piston_engine/surrogate_data/backup/h2_validated_woschni_18/yval_cleaned.csv', index_col=0)

# convert to numpy arrays
xt = pd.DataFrame.to_numpy(xt)
xval = pd.DataFrame.to_numpy(xval)
yt = pd.DataFrame.to_numpy(yt)
yval = pd.DataFrame.to_numpy(yval)


# focus firstly on first output (T_out)
yt_1 = yt[:, 0]

# convert to PyTorch tensors
# training data
Xt = torch.tensor(xt, dtype=torch.float32)
Yt = torch.tensor(yt_1, dtype=torch.float32).reshape(-1, 1)

# Normalize the data
X_scaler = MinMaxScaler()
X_scaled = X_scaler.fit_transform(Xt)
y_scaler = MinMaxScaler()
y_scaled = y_scaler.fit_transform(yt)

# test data
Xval = torch.tensor(xval, dtype=torch.float32)
Yval = torch.tensor(yval[:, 0], dtype=torch.float32).reshape(-1, 1)

model = nn.Sequential(
    nn.Linear(7, 32),  # input layer (number of inputs, number of neurons in hidden layer)
    nn.ReLU(),  # activation function
    nn.Linear(32, 32),  # hidden layer
    nn.ReLU(),
    nn.Linear(32, 32),  # hidden layer
    nn.ReLU(),
    nn.Linear(32, 1),  # output layer
    #nn.Sigmoid()
)


#loss_fn = nn.MSELoss()  # MSE maybe faster convergence but sensitive to outliers
loss_fn = nn.L1Loss()  # Not as sensitive to outliers
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # this is probably the best optimizer

n_epochs = 100
batch_size = 10

for epoch in range(n_epochs):
    for i in range(0, len(Xt), batch_size):
        Xbatch = Xt[i:i + batch_size]
        y_pred = model(Xbatch)
        ybatch = Yt[i:i + batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')


# prediction on validation set
Yval_pred = model(Xval)

# Mean square error on validation set
criterion = nn.MSELoss()
loss = criterion(Yval_pred, Yval)

print(loss)


xtest = Xval[0, :]
ytest = Yval[0,0]
print(ytest)
print(model(xtest))








