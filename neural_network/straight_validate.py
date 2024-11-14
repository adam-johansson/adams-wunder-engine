import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
import pandas as pd
from torcheval.metrics import R2Score
import numpy as np
import math

import matplotlib.pyplot as plt

from src import load

## NN-PP: Neural Network Post Processing


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
# Which scaler to use???  #minmax seems to perform worse (at least differently to standard)
X_scaler = StandardScaler()
#X_scaler = QuantileTransformer(output_distribution='normal')
X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)
y_scaler = StandardScaler()
#y_scaler = QuantileTransformer(output_distribution='normal')
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.fit_transform(y_test)


# Create new model and load states
model = load("models_old/straight_512_2.pth")
print(model)


# convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# test and train predictions
model.eval()
with torch.no_grad():
      y_pred_test = model(X_test)
      y_pred_train = model(X_train)

# convert back to actual values
y_pred_train = y_scaler.inverse_transform(y_pred_train.detach().numpy())
y_pred_test = y_scaler.inverse_transform(y_pred_test.detach().numpy())
y_test = y_scaler.inverse_transform(y_test.detach().numpy())
y_train = y_scaler.inverse_transform(y_train.detach().numpy())

# go from 12 cylinders to 1, for airflow, power and heat loss
divider = np.array([1, 1, 1/12, 1, 1, 1/12, 1/12, 1])
y_pred_train = y_pred_train * divider
y_pred_test = y_pred_test * divider
y_test = y_test * divider
y_train = y_train * divider


# root square error
RSE = np.sqrt(np.square(np.subtract(y_test, y_pred_test) ) )

# relative error
rel_error = np.divide(RSE, y_test)

# mean relative error
MRE = np.mean(rel_error, axis=0)


print(f'MRE T_2: {MRE[0] * 100:.2f} % \n'  
      f'MRE eff: {MRE[1] * 100:.2f} % \n' 
      f'MRE airflow: {MRE[2]*100:.2f} % \n' 
      f'MRE pmax: {MRE[3]*100:.2f} % \n'
      f'MRE T_max: {MRE[4]*100:.2f} % \n'
      f'MRE P: {MRE[5]*100:.2f} % \n'
      f'MRE Q : {MRE[6]*10:.2f} % \n'
      f'MRE p_tdc: {MRE[7]*100:.2f} %'
      )


t2 = np.vstack((y_test[:,0], y_pred_test[:,0])).T
eff = np.vstack((y_test[:,1], y_pred_test[:,1])).T
airflow = np.vstack((y_test[:,2], y_pred_test[:,2])).T
pmax = np.vstack((y_test[:,3], y_pred_test[:,3])).T
tmax = np.vstack((y_test[:,4], y_pred_test[:,4])).T
P = np.vstack((y_test[:,5], y_pred_test[:,5])).T
Q = np.vstack((y_test[:,6], y_pred_test[:,6])).T
ptdc = np.vstack((y_test[:,7], y_pred_test[:,7])).T

t2 = t2[t2[:,0].argsort()]
eff = eff[eff[:,0].argsort()]
airflow = airflow[airflow[:,0].argsort()]
pmax = pmax[pmax[:,0].argsort()]
tmax = tmax[tmax[:,0].argsort()]
P = P[P[:, 0].argsort()]
Q = Q[Q[:, 0].argsort()]
ptdc = ptdc[ptdc[:, 0].argsort()]

# trim some values
mask = eff[:, 0] > 0.01
eff = eff[mask]

mask = P[:, 0] > 20
P = P[mask]






fs = 24
figsize = (12, 8)
res = 50
dotsize = 2


fig, ax1 = plt.subplots(figsize=figsize)
ax1.scatter(t2[:,0], (t2[:,1] - t2[:,0]) / t2[:,0] * 100, label='Prediction', color="r", s=dotsize)
ax1.set_xlabel(r'Actual $T2$ [K]', fontsize=fs)
ax1.set_ylabel(r'Relative error [%]', fontsize=fs)
ax1.set_title(r'T2 relative error', fontsize=fs)
#ax2.set_xlim(660, 810)
#ax2.set_xticks([690, 720, 750, 780, 810])
#ax2.set_ylim(0, 55)
#ax2.set_yticks([0, 10, 20, 30, 40, 50])
ax1.tick_params(labelsize=fs)
plt.legend(loc='best', frameon=True, fontsize=fs)
#ax2.grid()
#plt.savefig('power_validation.pdf', dpi=res, bbox_inches='tight')

fig, ax2 = plt.subplots(figsize=figsize)
ax2.scatter(eff[:,0], (eff[:,1] - eff[:,0]) / eff[:,0] * 100, label='Prediction', color="r", s=dotsize)
ax2.set_xlabel(r'Actual $\eta$ [-]', fontsize=fs)
ax2.set_ylabel(r'Relative error [%]', fontsize=fs)
ax2.set_title(r'$\eta$ relative error', fontsize=fs)
ax2.tick_params(labelsize=fs)
plt.legend(loc='best', frameon=True, fontsize=fs)
ax2.grid()


fig, ax3 = plt.subplots(figsize=figsize)
ax3.scatter(airflow[:,0], (airflow[:,1] - airflow[:,0]) / airflow[:,0] * 100, label='Prediction', color="r", s=dotsize)
ax3.set_xlabel(r'Actual airflow [kg/s]', fontsize=fs)
ax3.set_ylabel(r'Relative error %', fontsize=fs)
ax3.set_title(r'Airflow relative error', fontsize=fs)
ax3.tick_params(labelsize=fs)
plt.legend(loc='best', frameon=True, fontsize=fs)

fig, ax4 = plt.subplots(figsize=figsize)
ax4.scatter(pmax[:,0], (pmax[:,1] - pmax[:,0]) / pmax[:,0] * 100, label='Prediction', color="r", s=dotsize)
ax4.set_xlabel(r'Actual $p_{max}$ [bar]', fontsize=fs)
ax4.set_ylabel(r'Relative error', fontsize=fs)
ax4.set_title(r'$p_{max}$ relative error', fontsize=fs)
ax4.tick_params(labelsize=fs)
plt.legend(loc='best', frameon=True, fontsize=fs)


fig, ax5 = plt.subplots(figsize=figsize)
ax5.scatter(tmax[:,0], (tmax[:,1] - tmax[:,0]) / tmax[:,0] * 100, label='Prediction', color="r", s=dotsize)
ax5.set_xlabel(r'Actual $T_{max}$ [K]', fontsize=fs)
ax5.set_ylabel(r'Relative error %', fontsize=fs)
ax5.set_title(r'$T_{max}$ relative error', fontsize=fs)
ax5.tick_params(labelsize=fs)
plt.legend(loc='best', frameon=True, fontsize=fs)

fig, ax6 = plt.subplots(figsize=figsize)
ax6.scatter(P[:,0], (P[:,1] - P[:,0]) / P[:,0] * 100, label='Prediction', color="r", s=dotsize)
ax6.set_xlabel(r'Actual $P_i$ [kW]', fontsize=fs)
ax6.set_ylabel(r'Relative error %', fontsize=fs)
ax6.set_title(r'$P_i$ relative error', fontsize=fs)
ax6.tick_params(labelsize=fs)
plt.legend(loc='best', frameon=True, fontsize=fs)

fig, ax7 = plt.subplots(figsize=figsize)
ax7.scatter(Q[:,0], (Q[:,1] - Q[:,0]) / Q[:,0] * 100, label='Prediction', color="r", s=dotsize)
ax7.set_xlabel(r'Actual $Q$ [kW]', fontsize=fs)
ax7.set_ylabel(r'Relative error %', fontsize=fs)
ax7.set_title(r'$Q$ relative error', fontsize=fs)
ax7.tick_params(labelsize=fs)
plt.legend(loc='best', frameon=True, fontsize=fs)

fig, ax8 = plt.subplots(figsize=figsize)
ax8.scatter(ptdc[:,0], (ptdc[:,1] - ptdc[:,0]) / ptdc[:,0] * 100, label='Prediction', color="r", s=dotsize)
ax8.set_xlabel(r'Actual $p_{tdc}$ [bar]', fontsize=fs)
ax8.set_ylabel(r'Relative error %', fontsize=fs)
ax8.set_title(r'$p_{tdc}$ relative error', fontsize=fs)
ax8.tick_params(labelsize=fs)
plt.legend(loc='best', frameon=True, fontsize=fs)


plt.show()


# create data for tikx
#power_trimmed[:,0], (power_trimmed[:,1] - power_trimmed[:,0]) / power_trimmed[:,0]

#power_diff = 100 * (power_trimmed[:,1] - power_trimmed[:,0]) / power_trimmed[:,0]
#power_diff = np.atleast_2d()
#power_true = power_trimmed[:,0]

#power_data = np.concatenate((power_true, power_diff), axis=1)





#np.savetxt("nn_output_data/power_diff.dat", power_data, fmt='%.5f')





