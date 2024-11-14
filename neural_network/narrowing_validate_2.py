import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
import pandas as pd
from torcheval.metrics import R2Score
import numpy as np
import math

import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt

from src import load_inference

## NN-PP: Neural Network Post Processing


# Load the trained model
model = load_inference("models_old/narrowing_512_2.pth")
print(model)

# putting the model in output mode (this is done in the class already)
#model.eval()

X = pd.read_csv('./input_data/H2/x_cleaned.csv', index_col=0)
y = pd.read_csv('./input_data/H2/y_cleaned.csv', index_col=0)

# convert to numpy arrays
X = pd.DataFrame.to_numpy(X)
y = pd.DataFrame.to_numpy(y)

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=42)


# prediction on test data
y_pred_test = model.inference(X_test)


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
#mask = eff[:, 0] > 0.01
#eff = eff[mask]

#mask = P[:, 0] > 20
#P = P[mask]


t2_real = t2[:, 0]
t2_error = (t2[:, 1] - t2[:, 0]) / t2[:, 0] * 100

eff_real = eff[:, 0]
eff_error = (eff[:, 1] - eff[:, 0]) / eff[:, 0] * 100

airflow_real = airflow[:, 0]
airflow_error = (airflow[:, 1] - airflow[:, 0]) / airflow[:, 0] * 100

pmax_real = pmax[:, 0]
pmax_error = (pmax[:, 1] - pmax[:, 0]) / pmax[:, 0] * 100

tmax_real = tmax[:, 0]
tmax_error = (tmax[:, 1] - tmax[:, 0]) / tmax[:, 0] * 100

P_real = P[:, 0]
P_error = (P[:, 1] - P[:, 0]) / P[:, 0] * 100

Q_real = Q[:, 0]
Q_error = (Q[:, 1] - Q[:, 0]) / Q[:, 0] * 100

ptdc_real = ptdc[:, 0]
ptdc_error = (ptdc[:, 1] - ptdc[:, 0]) / ptdc[:, 0] * 100

def myplot(x, y, s, bins=1000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent





fig, axs = plt.subplots(2, 2)
fig.suptitle('T2')

sigmas = [0, 1, 2, 4] #zoom in
zoom_in = True
sigma = 8

for ax, s in zip(axs.flatten(), sigmas):
    if s == 0:
        ax.plot(t2_real, t2_error, 'k.', markersize=2)
        ax.set_title("Scatter plot")
    elif s == 1:
        img, extent = myplot(t2_real, t2_error, sigma)
        ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet, aspect='auto')
        ax.set_title("Smoothing with  $\sigma$ = %d" % sigma)
    else:
        img, extent = myplot(t2_real, t2_error, sigma)
        ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet, aspect='auto')
        ax.set_title("Smoothing with  $\sigma$ = %d" % sigma)
        if zoom_in:
            ax.set_ylim(-4 / s, 4 / s)




fig2, axs2 = plt.subplots(2, 2)
fig2.suptitle('Eff')

for ax, s in zip(axs2.flatten(), sigmas):
    if s == 0:
        ax.plot(eff_real, eff_error, 'k.', markersize=2)
        ax.set_title("Scatter plot")
    elif s == 1:
        img, extent = myplot(eff_real, eff_error, sigma)
        ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet, aspect='auto')
        ax.set_title("Smoothing with  $\sigma$ = %d" % sigma)
    else:
        img, extent = myplot(eff_real, eff_error, sigma)
        ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet, aspect='auto')
        ax.set_title("Smoothing with  $\sigma$ = %d" % sigma)
        if zoom_in:
            ax.set_ylim(-10 / s, 10 / s)
            ax.set_xlim(0.15, 0.5)


fig3, axs3 = plt.subplots(2, 2)
fig3.suptitle('Airflow')

for ax, s in zip(axs3.flatten(), sigmas):
    if s == 0:
        ax.plot(airflow_real, airflow_error, 'k.', markersize=2)
        ax.set_title("Scatter plot")
    elif s == 1:
        img, extent = myplot(airflow_real, airflow_error, sigma)
        ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet, aspect='auto')
        ax.set_title("Smoothing with  $\sigma$ = %d" % sigma)
    else:
        img, extent = myplot(airflow_real, airflow_error, sigma)
        ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet, aspect='auto')
        ax.set_title("Smoothing with  $\sigma$ = %d" % sigma)
        if zoom_in:
            ax.set_ylim(-10 / s, 10 / s)
            ax.set_xlim(0, 5)


fig4, axs4 = plt.subplots(2, 2)
fig4.suptitle('pmax')

for ax, s in zip(axs4.flatten(), sigmas):
    if s == 0:
        ax.plot(pmax_real, pmax_error, 'k.', markersize=2)
        ax.set_title("Scatter plot")
    elif s == 1:
        img, extent = myplot(pmax_real, pmax_error, sigma)
        ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet, aspect='auto')
        ax.set_title("Smoothing with  $\sigma$ = %d" % sigma)
    else:
        img, extent = myplot(pmax_real, pmax_error, sigma)
        ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet, aspect='auto')
        ax.set_title("Smoothing with  $\sigma$ = %d" % sigma)
        if zoom_in:
            ax.set_ylim(-10 / s, 10 / s)
            ax.set_xlim(0, 3000)


fig5, axs5 = plt.subplots(2, 2)
fig5.suptitle('Tmax')

for ax, s in zip(axs5.flatten(), sigmas):
    if s == 0:
        ax.plot(tmax_real, tmax_error, 'k.', markersize=2)
        ax.set_title("Scatter plot")
    elif s == 1:
        img, extent = myplot(tmax_real, tmax_error, sigma)
        ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet, aspect='auto')
        ax.set_title("Smoothing with  $\sigma$ = %d" % sigma)

    else:
        img, extent = myplot(tmax_real, tmax_error, sigma)
        ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet, aspect='auto')
        ax.set_title("Smoothing with  $\sigma$ = %d" % sigma)
        if zoom_in:
            ax.set_ylim(-2 / s,2 / s)
            ax.set_xlim(1500, 3500)

fig6, axs6 = plt.subplots(2, 2)
fig6.suptitle('Power')

for ax, s in zip(axs6.flatten(), sigmas):
    if s == 0:
        ax.plot(P_real, P_error, 'k.', markersize=2)
        ax.set_title("Scatter plot")
    elif s == 1:
        img, extent = myplot(P_real, P_error, sigma)
        ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet, aspect='auto')
        ax.set_title("Smoothing with  $\sigma$ = %d" % sigma)
    else:
        img, extent = myplot(P_real, P_error, sigma)
        ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet, aspect='auto')
        ax.set_title("Smoothing with  $\sigma$ = %d" % sigma)
        if zoom_in:
            ax.set_ylim(-20 / s, 20 / s)
            ax.set_xlim(0, 1000)


fig7, axs7 = plt.subplots(2, 2)
fig7.suptitle('Heat losses')

for ax, s in zip(axs7.flatten(), sigmas):
    if s == 0:
        ax.plot(Q_real, Q_error, 'k.', markersize=2)
        ax.set_title("Scatter plot")
    elif s == 1:
        img, extent = myplot(Q_real, Q_error, sigma)
        ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet, aspect='auto')
        ax.set_title("Smoothing with  $\sigma$ = %d" % sigma)
    else:
        img, extent = myplot(Q_real, Q_error, sigma)
        ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet, aspect='auto')
        ax.set_title("Smoothing with  $\sigma$ = %d" % sigma)
        if zoom_in:
            ax.set_ylim(-20 / s, 20 / s)
            ax.set_xlim(0, 1000)

fig8, axs8 = plt.subplots(2, 2)
fig8.suptitle('p top dead center')

for ax, s in zip(axs8.flatten(), sigmas):
    if s == 0:
        ax.plot(ptdc_real, ptdc_error, 'k.', markersize=2)
        ax.set_title("Scatter plot")
    elif s == 1:
        img, extent = myplot(ptdc_real, ptdc_error, sigma)
        ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet, aspect='auto')
        ax.set_title("Smoothing with  $\sigma$ = %d" % sigma)
    else:
        img, extent = myplot(ptdc_real, ptdc_error, sigma)
        ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet, aspect='auto')
        ax.set_title("Smoothing with  $\sigma$ = %d" % sigma)
        if zoom_in:
            ax.set_ylim(-10 / s, 10 / s)
            ax.set_xlim(0, 2000)

plt.show()

"""

fs = 24
figsize = (12, 8)
res = 50
dotsize = 1



fig, ax1 = plt.subplots(figsize=figsize)
ax1.scatter(t2_real, t2_error, label='Prediction', color="r", s=dotsize)
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
plt.show()


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
"""

# create data for tikx
#power_trimmed[:,0], (power_trimmed[:,1] - power_trimmed[:,0]) / power_trimmed[:,0]

#power_diff = 100 * (power_trimmed[:,1] - power_trimmed[:,0]) / power_trimmed[:,0]
#power_diff = np.atleast_2d()
#power_true = power_trimmed[:,0]

#power_data = np.concatenate((power_true, power_diff), axis=1)





#np.savetxt("nn_output_data/power_diff.dat", power_data, fmt='%.5f')





