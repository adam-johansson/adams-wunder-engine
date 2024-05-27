import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



Q_validation_file = 'validation/H2/burned_fuel_klepatz.csv'
p_validation_file = 'validation/H2/p_klepatz.csv'
#htc_validation_file = 'validation/H2/htc_lateDI.csv'

rohr_file = 'simulation_data/rohr.csv'
P_file = 'simulation_data/P.csv'
phi_file = 'simulation_data/phi.csv'
phi_avg_file = 'simulation_data/phi2.csv'
#htc_file = 'simulation_data/htc.csv'
headers = ['ca', 'Q']
Q_val = pd.read_csv(Q_validation_file, header=None, names=headers)
headers = ['ca', 'p']
p_val = pd.read_csv(p_validation_file, header=None, names=headers)
#headers = ['ca', 'htc']
#htc_val = pd.read_csv(htc_validation_file, header=None, names=headers)

Q = pd.read_csv(rohr_file, header=None, names=['Q'])
phi = pd.read_csv(phi_file, header=None, names=['phi'])
p = pd.read_csv(P_file, header=None, names=['p'])
#htc = pd.read_csv(htc_file, header=None, names=['htc'])
phi_avg = pd.read_csv(phi_avg_file, header=None, names=['phi'])

Q = Q['Q']
phi = phi['phi']
p = p['p']
#htc = htc['htc']
phi_avg = np.array(phi_avg['phi'])


phi_deg = np.array(phi) * 180 / np.pi


mask = phi_deg < 450
p = p[mask]
Q = Q[mask]
phi_deg = phi_deg[mask]
mask = phi_deg > 270
p = p[mask]
Q = Q[mask]
phi_deg = phi_deg[mask]
phi_deg = phi_deg - 360

fs = 52
figsize = (24, 16)
res = 50

"""
fig, ax1 = plt.subplots(figsize=figsize)
ax1.plot(Q_val['ca'], Q_val['Q'], label='Validation', color="r", lw=4)
ax1.plot(phi_deg, Q / np.array(Q)[-1], label='Piston model', color="k", lw=4)
ax1.set_xlabel(r'Crank angle $\theta$ [$^{\circ}$]', fontsize=fs)
ax1.set_ylabel(r'Burned fuel [-]', fontsize=fs)
ax1.set_title(r'$ROHR - \theta$ diagram', fontsize=fs)
ax1.set_xlim(-90, 90)
ax1.set_xticks([-90, -60, -30, 0, 30, 60, 90])
ax1.tick_params(labelsize=fs)
plt.legend(loc='best', frameon=True, fontsize=fs)
ax1.grid()
plt.savefig('simulation_data/figures/rohr_validation_h2.png', dpi=res, bbox_inches='tight')
"""


fig, ax2 = plt.subplots(figsize=figsize)
ax2.plot(p_val['ca'], p_val['p'], label='Calibration data', color="r", lw=4)
ax2.plot(phi_deg, p*1e-5, label='Piston model', color="k", lw=4)
ax2.set_xlabel(r'Crank angle $\theta$ [$^{\circ}$]', fontsize=fs)
ax2.set_ylabel(r'Pressure $p$ [bar]', fontsize=fs)
ax2.set_title(r'$p - \theta$ diagram', fontsize=fs)
ax2.set_xlim(-90, 90)
#ax2.set_xticks([-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50])
ax2.tick_params(labelsize=fs)
plt.legend(loc='best', frameon=True, fontsize=fs)
ax2.grid()
plt.savefig('simulation_data/figures/p_validation_h2.png', dpi=res, bbox_inches='tight')

plt.show()


p = np.array(p)
mask = np.searchsorted(phi_deg, p_val['ca'])
p_filtered = p[mask]

plt.scatter(phi_deg[mask], p_filtered * 1e-5)
plt.scatter(p_val['ca'], p_val['p'])
plt.show()


MSE_p = np.square(np.subtract(p_filtered * 1e-5, p_val['p'])).mean()

print(MSE_p)
import math
RMSE_p = math.sqrt(MSE_p)

print(RMSE_p)
RMSE_p_rel = RMSE_p / (p_filtered.max() - p_filtered.min()) * 1e5

print(RMSE_p_rel)

