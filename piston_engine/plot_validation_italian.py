import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from scipy.interpolate import CubicSpline


p_validation_file_08 = 'validation/H2/p_2500_08_validation.csv'
p_validation_file_06 = 'validation/H2/p_2500_06_validation.csv'
p_validation_file_04 = 'validation/H2/p_2500_04_validation.csv'

p_simulation_file_04 = 'validation/H2/p_2500_04_simulation_v2.csv'
p_simulation_file_06 = 'validation/H2/p_2500_06_simulation_v2.csv'
p_simulation_file_08 = 'validation/H2/p_2500_08_simulation_v2.csv'
#p_simulation_file_08 = 'simulation_data/P.csv'

Q_simulation_04_file = 'validation/H2/Q_2500_04_simulation_v2.csv'
Q_simulation_06_file = 'validation/H2/Q_2500_06_simulation_v2.csv'
Q_simulation_08_file = 'validation/H2/Q_2500_08_simulation_v2.csv'
#Q_simulation_08_file = 'simulation_data/Qapparent.csv'

Q_validation_file_08 = 'validation/H2/Q_2500_08_validation.csv'
Q_validation_file_06 = 'validation/H2/Q_2500_06_validation.csv'
Q_validation_file_04 = 'validation/H2/Q_2500_04_validation.csv'


rohr_file = 'simulation_data/rohr.csv'
phi_file = 'simulation_data/phi.csv'
phi_avg_file = 'simulation_data/phi2.csv'
#htc_file = 'simulation_data/htc.csv'
#headers = ['ca', 'Q']
#Q_val = pd.read_csv(Q_validation_file, header=None, names=headers)
headers = ['ca', 'p']
p_val_08 = pd.read_csv(p_validation_file_08, header=None, names=headers)
p_val_06 = pd.read_csv(p_validation_file_06, header=None, names=headers)
p_val_04 = pd.read_csv(p_validation_file_04, header=None, names=headers)

headers = ['ca', 'Q']
Q_val_08 = pd.read_csv(Q_validation_file_08, header=None, names=headers)
Q_val_06 = pd.read_csv(Q_validation_file_06, header=None, names=headers)
Q_val_04 = pd.read_csv(Q_validation_file_04, header=None, names=headers)

phi = pd.read_csv(phi_file, header=None, names=['phi'])
p_04 = pd.read_csv(p_simulation_file_04, header=None, names=['p'])
p_06 = pd.read_csv(p_simulation_file_06, header=None, names=['p'])
p_08 = pd.read_csv(p_simulation_file_08, header=None, names=['p'])

Q_04 = pd.read_csv(Q_simulation_04_file, header=None, names=['Q'])
Q_06 = pd.read_csv(Q_simulation_06_file, header=None, names=['Q'])
Q_08 = pd.read_csv(Q_simulation_08_file, header=None, names=['Q'])



phi_avg = pd.read_csv(phi_avg_file, header=None, names=['phi'])

phi = phi['phi']
p_04 = p_04['p']
p_06 = p_06['p']
p_08 = p_08['p']

Q_04 = Q_04['Q']
Q_06 = Q_06['Q']
Q_08 = Q_08['Q']

s = 0.085
v_mean = 2505 * 2 * 0.085 / 60
dtdphi_04 = s / (np.pi * v_mean)
v_mean = 2507 * 2 * 0.085 / 60
dtdphi_06 = s / (np.pi * v_mean)
v_mean = 2501 * 2 * 0.085 / 60
dtdphi_08 = s / (np.pi * v_mean)

phi_deg = phi * 180 / np.pi
Qd_04 = np.diff(Q_04) / np.diff(phi_deg)  # Joule per degree
Qd_04 = Qd_04 / dtdphi_04  # Joule per second
Qd_04 = Qd_04 * 1e-3  #kW

Qd_06 = np.diff(Q_06) / np.diff(phi_deg)  # Joule per degree
Qd_06 = Qd_06 / dtdphi_06  # Joule per second
Qd_06 = Qd_06 * 1e-3  #kW

Qd_08 = np.diff(Q_08) / np.diff(phi_deg)  # Joule per degree
Qd_08 = Qd_08 / dtdphi_08  # Joule per second
Qd_08 = Qd_08 * 1e-3  #kW
phi2 = (phi_deg[:-1] + phi_deg[1:]) / 2
phi2 = phi2[1:]

mask = phi2 < 450

Qd_04 = Qd_04[mask]
Qd_06 = Qd_06[mask]
Qd_08 = Qd_08[mask]
phi2 = phi2[mask]
mask = phi2 > 300
Qd_04 = Qd_04[mask]
Qd_06 = Qd_06[mask]
Qd_08 = Qd_08[mask]
phi2 = phi2[mask]
phi2 = phi2 + 360



#phi_avg = np.array(phi_avg['phi'])

mask = phi_deg < 450
p_04 = p_04[mask]
p_06 = p_06[mask]
p_08 = p_08[mask]
Q_08 = Q_08[mask]
phi_deg = phi_deg[mask]
mask = phi_deg > 300
p_04 = p_04[mask]
p_06 = p_06[mask]
p_08 = p_08[mask]
Q_08 = Q_08[mask]
phi_deg = phi_deg[mask]
phi_deg = phi_deg + 360

fs = 52
figsize = (24, 16)
res = 50


fig, ax1 = plt.subplots(figsize=figsize)
ax1.plot(Q_val_08['ca'], Q_val_08['Q'], label='Exp 08', color="r", lw=4, linestyle='dashed')
ax1.plot(phi2, Qd_08, label='Sim 08', color="r", lw=4)
ax1.plot(Q_val_06['ca'], Q_val_06['Q'], label='Exp 06', color="b", lw=4, linestyle='dashed')
ax1.plot(phi2, Qd_06, label='Sim 06', color="b", lw=4)
ax1.plot(Q_val_04['ca'], Q_val_04['Q'], label='Exp 04', color="k", lw=4, linestyle='dashed')
ax1.plot(phi2, Qd_04, label='Sim 04', color="k", lw=4)

ax1.set_xlabel(r'Crank angle $\theta$ [$^{\circ}$]', fontsize=fs)
ax1.set_ylabel(r'Apparent rate of heat release [kW]', fontsize=fs)
ax1.set_title(r'Apparent heat release rate over $\theta$', fontsize=fs)
ax1.set_xlim(710, 770)
ax1.set_xticks([720, 730, 740, 750, 760, 770])
ax1.set_ylim(0, 14)
ax1.set_yticks([0, 2, 4, 6, 8, 10, 12, 14])
ax1.tick_params(labelsize=fs)
plt.legend(loc='best', frameon=True, fontsize=fs)
ax1.grid()
#plt.savefig('simulation_data/figures/Q_validation_h2.png', dpi=res, bbox_inches='tight')



fig, ax2 = plt.subplots(figsize=figsize)
ax2.plot(p_val_08['ca'], p_val_08['p'], label='Exp 08', color="r", lw=4, linestyle='dashed')
ax2.plot(phi_deg, p_08*1e-5, label='Sim 08', color="r", lw=4)
ax2.plot(p_val_06['ca'], p_val_06['p'], label='Exp 06', color="b", lw=4, linestyle='dashed')
ax2.plot(phi_deg, p_06*1e-5, label='Sim 06', color="b", lw=4)
ax2.plot(p_val_04['ca'], p_val_04['p'], label='Exp 04', color="k", lw=4, linestyle='dashed')
ax2.plot(phi_deg, p_04*1e-5, label='Sim 04', color="k", lw=4)
ax2.set_xlabel(r'Crank angle $\theta$ [$^{\circ}$]', fontsize=fs)
ax2.set_ylabel(r'Pressure $p$ [bar]', fontsize=fs)
ax2.set_title(r'$p - \theta$ diagram', fontsize=fs)
ax2.set_xlim(660, 810)
ax2.set_xticks([690, 720, 750, 780, 810])
ax2.set_ylim(0, 55)
ax2.set_yticks([0, 10, 20, 30, 40, 50])
ax2.tick_params(labelsize=fs)
plt.legend(loc='best', frameon=True, fontsize=fs)
ax2.grid()
#plt.savefig('simulation_data/figures/p_validation_h2.png', dpi=res, bbox_inches='tight')

plt.show()


p_04 = np.array(p_04)
p_06 = np.array(p_06)
p_08 = np.array(p_08)

phi_val_04 = np.array(p_val_04['ca'])
p_val_04 = np.array(p_val_04['p'])
phi_val_06 = np.array(p_val_06['ca'])
p_val_06 = np.array(p_val_06['p'])
phi_val_08 = np.array(p_val_08['ca'])
p_val_08 = np.array(p_val_08['p'])

phi_val_04_interp = np.linspace(phi_val_04[0], phi_val_04[-1], 1000)
phi_val_06_interp = np.linspace(phi_val_06[0], phi_val_06[-1], 1000)
phi_val_08_interp = np.linspace(phi_val_08[0], phi_val_08[-1], 1000)


# linear interplation
#p_val_08_interp = np.interp(phi_val_08_interp, phi_val_08, p_val_08)

# cubic spline interpolation
spl_04 = CubicSpline(phi_val_04, p_val_04)
spl_06 = CubicSpline(phi_val_06, p_val_06)
spl_08 = CubicSpline(phi_val_08, p_val_08)

p_val_04_interp = spl_04(phi_val_04_interp)
p_val_06_interp = spl_06(phi_val_06_interp)
p_val_08_interp = spl_08(phi_val_08_interp)


mask = np.searchsorted(phi_deg, phi_val_04_interp, side='right')
mask = mask - 1
p_04_filtered = p_04[mask]
phi_deg_04 = phi_deg.reset_index(drop=True)

mask = np.searchsorted(phi_deg, phi_val_06_interp, side='right')
mask = mask - 1
p_06_filtered = p_06[mask]
phi_deg_06 = phi_deg.reset_index(drop=True)

mask = np.searchsorted(phi_deg, phi_val_08_interp, side='right')
mask = mask - 1
p_08_filtered = p_08[mask]
phi_deg_08 = phi_deg.reset_index(drop=True)


plt.scatter(phi_deg_04[mask], p_04_filtered * 1e-5)
plt.scatter(phi_val_04_interp, p_val_04_interp)
plt.show()

plt.scatter(phi_deg_06[mask], p_06_filtered * 1e-5)
plt.scatter(phi_val_06_interp, p_val_06_interp)
plt.show()

plt.scatter(phi_deg_08[mask], p_08_filtered * 1e-5)
plt.scatter(phi_val_08_interp, p_val_08_interp)
plt.show()


MSE_p_04 = np.square(np.subtract(p_04_filtered * 1e-5, p_val_04_interp)).mean()
MSE_p_06 = np.square(np.subtract(p_06_filtered * 1e-5, p_val_06_interp)).mean()
MSE_p_08 = np.square(np.subtract(p_08_filtered * 1e-5, p_val_08_interp)).mean()

RMSE_p_04 = math.sqrt(MSE_p_04)
RMSE_p_06 = math.sqrt(MSE_p_06)
RMSE_p_08 = math.sqrt(MSE_p_08)

RMSE_p_norm_04 = RMSE_p_04 / (p_04_filtered.max() - p_04_filtered.min()) * 1e5
RMSE_p_norm_06 = RMSE_p_06 / (p_06_filtered.max() - p_06_filtered.min()) * 1e5
RMSE_p_norm_08 = RMSE_p_08 / (p_08_filtered.max() - p_08_filtered.min()) * 1e5

print(f'04 MSE: {MSE_p_04}')
print(f'06 MSE: {MSE_p_06}')
print(f'08 MSE: {MSE_p_08}')


print(f'04 MSE: {RMSE_p_04}')
print(f'06 MSE: {RMSE_p_06}')
print(f'08 MSE: {RMSE_p_08}')


print(f'04 RMSE norm: {RMSE_p_norm_04}')
print(f'06 RMSE norm: {RMSE_p_norm_06}')
print(f'08 RMSE norm: {RMSE_p_norm_08}')


# this if for peak pressure validation
"""
print(p_08.max() * 1e-5)
print(p_val_08['p'].max())
print(p_08.max() * 1e-5 / p_val_08['p'].max() - 1)

print(p_06.max() * 1e-5)
print(p_val_06['p'].max())
print(1 - p_06.max() * 1e-5 / p_val_06['p'].max())

print(p_04.max() * 1e-5)
print(p_val_04['p'].max())
print((p_04.max() * 1e-5 / p_val_04['p'].max() - 1)*100)
"""
