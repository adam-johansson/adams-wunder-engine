import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



rohr_validation_file = 'validation/H2/rohr_lateDI.csv'
p_validation_file = 'validation/H2/p_lateDI.csv'
htc_validation_file = 'validation/H2/htc_lateDI.csv'

rohr_file = 'simulation_data/rohr.csv'
P_file = 'simulation_data/P.csv'
phi_file = 'simulation_data/phi.csv'
phi_avg_file = 'simulation_data/phi2.csv'
htc_file = 'simulation_data/htc.csv'
headers = ['ca', 'rohr']
rohr_val = pd.read_csv(rohr_validation_file, header=None, names=headers)
headers = ['ca', 'p']
p_val = pd.read_csv(p_validation_file, header=None, names=headers)
headers = ['ca', 'htc']
htc_val = pd.read_csv(htc_validation_file, header=None, names=headers)

Q = pd.read_csv(rohr_file, header=None, names=['Q'])
phi = pd.read_csv(phi_file, header=None, names=['phi'])
p = pd.read_csv(P_file, header=None, names=['p'])
htc = pd.read_csv(htc_file, header=None, names=['htc'])
phi_avg = pd.read_csv(phi_avg_file, header=None, names=['phi'])

Q = Q['Q']
phi = phi['phi']
p = p['p']
htc = htc['htc']
phi_avg = np.array(phi_avg['phi'])


phi_deg = np.array(phi) * 180 / np.pi
rohr = np.diff(Q) / np.diff(phi_deg)
phi2 = (phi_deg[:-1] + phi_deg[1:]) / 2

mask = phi_avg < 420
htc = htc[mask]
phi_deg_htc = phi_avg[mask]
mask = phi_deg_htc > 320
htc = htc[mask]
phi_deg_htc = phi_deg_htc[mask]
phi_deg_htc = phi_deg_htc - 360

mask = phi2 < 410
rohr = rohr[mask]
phi2 = phi2[mask]
mask = phi2 > 310
rohr = rohr[mask]
phi2 = phi2[mask]
phi2 = phi2 - 360

mask = phi_deg < 410
p = p[mask]
phi_deg = phi_deg[mask]
mask = phi_deg > 310
p = p[mask]
phi_deg = phi_deg[mask]
phi_deg = phi_deg - 360

fs = 52
figsize = (20, 16)
res = 50

fig, ax1 = plt.subplots(figsize=figsize)
ax1.plot(rohr_val['ca'], rohr_val['rohr'], label='Validation', color="r", lw=4)
ax1.plot(phi2, rohr, label='Piston model', color="k", lw=4)
ax1.set_xlabel(r'Crank angle $\theta$ [$^{\circ}$]', fontsize=fs)
ax1.set_ylabel(r'Rate of heta release [J/$^{\circ} \theta$]', fontsize=fs)
ax1.set_title(r'$ROHR - \theta$ diagram', fontsize=fs)
ax1.set_xlim(-50, 50)
ax1.set_xticks([-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50])
ax1.tick_params(labelsize=fs)
plt.legend(loc='best', frameon=True, fontsize=fs)
ax1.grid()
plt.savefig('simulation_data/figures/rohr_validation_h2.png', dpi=res, bbox_inches='tight')


fig, ax2 = plt.subplots(figsize=figsize)
ax2.plot(p_val['ca'], p_val['p'], label='Validation', color="r", lw=4)
ax2.plot(phi_deg, p*1e-5, label='Piston model', color="k", lw=4)
ax2.set_xlabel(r'Crank angle $\theta$ [$^{\circ}$]', fontsize=fs)
ax2.set_ylabel(r'Pressure $p$ [bar]', fontsize=fs)
ax2.set_title(r'$p - \theta$ diagram', fontsize=fs)
ax2.set_xlim(-50, 50)
ax2.set_xticks([-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50])
ax2.tick_params(labelsize=fs)
plt.legend(loc='best', frameon=True, fontsize=fs)
ax2.grid()
plt.savefig('simulation_data/figures/p_validation_h2.png', dpi=res, bbox_inches='tight')


fig, ax3 = plt.subplots(figsize=figsize)
ax3.plot(htc_val['ca'], htc_val['htc'], label='Validation', color="r", lw=4)
ax3.plot(phi_deg_htc, htc, label='Piston model', color="k", lw=4)
ax3.set_xlabel(r'Crank angle $\theta$ [$^{\circ}$]', fontsize=fs)
ax3.set_ylabel(r'Heat transfer coefficient $\alpha$ [W/m^2K]', fontsize=fs)
ax3.set_title(r'$htc - \theta$ diagram', fontsize=fs)
ax3.set_xlim(-40, 60)
ax3.set_xticks([-40, -20, 0, 20, 40, 60])
ax3.tick_params(labelsize=fs)
plt.legend(loc='best', frameon=True, fontsize=fs)
ax3.grid()
plt.savefig('simulation_data/figures/htc_validation_h2.png', dpi=res, bbox_inches='tight')





mask = phi2 > 2
rohr = rohr[mask]
phi2 = phi2[mask]

mask = rohr_val['ca'] > 2
rohr_val_int = rohr_val['rohr'][mask]
rohr_val_ca = rohr_val['ca'][mask]

mask = rohr_val_ca < 40
rohr_val_int = rohr_val_int[mask]
rohr_val_ca = rohr_val_ca[mask]

qf_val = np.trapz(rohr_val_int, rohr_val_ca)
qf = np.trapz(rohr, phi2)

print(f'Qf_val: {qf_val}, qf: {qf}')

plt.show()
