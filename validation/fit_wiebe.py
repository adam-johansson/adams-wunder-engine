import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize

from piston_engine.src.piston.wiebe import dmfdphi_single_mass_vector
from thermo.fuel import fuel_props


# load data from Rakopoulos
import os

dirname = os.path.dirname(__file__)
# filename_fuel = os.path.join(dirname, '../piston_engine/validation_output_data/NO_diesel/fuel_late.txt')
filename_fuel = os.path.join(
    dirname, "../piston_engine/validation_output_data/H2_water/heat.txt"
)
fuel_val = np.loadtxt(filename_fuel, delimiter=",")

phi = np.linspace(358 / 180 * np.pi, 380 / 180 * np.pi, 1000)
# late injection
# mf_tot = 17.76*1e-6
# early
# mf_tot = 17.38*1e-6

# h2 water 5.51e-6 is good
mf_tot = 5.9 * 1e-6


far_s, LHV = fuel_props("H2")


val = np.interp(phi * 180 / np.pi, fuel_val[:, 0], fuel_val[:, 1])


m = 1.0
phi_sc = 355
phi_cd = 20


def fun(x):

    m = x[0]
    phi_sc = x[1]
    phi_cd = x[2]

    # print(m, phi_sc, phi_cd)

    phi_sc = phi_sc / 180 * np.pi
    phi_cd = phi_cd / 180 * np.pi
    dmf = dmfdphi_single_mass_vector(phi, m, phi_sc, phi_cd, mf_tot)
    mf = integrate.cumulative_simpson(dmf, x=phi, axis=0, initial=0.0) * 1e6

    heat = dmf * LHV * np.pi / 180

    # minimize mse
    # mse = np.mean(np.sqrt((mf - val)**2))
    mse = np.mean(np.sqrt((heat - val) ** 2))
    # print(mse)

    return mse


x0 = np.array([m, phi_sc, phi_cd])
bounds = [(0.1, 15.0), (357, 361), (15, 40)]
res = optimize.minimize(fun, x0, tol=1e-9, bounds=bounds)
print(f"m: {res.x[0]}, sc: {res.x[1]}, cd: {res.x[2]}")

phi_sc = res.x[1] / 180 * np.pi
phi_cd = res.x[2] / 180 * np.pi
dmf = dmfdphi_single_mass_vector(phi, res.x[0], phi_sc, phi_cd, mf_tot)
mf = integrate.cumulative_simpson(dmf, x=phi, axis=0, initial=0.0)

fig, ax1 = plt.subplots()
ax1.plot(phi * 180 / np.pi, dmf * LHV * np.pi / 180)
ax1.plot(phi * 180 / np.pi, val)


phi_plot = np.linspace(340 / 180 * np.pi, 400 / 180 * np.pi, 1000)
dmf = dmfdphi_single_mass_vector(
    phi_plot, res.x[0], (res.x[1] / 180 * np.pi), (res.x[2] / 180 * np.pi), mf_tot
)
mf = integrate.cumulative_simpson(dmf, x=phi_plot, axis=0, initial=0.0) * 1e6
val_plot = np.interp(phi_plot * 180 / np.pi, fuel_val[:, 0], fuel_val[:, 1])

# plot heat release
# fig, ax2 = plt.subplots()
# ax2.plot(phi_plot * 180 /  np.pi, dmf * LHV)
# ax2.plot(phi_plot * 180 / np.pi, val_plot)

plt.show()
