import matplotlib.pyplot as plt
import numpy as np
from thermo import polynomials
from thermo.chemical_equilibrium import equilibrium_OHC
import cantera as ct

from thermo import molar_fractions
from numba import jit

import time

gas = ct.Solution('gri30.yaml')

# pressure adjusted to fit fig 6.22 from "Simulating combustion" by Merker et al (100bar)
p = 70e5

num = 10
T = np.linspace(1500,3000, num)

i = 0

co2 = np.zeros(num)
h2o = np.zeros(num)
co = np.zeros(num)
o2 = np.zeros(num)
oh = np.zeros(num)
h2 = np.zeros(num)
o = np.zeros(num)
h = np.zeros(num)


N2 = np.zeros(num)
NO = np.zeros(num)
NO2 = np.zeros(num)

#inhouse mole fractions
mol_fracs = np.zeros([num, 10])


t_dummy = 1000
p_dummy = 1e5

# Assume stoichiometric combustion for now
equ = 1

fuel_type = "jetA"
R, M, x_N2, x_O2, x_CO2, x_H2O, x_Ar = molar_fractions(t_dummy, p_dummy, equ=equ, fuel_type=fuel_type)

x0 = np.array([0.0049914, 0.0011238, 0.0107067, 0.001178, 0.114522, 0.0085485, 0.098127, 0.03164, x_N2, x_Ar]) * p

# replace N2 with Ar
x_Ar = x_Ar + x_N2

start = time.time()
for t in T:
    #print(f"Temperature: {t} out of {T[-1]}")
    # cantera
    # with nitrogen in the air
    #gas.TPX = t, p, f'CO2:{x_CO2}, H2O:{x_H2O}, O2:{x_O2}, N2:{x_N2}, Ar:{x_Ar}'

    # since we want to look at the OHC system isolated, we replace all N2 with Ar
    gas.TPX = t, p, f'CO2:{x_CO2}, H2O:{x_H2O}, O2:{x_O2}, Ar:{x_Ar}'

    gas.equilibrate('TP')

    mixture = gas.mole_fraction_dict(threshold=1e-20)

    # OHC system (FAST)
    co2[i] = mixture["CO2"]
    h2o[i] = mixture["H2O"]
    co[i] = mixture["CO"]
    o2[i] = mixture["O2"]
    oh[i] = mixture["OH"]
    h2[i] = mixture["H2"]
    o[i] = mixture["O"]
    h[i] = mixture["H"]

    # Nitrogen (slow)
    #N2[i] = mixture["N2"]
    #NO[i] = mixture["NO"]
    #NO2[i] = mixture["NO2"]

    i = i + 1

end = time.time()
print(f"Sampling time cantera: {end - start} seconds")

start = time.time()

i = 0

for t in T:
    #print(f"Temperature: {t} out of {T[-1]}")

    # in house
    mol_fracs[i, :] = equilibrium_OHC(t, equ, p, fuel_type, x0) / p
    # guess for next calculation
    x0 = mol_fracs[i, :] * p


    i = i + 1
end = time.time()
print(f"Sampling time inhouse: {end - start} seconds")


# plot temperatures and pressure
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

lns1 = ax1.plot(T, co2 * 100, label="CO2")
lns2 = ax1.plot(T, h2o * 100, label="H2O")
lns3 = ax2.plot(T, co * 100, label="CO")
lns4 = ax2.plot(T, o2 * 100, label="O2")
lns5 = ax2.plot(T, oh * 100, label="OH")
lns6 = ax2.plot(T, h2 * 100, label="H2")
lns7 = ax2.plot(T, o * 100, label="O")
lns8 = ax2.plot(T, h * 100, label="H")

# Set y limits and grid visibility
for ax, ylim in zip([ax1, ax2], [16, 4]):
    ax.set_ylim(0, ylim)
    ax.grid(True)

ax1.set_xlim(1500, 3000)

# set which axis to which side
ax1.yaxis.tick_left()
ax2.yaxis.tick_right()

# added these three lines
lns = lns1+lns2+lns3+lns4+lns5+lns6+lns7+lns8
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="lower left")
ax1.set_title("Cantera")



# plotting in house results
fig, ax3 = plt.subplots()

ax4 = ax3.twinx()


# my own code
lns1 = ax3.plot(T, mol_fracs[:, 6] * 100, label="CO2")
lns2 = ax3.plot(T, mol_fracs[:, 4] * 100, label="H2O")
lns3 = ax4.plot(T, mol_fracs[:, 7] * 100, label="CO")
lns4 = ax4.plot(T, mol_fracs[:, 2] * 100, label="O2")
lns5 = ax4.plot(T, mol_fracs[:, 5] * 100, label="OH")
lns6 = ax4.plot(T, mol_fracs[:, 0] * 100, label="H2")
lns7 = ax4.plot(T, mol_fracs[:, 3] * 100, label="O")
lns8 = ax4.plot(T, mol_fracs[:, 1] * 100, label="H")

# Set y limits and grid visibility
for ax, ylim in zip([ax3, ax4], [16, 4]):
    ax.set_ylim(0, ylim)
    ax.grid(True)

ax3.set_xlim(1500, 3000)
ax3.set_title("OHC-system in house")

# set which axis to which side
ax3.yaxis.tick_left()
ax4.yaxis.tick_right()

# added these three lines
lns = lns1+lns2+lns3+lns4+lns5+lns6+lns7+lns8
labs = [l.get_label() for l in lns]
ax3.legend(lns, labs, loc="lower left")

plt.show()

# plot temperatures and pressure
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

lns1 = ax1.plot(T, N2 * 100, label="N2")
lns2 = ax2.plot(T, NO * 100, label="NO")
lns3 = ax2.plot(T, NO * 100, label="NO2")
# added these three lines
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="lower left")
plt.show()

# felsöker
#plt.plot(T, mol_fracs[:,4] * 100)
#plt.show()

# felsöker
#plt.plot(T, mol_fracs[:,8] * 100)
#plt.plot(T, mol_fracs[:,9] * 100)
#plt.show()