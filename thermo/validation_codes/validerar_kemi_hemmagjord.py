import matplotlib.pyplot as plt
import numpy as np
from thermo import polynomials
import cantera as ct


# Får klura på denna. Nu varierar ej!!! det är så kallade stoichiometric coefficients!!!!!

# OBS: denna funkar bara när det inte är standard state på Gibbs
# Och även då funkar den ej

gas = ct.Solution("gri30.yaml")

p = 5e5
T = np.linspace(1000, 5000, 100)

i = 0

h2s = np.zeros(100)
o2s = np.zeros(100)
h2os = np.zeros(100)

for t in T:

    gas.TPX = t, p, "H2:1, O2:0.5"
    gas.equilibrate("TP")

    # initialise
    xi_H2 = 0.05
    xi_O2 = 0.05
    xi_H2O = 0.9

    diff = 1

    while diff > 0.001:

        # save solution from last iteration
        xi_H2_old = xi_H2
        xi_O2_old = xi_O2
        xi_H2O_old = xi_H2O

        # partial pressure is p_i = c * RT where c is the concentration

        # partial pressures
        p_H2 = xi_H2 * p
        p_O2 = xi_O2 * p
        p_H2O = xi_H2O * p

        _, _, _, g_H2, M_H2 = polynomials.H2(t, p_H2)
        _, _, _, g_O2, M_O2 = polynomials.O2(t, p_O2)
        _, _, _, g_H2O, M_H2O = polynomials.H2O(t, p_H2O)

        # convert to molar based
        g_H2 = g_H2 * M_H2
        g_O2 = g_O2 * M_O2
        g_H2O = g_H2O * M_H2O

        # nu är inte okänd!!!
        nu_O2 = 0.5
        nu_H2 = 1
        nu_H2O = 1

        nu_tot = -nu_O2 - nu_H2 + nu_H2O
        xi_H2 = -nu_H2 / nu_tot
        xi_O2 = -nu_O2 / nu_tot
        xi_H2O = nu_H2O / nu_tot

        diff_H2 = xi_H2 - xi_H2_old
        diff_O2 = xi_O2 - xi_O2_old
        diff_H2O = xi_H2O - xi_H2O_old

        diff = np.sum(np.abs([diff_H2, diff_O2, diff_H2O]))

    h2s[i] = xi_H2
    o2s[i] = xi_O2
    h2os[i] = xi_H2O

    i = i + 1

print(gas.mole_fraction_dict(threshold=0.001))
# print(gas.mole_fraction_dict(threshold=0.001)["H"])

plt.figure()
plt.plot(T, h2s, label="H2")
plt.plot(T, o2s, label="O2")
plt.plot(T, h2os, label="H2O")
plt.legend()
plt.show()
