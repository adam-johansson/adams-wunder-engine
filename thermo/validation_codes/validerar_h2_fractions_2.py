import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

from thermo import cantera_combustion_fractions

fuel_type = "H2"

p_soc = 1e5
T_soc = 700
equ_sc = 0.5
equ_combustion = 1.0
num = 100

species = {S.name: S for S in ct.Species.list_from_file("gri30.yaml")}

ohc_species = [species[S] for S in ("H2O", "O2", "OH", "H2", "O", "H", "N2")]

gas = ct.Solution(thermo="ideal-gas", species=ohc_species)

equs = np.linspace(0.0, 0.99, num)
# temps = np.linspace(298,3000,num)

fracs_dir = np.zeros((num, 7))

# xi_N2_0, xi_CO2, xi_H2O_0, xi_CO, xi_O2_0, xi_OH_0, xi_H2_0, xi_O_0, xi_H_0 \
#    = cantera_combustion_fractions.molar_fractions_combustion(T_soc, p_soc, equ_sc, equ_combustion, fuel_type)

i = 0

T = 3000
for equ_sc in equs:
    # print(T, p_soc, equ_sc, equ_combustion, fuel_type)
    (
        xi_N2_dir,
        xi_CO2,
        xi_H2O_dir,
        xi_CO,
        xi_O2_dir,
        xi_OH_dir,
        xi_H2_dir,
        xi_O_dir,
        xi_H_dir,
    ) = cantera_combustion_fractions.molar_fractions_combustion(
        T, p_soc, equ_sc, equ_combustion, fuel_type
    )

    fracs_dir[i, :] = (
        np.array(
            [xi_N2_dir, xi_H2O_dir, xi_O2_dir, xi_OH_dir, xi_H2_dir, xi_O_dir, xi_H_dir]
        )
        * 100
    )

    # print(xi_N2_dir, xi_H2O_dir)
    # print(xi_N2 + xi_H2O + xi_O2 + xi_OH + xi_H2 + xi_O + xi_H)
    i += 1


labels1 = ["N2", "H2O", "O2"]
labels2 = ["OH", "H2", "O", "H"]
fs = 16

fig, ax1 = plt.subplots()
# ax1.plot(equs, fracs[:, :3], label=labels1)
ax1.plot(equs, fracs_dir[:, :3], label=labels1)
ax1.set_xlabel(r"initial equ", fontsize=fs)
ax1.set_ylabel(r"Mol fraction %", fontsize=fs)
ax1.legend()

fig, ax2 = plt.subplots()
# ax2.plot(equs, fracs[:, 3:], label=labels2)
ax2.plot(equs, fracs_dir[:, 3:], label=labels2)
ax2.set_xlabel(r"initial equ", fontsize=fs)
ax2.set_ylabel(r"Molar fraction %", fontsize=fs)
ax2.legend()

plt.show()
