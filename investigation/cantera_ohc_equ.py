import numpy as np
import matplotlib.pyplot as plt
import cantera as ct

import sys
sys.path.append("./../")

from thermo import fuel_props, JETA_L, H2, mixture, molar_fractions

from piston_engine.engine import run_piston_engine




equ = 1.0
fuel_type = "jetA"


xi_N2_0, xi_O2_0, xi_CO2_0, xi_H2O_0, _, _ = molar_fractions(equ, fuel_type)


species = {S.name: S for S in ct.Species.list_from_file("gri30.yaml")}



species_names = ["CO2", "H2O", "O2", "CO", "OH", "H2", "O", "H", "N2", "NO", "N"]

#species_names = ["CO2", "H2O", "O2", "CO", "OH", "H2", "O", "H", "N2"]


ohc_species = [species[name] for name in species_names]
gas = ct.Solution(thermo="ideal-gas", species=ohc_species)
#gas = ct.Solution('gri30.yaml')


num = 100
T_array = np.linspace(1500,3000,num)

p = 75e5

fractions_array = np.zeros((num,11))

i = 0
for T in T_array:
  # Set gas composition

    composition = (
        f"CO2:{xi_CO2_0}, H2O:{xi_H2O_0}, O2:{xi_O2_0}, N2:{xi_N2_0}"
    )

    gas.TPX = T, p, composition
    gas.equilibrate("TP")


    # Extract equilibrium concentrations
    fractions = gas.mole_fraction_dict(threshold=1e-15)

    #print(fractions)
    xi_CO2 = fractions["CO2"]
    xi_H2O = fractions["H2O"]
    xi_CO = fractions["CO"]
    xi_H2 = fractions["H2"]

    try:
        xi_O = fractions["O"]
    except:
        xi_O = 0.0
    try:
        xi_H = fractions["H"]
    except:
        xi_H = 0.0
    try:
        xi_O2 = fractions["O2"]
    except:
        xi_O2 = 0.0
    try:
        xi_OH = fractions["OH"]
    except:
        xi_OH = 0.0
    try:
        xi_N2 = fractions["N2"]
    except:
        xi_N2 = 0.0
    try:
        xi_NO = fractions["NO"]
    except:
        xi_NO = 0.0
    try:
        xi_N = fractions["N"]
    except:
        xi_N = 0.0
    
    fractions_array[i,:] =  [xi_CO2, xi_H2O, xi_CO, xi_O2, xi_OH, xi_H2, xi_O, xi_H, xi_N, xi_NO, xi_N2]

    print(np.sum([xi_CO2, xi_H2O, xi_CO, xi_O2, xi_OH, xi_H2, xi_O, xi_H, xi_N, xi_NO, xi_N2]))

    i = i + 1


_, ax0 = plt.subplots()

ax0.plot(T_array, fractions_array[:,10]*100)
ax0.set_xlabel("Temperature [K]")
ax0.set_ylabel("mol fraction N2 [%]")
ax0.legend(["N2"])
ax0.grid()

_, ax1 = plt.subplots()

ax1.plot(T_array, fractions_array[:,0:2]*100)
ax1.set_xlabel("Temperature [K]")
ax1.set_ylabel("molar fraction [%]")
ax1.legend(["CO2", "H2O"])
ax1.grid()

_, ax2 = plt.subplots()

ax2.plot(T_array, fractions_array[:,2:5]*100)
ax2.set_xlabel("Temperature [K]")
ax2.set_ylabel("molar fraction [%]")
ax2.legend(["CO", "O2", "OH"])
ax2.grid()

_, ax3 = plt.subplots()

ax3.plot(T_array, fractions_array[:,5:8]*100)
ax3.set_xlabel("Temperature [K]")
ax3.set_ylabel("molar fraction [%]")
ax3.legend(["H2", "O", "H"])
ax3.grid()

_, ax4 = plt.subplots()
ax4.plot(T_array, fractions_array[:,8:10]*100)
ax4.set_xlabel("Temperature [K]")
ax4.set_ylabel("molar fraction N and NO [%]")
ax4.legend(["N", "NO"])
ax4.grid()

plt.show()