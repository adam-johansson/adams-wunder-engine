import cantera as ct
import numpy as np



T_soc = 900
p_soc = 10e5

# Get all of the Species objects defined in the GRI 3.0 mechanism
species = {S.name: S for S in ct.Species.list_from_file("gri30.yaml")}

# Create an ideal gas phase object with species representing complete combustion
complete_species = [species[S] for S in ("CH4", "O2", "N2", "CO2", "H2O")]
gas1 = ct.Solution(thermo="ideal-gas", species=complete_species)

phi = np.linspace(0.5, 2.0, 100)
T_complete = np.zeros(phi.shape)
for i in range(len(phi)):
    gas1.TP = T_soc, p_soc
    gas1.set_equivalence_ratio(phi[i], "CH4", "O2:1, N2:3.76")
    gas1.equilibrate("HP")
    T_complete[i] = gas1.T


# Create an IdealGas object including incomplete combustion species
gas2 = ct.Solution(thermo="ideal-gas", species=species.values())
T_incomplete = np.zeros(phi.shape)
for i in range(len(phi)):
    gas2.TP = T_soc, p_soc
    gas2.set_equivalence_ratio(phi[i], "CH4", "O2:1, N2:3.76")
    gas2.equilibrate("HP")
    T_incomplete[i] = gas2.T


import matplotlib.pyplot as plt

plt.plot(phi, T_complete, label="complete combustion", lw=2)
plt.plot(phi, T_incomplete, label="incomplete combustion", lw=2)
plt.grid(True)
plt.xlabel(r"Equivalence ratio, $\phi$")
_ = plt.ylabel("Temperature [K]")
plt.show()