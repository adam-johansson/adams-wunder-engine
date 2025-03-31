import matplotlib.pyplot as plt
import numpy as np
import CEA_Wrap as cea
import cantera as ct

equs = np.linspace(0,0.99,100)

N2s = []
O2s = []
H2Os = []
Ars = []

for equ_sc in equs:
    N_air = 1 + 3.7274 + 0.0444  # (specific?) mole of air. if CO2 is added don't forget to add it here
    x_O2_air = 1 / N_air  # molar fraction of O2
    x_N2_air = 3.7274 / N_air  # molar fraction of N2
    x_Ar_air = 0.0444 / N_air  # molar fraction of Ar


    N = 0.5 * equ_sc + 0.5 * (1 + x_N2_air / x_O2_air + x_Ar_air / x_O2_air)  # total number of moles in gas

    f1 = 0.5 * (x_N2_air / x_O2_air)  # N2
    f2 = 0.5 * (1 - equ_sc)  # O2
    f4 = equ_sc  # H2O
    f5 = 0.5 * (x_Ar_air / x_O2_air)  # Ar

    x_N2 = f1 / N  # molar fractions
    x_O2 = f2 / N
    x_H2O = f4 / N
    x_Ar = f5 / N

    N2s.append(x_N2)
    O2s.append(x_O2)
    H2Os.append(x_H2O)
    Ars.append(x_Ar)

    #print(x_Ar + x_H2O + x_N2 + x_O2)


plt.plot(N2s)
plt.plot(O2s)
plt.plot(H2Os)
plt.plot(Ars)
plt.show()

t_soc = 3000
Air = cea.Oxidizer("Air", temp=700, mols=1.0)
o2 = cea.Oxidizer("O2", temp=t_soc, mols=x_O2)
n2 = cea.Oxidizer("N2", temp=t_soc, mols=x_N2)
h2o = cea.Oxidizer("H2O", temp=t_soc, mols=x_H2O)


h2 = cea.Fuel("H2", temp=t_soc, mols=1.0)
fuel = h2

# print(x_H2O)

# HP problem is like a burner
# massf means output is in mass fraction
equ_combustion = 1.0

print(x_N2, x_O2, x_H2O)

burning = cea.HPProblem(pressure=1, pressure_units="bar", materials=[o2, n2, h2o, fuel], massf=False,
                        phi=equ_combustion)
#burning = cea.TPProblem(pressure=1, temperature=2500, pressure_units="bar", temperature_units="K", materials=[Air, fuel], massf=False,
#                        phi=equ_combustion)
exhaust = burning.run()

t_flame = exhaust.t

# mass fractions of combustion produc
print(t_flame)
print(exhaust.prod_c)
species = {S.name: S for S in ct.Species.list_from_file('gri30.yaml')}
ohc_species = [species[S] for S in ("H2O", "O2", "OH", "H2", "O", "H", "N2")]

# Load the H2 mechanism
gas2 = ct.Solution(thermo="ideal-gas", species=ohc_species)
#gas2 = ct.Solution('gri30.yaml')
gas2.TP = t_soc, 1e5

print(t_soc, equ_combustion, x_N2, x_O2, x_H2O)
gas2.set_equivalence_ratio(phi=equ_combustion, fuel="h2:1", oxidizer=f"O2:{x_O2}, N2:{x_N2}, H2O:{x_H2O}", basis="mole")

#gas2.equilibrate("HP")
gas2.equilibrate("TP")

fractions = gas2.mole_fraction_dict(threshold=1e-20)
#fractions = gas2.mole_fraction_dict()
print(fractions)

x = "kanske"

if x in ("ja", "nej"):
    print("hej")
elif x == "kanske":
    print("hejsan")