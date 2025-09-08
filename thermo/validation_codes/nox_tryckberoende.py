import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
from thermo import polynomials, molar_fractions
import CEA_Wrap as cea

# temperature
T = 2500

fuel_type = "jetA"

# air-fuel ratio in burned zone
lambda_z1 = 1.0

# total average fuel-air-equivalence ratio (dont think it matters here)
equ = 1 / lambda_z1

# only for emission index
mf_tot = 1
equ_sc = 0.0

# chemical equilibrium

# air composition before combustion
x_N2, x_O2, x_CO2, x_H2O, x_Ar, _ = molar_fractions(equ_sc, fuel_type)
x_jetA = x_O2 * 17.75 * equ
x_H2 = x_O2 * 2.0 * equ


o2 = cea.Oxidizer("O2", temp=T, mols=x_O2)
n2 = cea.Oxidizer("N2", temp=T, mols=x_N2)
h2o = cea.Oxidizer("H2O", temp=T, mols=x_H2O)
co2 = cea.Oxidizer("CO2", temp=T, mols=x_CO2)


if fuel_type == "jetA":
    jetA = cea.Fuel("Jet-A(g)", temp=T, mols=x_jetA)
    fuel = jetA
elif fuel_type == "H2":
    h2 = cea.Fuel("H2", temp=T, mols=x_H2)
    fuel = h2


num = 500


fractions = np.zeros(num)
t_flames = np.zeros(num)

pressures = np.linspace(1e5,100e5,num)


i = 0
for p in pressures:

    burning = cea.TPProblem(
        pressure=p * 1e-5,
        pressure_units="bar",
        materials=[n2, o2, h2o, co2, fuel],
        massf=False,
        phi=equ,
        temperature=T,
    )
    exhaust = burning.run()

    t_flame = exhaust.t

    # molar fractions of combustion products
    fracs_cea = exhaust.prod_c

    xi_NO = fracs_cea["NO"]

    fractions[i] = xi_NO
    t_flames[i] = t_flame

    i = i + 1
    print(i)


_, ax1 = plt.subplots()
ax1.plot(pressures*1e-5, fractions)
ax1.set_xlabel("p [bar]")
ax1.set_ylabel("molar fraction NO [-]")

plt.show()



