import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
from thermo import polynomials, molar_fractions
import CEA_Wrap as cea

from piston_engine.src.piston.nox_model_cantera import nox_calculations

num = 1000
# Universal gas constant from NASA polynomials pdf
R_univ = 8.314510  # J mol^-1 K^-1

# temperature
T = 2500
# pressure
p = 15 * 1e5

# volume
V = 1.0

# dont think this matters for this test
m = 1

T_z1 = np.ones((num, 1)) * T
m_z1 = np.ones((num, 1)) * m
p_z1 = np.ones((num, 1)) * p
V_z1 = np.ones((num, 1)) * V

fuel_type = "H2"

# air-fuel ratio in burned zone
lambda_z1 = 1.2

# crank angle
phi_z1 = np.linspace(0, 2*np.pi, num)

# rpm
rpm = 3000

# total mass outflow during one cycle
m_out_total = 1

# total average fuel-air-equivalence ratio (dont think it matters here)
equ = 1 / lambda_z1

# only for emission index
mf_tot = 1
equ_sc = 0.0


#ej viktig för detta
m_trapped = 1
equ_trapped = equ

no_ppm, dNOdt, no_times, EI_nox = nox_calculations(T_z1, m_z1, p_z1, V_z1, fuel_type, lambda_z1,
                                                                     phi_z1,
                                                                     rpm,
                                                                     m_out_total, mf_tot, equ_trapped, m_trapped,
                                                                     equ_sc)


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


burning = cea.TPProblem(pressure=p * 1e-5, pressure_units="bar", materials=[n2, o2, h2o, co2, fuel], massf=False,
                        phi=equ, temperature=T)
exhaust = burning.run()

t_flame = exhaust.t

# mass fractions of combustion products
fracs_cea = exhaust.prod_c

xi_NO = fracs_cea["NO"]

# concentration
c_NO = (xi_NO * p) / (R_univ * T)

#moles of NO
mole_NO = V * c_NO

# mass of NO
# (kg/mol molar mass)
_, _, _, _, M_NO = polynomials.NO(T, p)

# mass of NO in the cylinder
m_NO = mole_NO * M_NO

# NOx concentration mass of NO divided by total mass of gas leaving cylinder
no_concentration_mass = m_NO / m_out_total
no_concentration_mass = np.array([no_concentration_mass, no_concentration_mass]) * 1e6


fs = 16

fig, ax1 = plt.subplots()
ax1.plot(no_times * 1000, no_ppm)
ax1.plot([no_times[0], no_times[-1] * 1000], no_concentration_mass)
ax1.set_xlabel(r'Time [ms]', fontsize=fs)
ax1.set_ylabel(r'NO ppm (mass)', fontsize=fs)
plt.show()