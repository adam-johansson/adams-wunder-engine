from piston_engine.src.piston.polynomials_outdated import O2, N2, CO2, Ar, H2O

from CoolProp.CoolProp import PropsSI

import numpy as np
import matplotlib.pyplot as plt


t = 1000
p = 10e5

N_air = 1 + 3.7274 + 0.0444  # (specific?) mole of air. if CO2 is added don't forget to add it here
x_O2_air = 1 / N_air  # molar fraction of O2
x_N2_air = 3.7274 / N_air  # molar fraction of N2
x_Ar_air = 0.0444 / N_air  # molar fraction of Ar

cpN2_cp = PropsSI('CPMASS', 'T', t, 'P', p, 'Nitrogen')
cpO2_cp = PropsSI('CPMASS', 'T', t, 'P', p, 'Oxygen')
cpAr_cp = PropsSI('CPMASS', 'T', t, 'P', p, 'Argon')
cpH2O_cp = PropsSI('CPMASS', 'T', t, 'P', p, 'H2O')
cpCO2_cp = PropsSI('CPMASS', 'T', t, 'P', p, 'CO2')

cpN2, hn2, sN2, M_N2 = N2(t, p)
cpO2, ho2, sO2, M_O2 = O2(t, p)
cpCO2, hco2, sCO2, M_CO2 = CO2(t, p)
cpH2O, hh2o, sH2O, M_H2O = H2O(t, p)
cp_Ar, h_Ar, s_Ar, M_Ar = Ar(t, p)

print(f'N2. NASA:{cpN2}, CoolProp: {cpN2_cp}')
print(f'O2. NASA:{cpO2}, CoolProp: {cpO2_cp}')
print(f'Ar. NASA:{cp_Ar}, CoolProp: {cpAr_cp}')
print(f'CO2. NASA:{cpCO2}, CoolProp: {cpCO2_cp}')
print(f'H2O. NASA:{cpH2O}, CoolProp: {cpH2O_cp}')


cp_Air_list = []
cp_Air_cp_list = []

equ_list = np.linspace(0.0, 1.0)
t_list = np.linspace(298.15, 1000)

fuel_type = 'jetA'
equ = 0

for equ in equ_list:

    if fuel_type == 'jetA':
        N = 5.75 * equ + 17.75 * (1 + x_N2_air / x_O2_air + x_Ar_air / x_O2_air)  # total number of moles in gas

        f1 = 17.75 * (x_N2_air / x_O2_air)  # N2
        f2 = 17.75 * (1 - equ)  # O2
        f3 = 12 * equ  # CO2
        f4 = 11.5 * equ  # H2O
        f5 = 17.75 * (x_Ar_air / x_O2_air)  # Ar

        x_N2 = f1 / N  # molar fractions
        x_O2 = f2 / N
        x_CO2 = f3 / N
        x_H2O = f4 / N
        x_Ar = f5 / N

        M = x_N2 * M_N2 + x_O2 * M_O2 + x_CO2 * M_CO2 + x_H2O * M_H2O + x_Ar * M_Ar  # molar mass of the fluid

        mu_N2 = x_N2 * (M_N2 / M)  # mass fraction of N2 in the fluid
        mu_O2 = x_O2 * (M_O2 / M)  # mass fraction of O2 in the fluid
        mu_CO2 = x_CO2 * (M_CO2 / M)  # mass fraction of CO2 in the fluid
        mu_H2O = x_H2O * (M_H2O / M)  # mass fraction of H2O in the fluid
        mu_Ar = x_Ar * (M_Ar / M)  # mass fraction of Ar in the fluid

    elif fuel_type == 'H2':
        N = 0.5 * equ + 0.5 * (1 + x_N2_air / x_O2_air + x_Ar_air / x_O2_air)  # total number of moles in gas

        f1 = 0.5 * (x_N2_air / x_O2_air)  # N2
        f2 = 0.5 * (1 - equ)  # O2
        f4 = equ  # H2O
        f5 = 0.5 * (x_Ar_air / x_O2_air)  # Ar

        x_N2 = f1 / N  # molar fractions
        x_O2 = f2 / N
        x_H2O = f4 / N
        x_Ar = f5 / N

        M = x_N2 * M_N2 + x_O2 * M_O2 + x_H2O * M_H2O + x_Ar * M_Ar  # molar mass of the fluid

        mu_N2 = x_N2 * (M_N2 / M)  # mass fraction of N2 in the fluid
        mu_O2 = x_O2 * (M_O2 / M)  # mass fraction of O2 in the fluid
        mu_H2O = x_H2O * (M_H2O / M)  # mass fraction of H2O in the fluid
        mu_Ar = x_Ar * (M_Ar / M)  # mass fraction of Ar in the fluid
        mu_CO2 = 0.0  # no CO2 for H2

    cp_Air    = cpN2 * mu_N2    + cpO2 * mu_O2    + cp_Ar * mu_Ar   + cpH2O * mu_H2O
    cp_Air_cp = cpN2_cp * mu_N2 + cpO2_cp * mu_O2 + cpAr_cp * mu_Ar + cpH2O_cp * mu_H2O

    cp_Air_list.append(cp_Air)
    cp_Air_cp_list.append(cp_Air_cp)


cp_Air_list = np.array(cp_Air_list)
cp_Air_cp_list = np.array(cp_Air_cp_list)

#s_Air_list = s_Air_list - s_Air_list[0]
#s_Air_cp_list = s_Air_cp_list - s_Air_cp_list[0]

plt.plot(equ_list, cp_Air_list, label='NASA')
plt.plot(equ_list, cp_Air_cp_list, label='CoolProp')
plt.legend()
plt.show()












