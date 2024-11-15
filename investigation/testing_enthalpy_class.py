from thermo_outdated import PureSubstance

from CoolProp.CoolProp import PropsSI

import numpy as np
import matplotlib.pyplot as plt


t = 298.15
p = 1e5

N_air = 1 + 3.7274 + 0.0444  # (specific?) mole of air. if CO2 is added don't forget to add it here
x_O2_air = 1 / N_air  # molar fraction of O2
x_N2_air = 3.7274 / N_air  # molar fraction of N2
x_Ar_air = 0.0444 / N_air  # molar fraction of Ar

hN2_cp = PropsSI('HMASS', 'T', t, 'P', p, 'Nitrogen')
hO2_cp = PropsSI('HMASS', 'T', t, 'P', p, 'Oxygen')
hAr_cp = PropsSI('HMASS', 'T', t, 'P', p, 'Argon')
hH2O_cp = PropsSI('HMASS', 'T', t, 'P', p, 'Water')
hCO2_cp = PropsSI('HMASS', 'T', t, 'P', p, 'CarbonDioxide')

N2 = PureSubstance("N2", temperature=t)
O2 = PureSubstance("O2", temperature=t)
Ar = PureSubstance("Ar", temperature=t)
CO2 = PureSubstance("CO2", temperature=t)
H2O = PureSubstance("H2O", temperature=t)

M_N2 = N2.molar_mass
M_O2 = O2.molar_mass
M_CO2 = CO2.molar_mass
M_H2O = H2O.molar_mass
M_Ar = Ar.molar_mass

print(f'N2. NASA:{N2.enthalpy()}, CoolProp: {hN2_cp}')
print(f'O2. NASA:{O2.enthalpy()}, CoolProp: {hO2_cp}')
print(f'Ar. NASA:{Ar.enthalpy()}, CoolProp: {hAr_cp}')
print(f'CO2. NASA:{CO2.enthalpy() * CO2.molar_mass}, CoolProp: {hCO2_cp * CO2.molar_mass}')
print(f'H2O. NASA:{H2O.enthalpy() * H2O.molar_mass}, CoolProp: {hH2O_cp * H2O.molar_mass}')


cp_Air_list = []
cp_Air_cp_list = []

equ_list = np.linspace(0.0, 1.0)
t_list = np.linspace(298.15, 1000)

fuel_type = 'H2'
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

    h_Air = N2.enthalpy() * mu_N2 + O2.enthalpy() * mu_O2 + Ar.enthalpy() * mu_Ar + H2O.enthalpy() * mu_H2O + CO2.enthalpy() * mu_CO2
    h_Air_cp = hN2_cp * mu_N2 + hO2_cp * mu_O2 + hAr_cp * mu_Ar + hH2O_cp * mu_H2O + hCO2_cp * mu_CO2

    cp_Air_list.append(h_Air)
    cp_Air_cp_list.append(h_Air_cp)


cp_Air_list = np.array(cp_Air_list)
cp_Air_cp_list = np.array(cp_Air_cp_list)

#cp_Air_list = cp_Air_list - cp_Air_list[0]
#cp_Air_cp_list = cp_Air_cp_list - cp_Air_cp_list[0]

plt.plot(equ_list, cp_Air_list, label='NASA')
plt.plot(equ_list, cp_Air_cp_list, label='CoolProp')
plt.legend()
plt.show()












