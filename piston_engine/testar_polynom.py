import cantera as ct
from CoolProp.CoolProp import PropsSI
import numpy as np
from src.piston.polynomials import N2, O2, CO2, H2O, JETA, Ar, H2
import matplotlib.pyplot as plt


P = 100e5
Tarray = np.linspace(400, 5000, 1000)

cpN2_NASA = []
cpN2_cantera = []
cpN2_coolprop = []
cpO2_NASA = []
cpO2_cantera = []
cpO2_coolprop = []
cpCO2_NASA = []
cpCO2_cantera = []
cpCO2_coolprop = []
cpH2O_NASA = []
cpH2O_cantera = []
cpH2O_coolprop = []
cpair_NASA = []
cpair_cantera = []
cpair_coolprop = []
cp_H2_NASA = []
cp_H2_coolprop = []

hN2_NASA = []
hN2_cantera = []
hN2_coolprop = []
hO2_NASA = []
hO2_cantera = []
hO2_coolprop = []
hCO2_NASA = []
hCO2_cantera = []
hCO2_coolprop = []
hH2O_NASA = []
hH2O_cantera = []
hH2O_coolprop = []
hair_NASA = []
hair_cantera = []
hair_coolprop = []
h_H2_NASA = []
h_H2_coolprop = []

s_air_NASA = []
s_air_coolprop = []
s_air_cantera = []

s_N2_coolprop = []
s_O2_coolprop = []
s_Ar_coolprop = []
s_CO2_coolprop = []
s_H2O_coolprop = []
s_H2_NASA = []
s_H2_coolprop = []


nitrogen = ct.Solution('gri30.yaml')
compN2 = 'N2:1'
oxygen = ct.Solution('gri30.yaml')
compO2 = 'O2:1'
carbondioxide = ct.Solution('gri30.yaml')
compCO2 = 'CO2:1'
water = ct.Solution('gri30.yaml')
compwater = 'H2O:1'
air = ct.Solution('gri30.yaml')
compair = 'N2:0.78,O2:0.21,Ar:0.01'


cpco2, hco2,sco2, mass = CO2(298.15)
cph2o, hh2o,sh2o, mass = H2O(298.15)
cpn2, hn2, sn2, mass = N2(298.15)
cpo2, ho2, so2, mass = O2(298.15)
cpjet, hjet = JETA(298.15)

print(hco2)
print(hh2o)
print(hn2)
print(ho2)
print(hjet)


for T in Tarray:
    cpn2, hn2, sn2, mass = N2(T)
    cpo2, ho2, so2, mass = O2(T)
    cpco2, hco2, sco2, mass = CO2(T)
    cph2o, hh2o, sh2o, mass = H2O(T)
    cp_Ar, h_Ar, s_Ar, M_Ar = Ar(T)
    cp_H2, h_H2, s_H2, M_H2 = H2(T)
    cpair = 0.7811308101764534 * cpn2 + 0.20956452491722202 * cpo2 + 0.009304664906324658 * cp_Ar
    hair = 0.7811308101764534 * hn2 + 0.20956452491722202 * ho2 + 0.009304664906324658 * h_Ar
    s_air = 0.7811308101764534 * sn2 + 0.20956452491722202 * so2 + 0.009304664906324658 * s_Ar

    
    cpN2_NASA.append(cpn2)
    cpO2_NASA.append(cpo2)
    cpCO2_NASA.append(cpco2)
    cpH2O_NASA.append(cph2o)
    cpair_NASA.append(cpair)
    cp_H2_NASA.append(cp_H2)
    
    hN2_NASA.append(hn2)
    hO2_NASA.append(ho2)
    hCO2_NASA.append(hco2)
    hH2O_NASA.append(hh2o)
    hair_NASA.append(hair)
    h_H2_NASA.append(h_H2)

    s_air_NASA.append(s_air)
    s_H2_NASA.append(s_H2)

    
    
    nitrogen.TPX = T, P, compN2
    oxygen.TPX = T, P, compO2
    carbondioxide.TPX = T, P, compCO2
    water.TPX = T, P, compwater
    air.TPY = T, P, compair  # compositIon by mass
    
    cpN2_cantera.append(nitrogen.cp)
    cpO2_cantera.append(oxygen.cp)
    cpCO2_cantera.append(carbondioxide.cp)
    cpH2O_cantera.append(water.cp)
    cpair_cantera.append(air.cp)
    
    hN2_cantera.append(nitrogen.h)
    hO2_cantera.append(oxygen.h)
    hCO2_cantera.append(carbondioxide.h)
    hH2O_cantera.append(water.h)
    hair_cantera.append(air.h)

    s_air_cantera.append(air.s)
    
    cpN2_coolprop.append(PropsSI('Cpmass','T', T, 'P',P,'Nitrogen'))
    cpO2_coolprop.append(PropsSI('Cpmass','T', T, 'P',P,'Oxygen'))
    cpCO2_coolprop.append(PropsSI('Cpmass','T', T, 'P',P,'CO2'))
    cpH2O_coolprop.append(PropsSI('Cpmass','T', T, 'P',P,'H2O'))
    cpair_coolprop.append(PropsSI('Cpmass','T', T, 'P',P,'Air'))
    cp_H2_coolprop.append(PropsSI('Cpmass', 'T', T, 'P', P, 'Hydrogen'))
    
    hN2_coolprop.append(PropsSI('Hmass','T', T, 'P',P,'Nitrogen'))
    hO2_coolprop.append(PropsSI('Hmass','T', T, 'P',P,'Oxygen'))
    hCO2_coolprop.append(PropsSI('Hmass','T', T, 'P',P,'CO2'))
    hH2O_coolprop.append(PropsSI('Hmass','T', T, 'P',P,'H2O'))
    hair_coolprop.append(PropsSI('Hmass','T', T, 'P',P,'Air'))
    h_H2_coolprop.append(PropsSI('Hmass', 'T', T, 'P', P, 'Hydrogen'))

    s_air_coolprop.append(PropsSI('Smass', 'T', T, 'P', P, 'Air'))
    s_N2_coolprop.append(PropsSI('Smass', 'T', T, 'P', P, 'Nitrogen'))
    s_O2_coolprop.append(PropsSI('Smass', 'T', T, 'P', P, 'Oxygen'))
    s_Ar_coolprop.append(PropsSI('Smass', 'T', T, 'P', P, 'Argon'))
    s_CO2_coolprop.append(PropsSI('Smass', 'T', T, 'P', P, 'CO2'))
    s_H2O_coolprop.append(PropsSI('Smass', 'T', T, 'P', P, 'Water'))
    s_H2_coolprop.append(PropsSI('Smass', 'T', T, 'P', P, 'Hydrogen'))
   
    
   
"""" 
fig,ax1 = plt.subplots()
ax1.plot(Tarray,cpN2_NASA,label="NASA")
ax1.plot(Tarray,cpN2_cantera,label="Cantera")
ax1.plot(Tarray,cpN2_coolprop,label="Coolprop")
ax1.set_title("N2")
ax1.legend(loc = 2)

fig,ax2 = plt.subplots()
ax2.plot(Tarray,cpO2_NASA,label="NASA")
ax2.plot(Tarray,cpO2_cantera,label="Cantera")
ax2.plot(Tarray,cpO2_coolprop,label="Coolprop")
ax2.set_title("O2")
ax2.legend(loc = 2)

fig,ax3 = plt.subplots()
ax3.plot(Tarray,cpCO2_NASA,label="NASA")
ax3.plot(Tarray,cpCO2_cantera,label="Cantera")
ax3.plot(Tarray,cpCO2_coolprop,label="Coolprop")
ax3.set_title("CO2")
ax3.legend(loc = 2)

fig,ax4 = plt.subplots()
ax4.plot(Tarray,cpH2O_NASA,label="NASA")
ax4.plot(Tarray,cpH2O_cantera,label="Cantera")
ax4.plot(Tarray,cpH2O_coolprop,label="Coolprop")
ax4.set_title("H2O")
ax4.legend(loc = 2)


fig,ax5 = plt.subplots()
ax5.plot(Tarray,cpair_NASA,label="NASA")
ax5.plot(Tarray,cpair_cantera,label="Cantera")
ax5.plot(Tarray,cpair_coolprop,label="Coolprop")
ax5.set_title("cp - Air")
ax5.legend(loc = 2)


fig,ax6 = plt.subplots()
ax6.plot(Tarray,hN2_NASA,label="NASA")
ax6.plot(Tarray,hN2_cantera,label="Cantera")
ax6.plot(Tarray,hN2_coolprop,label="Coolprop")
ax6.set_title("h - N2")
ax6.legend(loc = 2)

fig,ax7 = plt.subplots()
ax7.plot(Tarray,hO2_NASA,label="NASA")
ax7.plot(Tarray,hO2_cantera,label="Cantera")
ax7.plot(Tarray,hO2_coolprop,label="Coolprop")
ax7.set_title("h - O2")
ax7.legend(loc = 2)

fig,ax8 = plt.subplots()
ax8.plot(Tarray,hCO2_NASA,label="NASA")
ax8.plot(Tarray,hCO2_cantera,label="Cantera")
ax8.plot(Tarray,hCO2_coolprop,label="Coolprop")
ax8.set_title("h - CO2")
ax8.legend(loc = 2)

fig,ax9 = plt.subplots()
ax9.plot(Tarray,hH2O_NASA,label="NASA")
ax9.plot(Tarray,hH2O_cantera,label="Cantera")
ax9.plot(Tarray,hH2O_coolprop,label="Coolprop")
ax9.set_title("h - H2O")
ax9.legend(loc = 2)

fig,ax10 = plt.subplots()
ax10.plot(Tarray,hair_NASA,label="NASA")
ax10.plot(Tarray,hair_cantera,label="Cantera")
ax10.plot(Tarray,hair_coolprop,label="Coolprop")
ax10.set_title("h - Air")
ax10.legend(loc = 2)

fig,ax11 = plt.subplots()
ax11.plot(Tarray,s_air_NASA,label="NASA")
ax11.plot(Tarray,s_air_cantera,label="Cantera")
ax11.plot(Tarray,s_air_coolprop,label="Coolprop")
ax11.set_title("s - Air")
ax11.legend(loc = 2)

"""

fig,ax12 = plt.subplots()
ax12.plot(Tarray, cp_H2_NASA, label="NASA")
ax12.plot(Tarray, cp_H2_coolprop, label="Coolprop")
ax12.set_title("cp - H2")
ax12.legend(loc=2)

fig,ax13 = plt.subplots()
ax13.plot(Tarray, h_H2_NASA, label="NASA")
ax13.plot(Tarray, h_H2_coolprop, label="Coolprop")
ax13.set_title("h - H2")
ax13.legend(loc=2)

fig,ax14 = plt.subplots()
ax14.plot(Tarray, s_H2_NASA, label="NASA")
ax14.plot(Tarray, s_H2_coolprop, label="Coolprop")
ax14.set_title("s - H2")
ax14.legend(loc=2)

plt.show()
    
    
    
    
    
