from piston_engine.src.piston import polynomials
from CoolProp.CoolProp import PropsSI
import matplotlib.pyplot as plt
import numpy as np

R = 8.3144626  # J mol^-1 K^-1
T = np.linspace(20, 700, 1000)
p = 1e5

ss = []
gammas = []
ss_cp = []
hs = []
cps = []
hs_cp = []
cps_cp = []

for t in T:
    #cp, h, s, M = polynomials.H2(t)
    #ss.append(s)
    #hs.append(h)
    #cps.append(cp)
    #cv = cp - R/M
    #gamma = cp/cv
    #gammas.append(gamma)
    ss_cp.append(PropsSI('S', 'T', t, 'P', p, 'Hydrogen'))
    hs_cp.append(PropsSI('Hmass', 'T', t, 'P', p, 'Hydrogen'))
    cps_cp.append(PropsSI('Cpmass', 'T', t, 'P', p, 'Hydrogen'))

#plt.plot(T, hs, label='polynom')
plt.plot(T, cps_cp, label='coolprop')
plt.legend()
#plt.show()


from CCE.src import thermo


wp = thermo.work_potential(1119, 777517, 0.7483, 23855, "H2")

print(wp)