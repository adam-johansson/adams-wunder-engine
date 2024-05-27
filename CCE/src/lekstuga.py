import numpy as np
from CCE.src import components
import matplotlib.pyplot as plt
from CCE.src import thermo
from CoolProp.CoolProp import PropsSI
from piston_engine.src.piston import polynomials

""""
p1 = 0.53e5
T1 = 300
pa = 23855
eta = 1.0
cp = 1005
gamma = 1.4
ms = 214.56

equ6 = 0.0

cfg_core = 1.0
cd_nozzle = 1.0

choked, Fs, pr, c2, c_ideal = components.nozzle_old(p1, T1, pa, eta, cp, gamma, ms)

print(Fs, c_ideal, choked, c2)

F8, v8_id, v8 = components.nozzle(p1, T1, pa, equ6, ms, cfg_core, cd_nozzle)


print(F8, v8_id, v8)


p1 = 1123.33e3
t1 = 1225.78
m1 = 11.71
equ1 = 0.448
#equ1 = 0.465
power = 9002.1868e3
w = power/m1

#p2, t13, t16, t2, m13, m16, equ3 = components.turbine(t1, p1, m1, equ1, power, 0.94, cooling=False)

#p2_simple, t2_simple = components.turbine_simple(p1, t1, 1148, 1.33, w, 0.94, 0.00, 0.0, 'Isen', cooling=False)
#p2_simple, t2_simple = components.turbine_simple(p1, t1, 1004, 1.404, w, 0.94, 0.00, 0.0, 'Isen', cooling=False)

p2, t13, t16, t2, m13, m16, equ3 = components.turbine(t1, p1, m1, equ1, power, 0.941, cooling=True,
                                                      t_cool=708.82, m1_cool=0.46, q_ngv=0.87)

#p2_simple, t2_simple = components.turbine_simple(p1, t1, 1148, 1.33, w, 0.94, 0.00, 0.1, 'Poly',
#                                                 cooling=True, alpha=0.8, T_cooling=708.82, cp_c=1005)

print(p2*1e-3, t13, t16, t2, m13, m16, equ3)

p1 = 1123.33e3
t1 = 1225.78
m1 = 11.71
equ1 = 0.5
power = 9015.87821744e3
w = power/m1

#p2, t13, t16, t2, m13, m16, equ3 = components.turbine(t1, p1, m1, equ1, power, 0.95, cooling=False)

#p2_simple, t2_simple = components.turbine_simple(p1, t1, 1148, 1.33, w, 0.95, 0.00, 0.0, 'Isen', cooling=False)

p2, t13, t16, t2, m13, m16, equ3 = components.turbine(t1, p1, m1, equ1, power, 0.94, cooling=True,
                                                      t_cool=708.82, m1_cool=0.46, q_ngv=0.87)

#p2_simple, t2_simple = components.turbine_simple(p1, t1, 1148, 1.33, w, 0.94, 0.00, 0.1, 'Poly',
#                                                 cooling=True, alpha=0.8, T_cooling=708.82, cp_c=1005)

print(p2*1e-3, t13, t16, t2, m13, m16)



cp, h, s, m = polynomials.CO2(298.15)
print(cp * m, h * m, s * m, m, 'CO2')
cp, h, s, m = polynomials.H2O(298.15)
print(cp * m, h * m, s * m, m, 'H2O')
cp, h, s, m = polynomials.Ar(298.15)
print(cp * m, h * m, s * m, m, 'Ar')


from components import burner, burner_old

p2ny, t2ny, f = burner(p1, 300, 0, 1800, 0.05, 0.99)
p2, t2, FAR = burner_old(p1, 300, 1800, 0.05, 0.99)
print( p2ny, t2ny, f)
print(p2, t2, FAR)




cp, h, s, M = polynomials.N2(298.15)
print(cp, h, s)


t = 386.4
p = 128.5e3 * 0.99
pr = 912.02e3/p
m = 11.85

p2, t2, w = components.compressor(t, p, eta=0.89823, pr=pr, m=m)
p2_simple, t2_simple, w_simple = components.compressor_simple(p, t, 0.935, pr, 1.404, 1003.5, 'Poly')

print(p2*1e-5, t2, w)
print(p2_simple*1e-5, t2_simple, w_simple*m)



T = np.linspace(200, 1000)
p = 100e5

ss = []
gammas = []
ss_cp = []

for t in T:
    R = 8.3144626  # J mol^-1 K^-1
    cp, h, s, M = thermo.properties(t, 0)
    ss.append(s)
    cv = cp - R/M
    gamma = cp/cv
    gammas.append(gamma)
    ss_cp.append(PropsSI('S', 'T', t, 'P', p, 'Air'))

plt.plot(T, gammas, label='polynom')
#plt.plot(T, ss_cp, label='coolprop')
plt.legend()
plt.show()



t1 = 1208.611
p1 = 1170.14
dp = 4.0/100
equ1 = 0.029268848143744124 / 0.0682
eta = 0.99
t2 = 1225.78

p2, t2, f = components.burner(p1, t1, equ1, t2, dp, eta)

print(p2, f)
"""

t = 700

from piston_engine.src.piston.thermo_computations import mixture
from CCE.src import thermo

equ = np.linspace(0, 1)

y = []
for x in equ:
    h, u, cp, cv, R, gamma, s = mixture(x, t, fuel_type='H2')
    y.append(R)


plt.plot(equ,y)
plt.show()

#Runiv = 8.3144626  # J mol^-1 K^-1
#cp, h, s, M = thermo.properties(t, equ, fuel_type='jetA')
#print(h)


from CCE.src.thermo.fuel_func import fuel_props

x, y = fuel_props('H2')

print(1/x, y)
