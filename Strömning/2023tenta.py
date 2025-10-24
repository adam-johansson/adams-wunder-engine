


import numpy as np


Q = 0.09
D = 7.5e-2
rho = 998
mu = 1e-3
g = 9.82
L=30
hf = 200

f = 1e-2


D = (f * 8 * Q**2 * L / ((np.pi)**2 * g * hf) )**(1/5)
Re = (4 * rho * Q) / (mu * np.pi * D)
f2 = (1 / (2 * np.log10(Re*np.sqrt(f) ) - 0.8   ))**2

print(f2)
print(D)

D = (f2 * 8 * Q**2 * L / ((np.pi)**2 * g * hf) )**(1/5)
Re = (4 * rho * Q) / (mu * np.pi * D)
f3 = (1 / (2 * np.log10(Re*np.sqrt(f2) ) - 0.8   ))**2

print(f3)
print(D)

D = (f3 * 8 * Q**2 * L / ((np.pi)**2 * g * hf) )**(1/5)
Re = (4 * rho * Q) / (mu * np.pi * D)
f4 = (1 / (2 * np.log10(Re*np.sqrt(f3) ) - 0.8   ))**2

print(f4)
print(D)

D = (f4 * 8 * Q**2 * L / ((np.pi)**2 * g * hf) )**(1/5)
Re = (4 * rho * Q) / (mu * np.pi * D)
f5 = (1 / (2 * np.log10(Re*np.sqrt(f4) ) - 0.8   ))**2

print(f5)
print(D)