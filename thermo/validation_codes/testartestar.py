import matplotlib.pyplot as plt
import numpy as np

from thermo import mixture


equs = np.linspace(0,1,100)

t = 800
p = 1e5

hs = []
for equ in equs:
    h, u, cp, cv, R, gamma, s, M = mixture(t, p, equ, "jetA")
    hs.append(h)


plt.plot(equs, hs)
plt.show()