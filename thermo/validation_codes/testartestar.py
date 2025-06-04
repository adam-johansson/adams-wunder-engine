import matplotlib.pyplot as plt
import numpy as np

from thermo import mixture, fuel_props
from thermo.thermo_computations_old import mixture_old


far_s, _ = fuel_props("H2")

t = 800
p = 5e5

equ = 1 / 1.9


h, u, cp, cv, R, gamma, s, M = mixture(t, p, equ, "H2", include_fuel_in_reactants=True, fuel_air_equ_ratio=1/1.9)

print(h)

h, u, cp, cv, R, gamma, s, M = mixture_old(t, p, equ, "H2", pure_fuel=True, fuel_equ_ratio=1/1.9)

print(h)


h, u, cp, cv, R, gamma, s, M = mixture(t, p, equ, "H2")

print(h)

h, u, cp, cv, R, gamma, s, M = mixture_old(t, p, equ, "H2")

print(h)
