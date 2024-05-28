import cantera as ct
import numpy.ma
from timeit import default_timer as timer

from src.piston import valve_isentrop, walls, wiebe, polynomials
from src.piston import thermo_computations
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from NASAdata import load_NASA

from CoolProp.CoolProp import PropsSI



T = 800
P = 10e5


print(PropsSI('Cpmass','T', T, 'P',P,'Nitrogen'))
print(PropsSI('Cpmass','T', T, 'P',P,'Oxygen'))
print(PropsSI('Cpmass','T', T, 'P',P,'CO2'))
print(PropsSI('Cpmass','T', T, 'P',P,'H2O'))
print(PropsSI('Cpmass','T', T, 'P',P,'Air'))


print(PropsSI('Hmass','T', T, 'P',P,'Nitrogen'))
print(PropsSI('Hmass','T', T, 'P',P,'Oxygen'))
print(PropsSI('Hmass','T', T, 'P',P,'CO2'))
print(PropsSI('Hmass','T', T, 'P',P,'H2O'))
print(PropsSI('Hmass','T', T, 'P',P,'Air'))


#gas = ct.Solution("gas.yaml")

Ntot = 1 + 3.7274 + 0.0444
xo2 = 1/Ntot
xn2 = 3.7274/Ntot
xar = 0.0444/Ntot
print(xo2, xn2, xar, xo2 + xn2 + xar)