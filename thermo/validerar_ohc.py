from thermo import equilibrium_OHC
import numpy as np


T = 3000  # Example temperature in Kelvin
equ = 1  # Example equivalence ratio
p = 10e5  # Example pressure in Pascals

x0 = np.array([0.0049914, 0.0011238, 0.0107067, 0.001178, 0.114522, 0.0085485, 0.098127, 0.03164, 0.7, 0.01]) * p
result = equilibrium_OHC(T, equ, p, x0)
# results are the mole fractions
print(result)