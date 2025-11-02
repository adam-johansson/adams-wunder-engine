import matplotlib.pyplot as plt
import numpy as np

from thermo import mixture, fuel_props

from thermo import JETA_L, JETA_G


fuel_type = "jetA"

far_s, LHV = fuel_props(fuel_type)

print(far_s)

t = 300
p = 5e5

equ = 0.5

h_in, _, _, _, _, _, _, _ = mixture(t, p, 0.0, fuel_type=fuel_type, include_fuel_in_reactants=False, fuel_air_equ_ratio=0.0)
h_out, _, _, _, _, _, _, _ = mixture(t, p, equ, fuel_type=fuel_type, include_fuel_in_reactants=False, fuel_air_equ_ratio=0.0)
#_, h_fuel, _, _, _ = JETA_G(t, p)
_, h_fuel, _, _ = JETA_L(t)

term1 = h_in + h_fuel * 0.5* far_s
term2 =  h_out * (1 + 0.5 * far_s)
dh = -(term2 - term1) / (0.5 *far_s)

print(f"dh: {dh*1e-6} MJ/kg")
print(f"LHV: {LHV*1e-6} MJ/kg")