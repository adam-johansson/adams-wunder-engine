import numpy as np

import thermo

# stroke
s = 74.67 * 1e-3

# bore
b = 0.09

# cylinder volume
V = s * (b/2)**2 * np.pi

rpm = 3000

# period
rps = 3000 / 60

# frequency of gas changing (two revolutions per cycle)
f = rps / 2

# inlet and outlet conditions
p_in = 1.85e5  # inlet pressure
T_in = 273.15 + 32  # inlet temperature 32 celsius

# density at the inlet
_, _, _, _, R, _, _, _ = thermo.mixture(T_in, p_in, equ=0, fuel_type="H2", pure_fuel=True, fuel_equ_ratio=1/1.8)

rho_in = p_in / (R * T_in)

# mass in cylinder
m = rho_in * V

# mass flow
mdot = m * f

# stoichiometric fuel air ratio
far_s, _ = thermo.fuel_props("H2")

# fuel air ratio
far = far_s * (1 / 1.8)

# air flow
mdot_air = mdot / (1 + far)

# fuel flow
mdot_fuel = mdot - mdot_air

print(f"Mass flow tot: {mdot*1000} g/s")
print(f"Mass flow air: {mdot_air*1000} g/s")
print(f"Mass flow fuel: {mdot_fuel*1000} g/s")

