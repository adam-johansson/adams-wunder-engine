import numpy as np

# displacement
V = 0.475 *1e-3 #0.475 dm3

# bore
b = 0.09

# stroke
s = V /(np.pi * (b/2)**2)

imep = 11.5*1e5

rpm = 3000

rps = 3000 / 60

# divide with 2 if 4 stroke
power = imep * V * rps / 2

print(f"Displacement: {V*1e3} dm^3")
print(f"Stroke: {s*1000} mm")
print(f"Power: {power * 1e-3} [kW]")
