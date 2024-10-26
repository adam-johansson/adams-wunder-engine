from CCE.src import components, thermo, compressible, misc
from CCE.src.gas_props.air_properties import isa


# Take-off alt = 0 Mach = 0 disa = + 30
# Cruise alt = 10000 Mach = 0.7 disa = ?

alt = 11800
disa = 0
Mach = 0.7

# compressor pressure ratio
pr = 10

# compression ratio engine
cr = 10.5

eta = 0.85

# ISA table
pa, Ta, a = isa(alt, disa, False)  # static pressure, static temperature and speed of sound


print(f"p_a: {pa * 1e-5}, T_a: {Ta}")


# Total properties
p1, T1 = compressible.stagnation(pa, Ta, Mach)


print(f"p_1: {p1 * 1e-5}, T_1: {T1}")

# piston inlet temperature and pressure
T2 = T1 * (pr) ** ((1.4 - 1) / (1.4 * eta))
p2 = p1 * pr

# top dead center pressure estimate
p_tdc = p2 * (cr)**1.4

print(f"Inlet temperature: {T2} inlet pressure: {p2 * 1e-5} bar, p_tdc: {p_tdc * 1e-5} bar")