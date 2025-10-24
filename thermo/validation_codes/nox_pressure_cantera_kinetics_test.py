import sys

import cantera as ct

from thermo import molar_fractions

fuel_type = "jetA"


equ_combustion = 1.0

xi_N2_0, xi_O2_0, xi_CO2_0, xi_H2O_0, xi_Ar_0, _ = molar_fractions(
    equivalence_ratio=equ_combustion, fuel_type=fuel_type
)


T = 1800
p = 10e5

gas = ct.Solution("gri30.yaml")
gas.TPX = T, p, f'CO2:{xi_CO2_0},O2:{xi_O2_0},N2:{xi_N2_0},Ar:{xi_Ar_0}, NO:{1e-3}'
r = ct.IdealGasConstPressureReactor(gas)

# Create a thermal reservoir to maintain temperature
env = ct.Reservoir(gas)  # Large thermal reservoir

# Connect with a heat transfer wall
wall = ct.Wall(r, env)
wall.heat_transfer_coeff = 1e12  # Large heat transfer coefficient

# Add to reactor network
sim = ct.ReactorNet([r])
sim.verbose = True

# limit advance when temperature difference is exceeded
delta_T_max = 20.
r.set_advance_limit('temperature', delta_T_max)

dt_max = 1.e-5
t_end = 1e3 * dt_max
states = ct.SolutionArray(gas, extra=['t'])

print('{:10s} {:10s} {:10s} {:14s}'.format(
    't [s]', 'T [K]', 'P [Pa]', 'u [J/kg]'))
while sim.time < t_end:
    sim.advance(sim.time + dt_max)
    states.append(r.thermo.state, t=sim.time*1e3)
    print('{:10.3e} {:10.3f} {:10.3f} {:14.6f}'.format(
            sim.time, r.T, r.thermo.P, r.thermo.u))

# Plot the results if matplotlib is installed.

import matplotlib.pyplot as plt
plt.clf()
plt.subplot(2, 2, 1)
plt.plot(states.t, states.T)
plt.xlabel('Time (ms)')
plt.ylabel('Temperature (K)')

plt.subplot(2, 2, 2)
plt.plot(states.t, states.X[:, gas.species_index('NO')])
plt.xlabel('Time (ms)')
plt.ylabel('NO Mole Fraction')
plt.tight_layout()
plt.title(f"Pressure: {p*1e-5} bar")
#plt.ylim(1e-5, 1e-2)
#plt.xlim(1, 1e3)
#plt.xscale('log')
#plt.yscale('log')

plt.show()
