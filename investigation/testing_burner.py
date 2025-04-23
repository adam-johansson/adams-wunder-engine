from CCE.src import components
from CCE.src import thermo_outdated

from piston_engine.src.piston.polynomials_outdated import H2, JETA


p35 = 10e5
T35 = 1000
T4_req = 1800
dPcomb = 0.05
eta_b = 99.99 / 100
fuel_type = "H2"
t_fuel = 450

equ35 = 0.0

m31 = 1

# fuel props
far_s, LHV = thermo_outdated.fuel_props(fuel_type)

# far into burner
far_before_burner = equ35 * far_s

fuel_flow_piston = m31 * far_before_burner

# mass flow into burner
m35 = m31 + fuel_flow_piston


p4, T4, far_burner = components.burner(
    p35, T35, equ35, T4_req, dPcomb, eta_b, fuel_type, t_fuel=t_fuel
)


# m31 is pure air. After cooling flow removed but before piston.
m4 = m35 + far_burner * m31  # flow after burner. air + fuel piston + fuel burner

# burner fuel flow
fuel_flow_burner = far_burner * m31

# fuel air ratio after burner and piston engine
far_tot = far_burner + far_before_burner

equ4 = far_tot / far_s


#
_, hf, _, _ = H2(t_fuel, p4)

# lägg till hf här
fuel_power = fuel_flow_burner * (LHV + hf)

# enthalpy in burner
_, h_b_in, _, _ = thermo_outdated.properties(T35, p35, equ35, fuel_type)

# enthalpy out burner
_, h_b_out, _, _ = thermo_outdated.properties(T4, p4, equ4, fuel_type)

# energy increase in burner per second
P_burner = m4 * h_b_out - m35 * h_b_in

# fuel energy converted to enthalpy gain
frac = P_burner / fuel_power

print(f"Fraction of fuel energy to gas: {frac}. Fuel air ratio after burner: {far_tot}")
