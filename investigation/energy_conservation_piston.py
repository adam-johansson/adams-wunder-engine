from piston_engine import engine
import importlib

from thermo import mixture, fuel_props, polynomials


input_file = "4stroke_hydrogen_problemas"

input_dir = "piston_engine.input"
path = input_dir + "." + input_file

d = importlib.import_module(path)

flaggus = [
    "single",
    "output_all",
    "plot_convergence",
    "plot_essentials",
]  # normal case no plots


fuel_type = d.fuel

far_s, LHV = fuel_props(fuel_type)

# far_goal = 0.9 * far_s
# far_goal = 0.024
far_goal = d.far_goal

data = [
    d.p_in,
    d.T_in,
    d.p_ratio,
    d.cycle,
    d.thermo,
    d.cooling,
    d.opposed,
    d.cr,
    d.d,
    d.bsr,
    d.v_mean,
    d.lms,
    d.Twalls,
    d.ch,
    d.valve_timings,
    d.n_valve,
    d.lv_max,
    d.cd,
    d.eta_c,
    d.mf_tot,
    d.wa,
    d.wm,
    d.m_wiebe,
    d.phi_sc,
    d.phi_cd,
    d.T_fuel,
    d.p_fuel,
    d.it,
    d.wiebe_type,
    d.valve_type,
    far_goal,
    d.cylinders,
    d.fuel,
    d.c1,
    d.c4,
    d.c5,
]


(
    T4,
    work_piston,
    eta_th,
    air_flow,
    p_max,
    T_max,
    far_output,
    equ_trapped,
    induced_power,
    friction_loss,
    aux_loss,
    heat_loss,
    p_tdc,
    out_flow,
) = engine.run_piston_engine(indata=data, flags=flaggus)


# actual fuel flow into the engine
fuel_flow = far_goal * air_flow

mass_conservation = air_flow + fuel_flow - out_flow


if fuel_type == "H2":
    _, hf, _, _ = polynomials.H2(d.T_fuel, 1e5)
else:
    _, hf, _, _ = polynomials.JETA(d.T_fuel)

h_in, u, cp, cv, R, gamma, s, M = mixture(d.T_in, d.p_in, 0, fuel_type)
h_out, u, cp, cv, R, gamma, s, M = mixture(
    T4, d.p_in * d.p_ratio, far_output / far_s, fuel_type
)

energy_in = air_flow * h_in + fuel_flow * LHV + hf * fuel_flow


energy_out = induced_power + heat_loss + out_flow * h_out

energy_conservation = energy_in - energy_out

# air_flow is kg/s
# h_in is J/kg
# enthalpy flow is J/s = W

# fuel_flow = kg/s, LHV = J/kg, flow= W
print(f"Enthalpy flow intake: {air_flow * h_in * 1e-3} [kW]")
print(f"Fuel energy flow: {fuel_flow * LHV * 1e-3} [kW]")
print(f"Fuel enthalpy flow: {fuel_flow * hf * 1e-3} [kW]")

# Piston work: W
print(f"Piston power: {work_piston * 1e-3} [kW]")
print(f"Wall heating power: {heat_loss * 1e-3} [kW]")
print(f"Enthalpy flow out: {out_flow * h_out * 1e-3} [kW]")

print(f"far-goal %: {far_goal * 100}")
print(f"far-output %: {far_output * 100}")
print(f"Diff in far %: {(far_output - far_goal) * 100}")
print(f"Mass conservation: {mass_conservation} [kg/s]")
print(f"Energy conservation: {energy_conservation * 1e-3} [kW]")

print(equ_trapped)
