import numpy as np


def output_power(power, imep, friction_power, fmep, break_power, bmep, heat_power, equ_avg):
    print(f"Gas power on the piston: {power * 1e-3} [kW]")
    print(f"Indicated mean effective pressure: {imep * 1e-5} [bar]")
    print(f"Friction power: {friction_power * 1e-3} [kW]")
    print(f"Friction mean effective pressure: {fmep * 1e-5} [bar]")
    print(f"Engine total break power: {break_power * 1e-3} [kW]")
    print(f"Break mean effective pressure: {bmep * 1e-5} [bar]")
    print(f"Heating power into surroundings: {heat_power * 1e-3} [kW]")
    print(f"Equivalence ratio based on fuel and air flow {equ_avg}")
    return


def output_efficiencies(eta_th, hl, ):
    print(f"Thermal efficiency: {eta_th * 100} [%]")
    print(f"Percentage heat loss: {100 * hl} [%]")
    print(f"Percentage energy to exhaust: {100 - 100 * hl - 100 * eta_th} [%]")
    return


def output_thermo(phi, P, T, T_out, air_flow, m_in_IP, fuel_flow, mf):
    print(f"Mass flow of air: {air_flow * 1000} [g/s]")
    print(f"Mass of air inducted per cycle: {m_in_IP[-1][-1] * 1e3} [g]")
    print(f"Mass flow of fuel: {fuel_flow * 1000} [g/s]")
    print(f"Mass of fuel injected per cycle: {mf[-1][-1] * 1e6} [mg]")
    print(
        f"Maximum pressure: {np.max(P[-1]) * 1e-5} [bar] at crank angle: {phi[np.argmax(P[-1])] * 180 / np.pi}.")
    print(
        f"Maximum temperature: {np.max(T[-1])} [K] at crank angle: {phi[np.argmax(T[-1])] * 180 / np.pi}.")
    print(f"Outflow temperature: {T_out} [K]")
    return

def output_scavenging_validation(purity, residual_fraction, eta_trapping, eta_charging, delivery_ratio, eta_sc, equ_avg):
    print(f"Purity p: {purity}. NASA: 0.7343")
    print(f"Residual fraction: {residual_fraction}. Validation: 0.3164")
    print(f"Trapping efficiency: {eta_trapping}. Validation: 0.5941")
    print(f"Charging efficency: {eta_charging}. Validation: 0.3241")
    print(f"Delivery ratio: {delivery_ratio}. Validation: 0.5455")  # "air effiency"
    print(f"Scavenging efficency: {eta_sc}. Validation: 0.6965, Error: {eta_sc / 0.6965}")
    print(f"Equivalence ratio based on fuel and air flow {equ_avg}")
    return


def output_power_validation(power, imep, friction_power, fmep, break_power, bmep, heat_power):
    print(f"Gas power on the piston: {power * 1e-3} [kW], NASA: 161.15 kW, Error: {power * 1e-3 / 161.15}")
    print(f"Indicated mean effective pressure: {imep * 1e-5} [bar], NASA 21.698 bar")
    print(f"Friction power: {friction_power * 1e-3} [kW], NASA: 14.3174 kW")
    print(f"Friction mean effective pressure: {fmep * 1e-5} [bar], NASA 1.93 bar")
    print(f"Engine total break power: {break_power * 1e-3} [kW], NASA: 146.828 kW")
    print(f"Break mean effective pressure: {bmep * 1e-5} [bar], NASA 19.7673 bar")
    print(f"Heating power into surroundings: {heat_power * 1e-3} [kW]")
    return


def output_efficiencies_validation(eta_th, hl, ):
    print(f"Thermal efficiency: {eta_th * 100} [%], NASA 37.38%, Error: {eta_th * 100 / 37.38}")
    print(f"Percentage heat loss: {100 * hl} [%], NASA 8.11%, Error: {hl * 100 / 8.11}")
    print(f"Percentage energy to exhaust: {100 - 100 * hl - 100 * eta_th} [%], NASA 54.52%")
    return


def output_thermo_validation(phi, P, T, T_out, air_flow, m_in_IP, fuel_flow, mf):
    print(f"Mass flow of air: {air_flow * 1000} [g/s], NASA 296.83 g, Error: {air_flow * 1000 / 296.83}")
    print(f"Mass of air inducted per cycle: {m_in_IP[-1][-1] * 1e3} [g], NASA 2.90917 g")
    print(f"Mass flow of fuel: {fuel_flow * 1000} [g/s], NASA 10.16 g")
    print(f"Mass of fuel injected per cycle: {mf[-1][-1] * 1e6} [mg], NASA 99.8 mg")
    print(
        f"Maximum pressure: {np.max(P[-1]) * 1e-5} [bar] at crank angle: {phi[np.argmax(P[-1])] * 180 / np.pi}. NASA 191.98 bar at 368.8, Error: {np.max(P[-1]) * 1e-5 / 191.98}")
    print(
        f"Maximum temperature: {np.max(T[-1])} [K] at crank angle: {phi[np.argmax(T[-1])] * 180 / np.pi}. NASA 2539.7 K at 378.8, Error: {np.max(T[-1]) / 2539.7}")
    print(f"Outflow temperature: {T_out} [K], NASA  1143.05 K, Error: {T_out / 1143.05}")
    return
