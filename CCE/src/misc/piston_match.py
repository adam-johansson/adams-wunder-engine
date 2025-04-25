from piston_engine.engine import run_piston_engine
from scipy.optimize import fsolve, least_squares, brentq, minimize
from CCE.src import components
from CCE.src import auxiliaries
from timeit import default_timer as timer

import numpy as np
import pickle


def match_piston_engine(data, flags, match, power_req, flow_req):
    # First run through the CCE engine, before the mass flow is known
    if not match:
        (
            T45,
            power_piston,
            eta_th,
            air_flow,
            p_max,
            T_max,
            far,
            equ_trapped,
            induced_power,
        ) = run_piston_engine(data, flags)

        return (
            T45,
            power_piston,
            eta_th,
            air_flow,
            p_max,
            T_max,
            far,
            data[8],
            equ_trapped,
            data[30],
            induced_power,
        )

    else:  # now the power output from the piston engine has to match the power needed for the hpc

        def find_match(x):
            if (
                x[0] < 0.168 or x[0] > 0.17 or x[1] < 0.028 or x[1] > 0.030
            ):  # first is diameter second far
                print("otu of bounds")
                return np.array([1e9, 1e9])
            data_temp = data
            data_temp[8] = x[0]  # Use the new bore for the simulation
            data_temp[16] = x[0] * 0.1  # Maximum valve lift is 10% of bore
            data_temp[30] = x[1]  # fuel air ratio
            (
                T45,
                power_piston,
                eta_th,
                air_flow,
                p_max,
                T_max,
                far,
                equ_trapped,
                induced_power,
                friction_loss,
                aux_loss,
                heat_loss,
            ) = run_piston_engine(data_temp, flags)
            rpm = 60 * data[10] / (2 * x[0])  # rpm from mean velocity
            print(f"rpm, d, throttle: {rpm ,x[0], x[1]}")
            print(f"Air flow: {air_flow} [kg/s]")
            print(f"Required flow: {flow_req} [kg/s]")
            shaft_power = (
                induced_power - aux_loss - friction_loss
            )  # induced power - friction losses and auxiliaries
            print(f"Shaft power: {shaft_power * 1e-3} [kW]")
            print(f"Required power: {power_req * 1e-3} [kW]")
            return np.array([shaft_power - power_req * 1e-3, air_flow - flow_req])

        x0 = np.array([data[8], data[30]])
        x_match = fsolve(find_match, x0, xtol=1e-3)
        data[8] = x_match[0]
        data[30] = x_match[1]
        print(f"Matching diameter: {x_match[0]} [m]")
        print(f"Matching throttle: {x_match[1]} [-]")
        # This last simulation run could probably be done without, just taking the values from find_match
        (
            T45,
            power_piston,
            eta_th,
            air_flow,
            p_max,
            T_max,
            far,
            equ_trapped,
            induced_power,
            friction_loss,
            aux_loss,
            heat_loss,
        ) = run_piston_engine(data, flags)

    return (
        T45,
        power_piston,
        eta_th,
        air_flow,
        p_max,
        T_max,
        far,
        x_match[0],
        equ_trapped,
        x_match[1],
        induced_power,
        friction_loss,
        aux_loss,
        heat_loss,
    )


def match_piston_engine_cr(data, flags, match, power_req, flow_req):
    # First run through the CCE engine, before the mass flow is known
    if not match:
        (
            T45,
            power_piston,
            eta_th,
            air_flow,
            p_max,
            T_max,
            far,
            equ_trapped,
            induced_power,
        ) = run_piston_engine(data, flags)

        return (
            T45,
            power_piston,
            eta_th,
            air_flow,
            p_max,
            T_max,
            far,
            data[8],
            equ_trapped,
            data[30],
            induced_power,
        )

    else:  # now the power output from the piston engine has to match the power needed for the hpc

        def find_match(x):
            if x[0] > 20 or x[0] < 3:  # compression ratio
                return np.array([1e9, 1e9])
            data_temp = data
            data_temp[7] = x[0]  # Use the compression ratio
            (
                T45,
                power_piston,
                eta_th,
                air_flow,
                p_max,
                T_max,
                far,
                equ_trapped,
                induced_power,
            ) = run_piston_engine(data_temp, flags)
            print(f"cr: {x[0]}")
            print(f"Shaft power: {power_piston} [kW]")
            print(f"Required power: {power_req * 1e-3} [kW]")
            print(f"Air flow: {air_flow} [kg/s]")
            print(f"Required flow: {flow_req} [kg/s]")
            print(power_piston, air_flow, p_max, T45)
            return np.array([power_piston - power_req * 1e-3])

        x0 = data[7]
        x_match = fsolve(find_match, x0, xtol=1e-3)
        data[7] = x_match[0]
        print(f"Matching cr: {x_match[0]} [-]")
        # This last simulation run could probably be done without, just taking the values from find_match
        (
            T45,
            power_piston,
            eta_th,
            air_flow,
            p_max,
            T_max,
            far,
            equ_trapped,
            induced_power,
        ) = run_piston_engine(data, flags)

    return (
        T45,
        power_piston,
        eta_th,
        air_flow,
        p_max,
        T_max,
        far,
        x_match[0],
        equ_trapped,
        induced_power,
    )


def match_piston_new(data, flags, power_req, core_flow):

    def find_match(x):
        if data[32] == "H2":
            if x[0] < 0.01 or x[0] > 0.030:  # far
                print("Out of bounds")
                return np.array([1e9])
        elif data[32] == "jetA":
            if x[0] < 0.01 or x[0] > 0.055:  # far
                print("Out of bounds")
                return np.array([1e9])
        data_temp = data
        data_temp[30] = x[0]  # fuel air ratio
        (
            T45,
            power_piston,
            eta_th,
            air_flow,
            p_max,
            T_max,
            far,
            equ_trapped,
            induced_power,
            friction_loss,
            aux_loss,
            heat_loss,
            p_tdc,
        ) = run_piston_engine(data_temp, flags)
        print(f"throttle: {x[0]}")

        # pressurise circumventing flow
        m_circumvent = core_flow - air_flow
        pressure_circ, T_circumv, P_circumv = components.compressor(
            data[1], data[0] / 0.99, m_circumvent, 0.85, data[2] * 0.99 * 0.99
        )

        # power needed to pressurise the fuel
        fuel_type = data[32]
        fuel_flow = air_flow * far
        P_fuel_pump = components.fuel_pump(p_tdc, fuel_type, fuel_flow)

        shaft_power = induced_power - aux_loss - friction_loss - P_circumv - P_fuel_pump
        # induced power - friction losses - auxiliaries - pressurise fuel - circumventing flow pressure rise
        print(f"Shaft power: {shaft_power * 1e-3} [kW]")
        print(f"Required power: {power_req * 1e-3} [kW]")
        print(np.array([shaft_power - power_req]))
        return np.array([shaft_power - power_req])

    x0 = np.array([data[30]])
    x_match = fsolve(find_match, x0, xtol=1e-6)
    data[30] = x_match[0]
    print(f"Matching throttle: {x_match[0]} [-]")
    # This last simulation run could probably be done without, just taking the values from find_match
    (
        T45,
        power_piston,
        eta_th,
        air_flow,
        p_max,
        T_max,
        far,
        equ_trapped,
        induced_power,
        friction_loss,
        aux_loss,
        heat_loss,
        p_tdc,
    ) = run_piston_engine(data, flags)

    # power needed to compress circumventing air
    m_circumvent = core_flow - air_flow
    pressure_circ, T_circumv, P_circumv = components.compressor(
        data[1], data[0] / 0.99, m_circumvent, 0.85, data[2] * 0.99 * 0.99
    )

    # power needed to pressurise the fuel
    fuel_type = data[32]
    fuel_flow = air_flow * far
    P_fuel_pump = components.fuel_pump(p_tdc, fuel_type, fuel_flow)

    shaft_power = induced_power - aux_loss - friction_loss - P_circumv - P_fuel_pump

    return (
        T45,
        shaft_power,
        eta_th,
        air_flow,
        p_max,
        T_max,
        far,
        equ_trapped,
        x_match[0],
        induced_power,
        friction_loss,
        aux_loss,
        heat_loss,
        P_fuel_pump,
    )
