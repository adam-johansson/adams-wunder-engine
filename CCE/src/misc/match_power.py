from piston_engine.src.misc import post_processing
from scipy.optimize import fsolve, brentq
from CCE.src import components
from CCE.src import thermo_outdated
from timeit import default_timer as timer
from piston_engine.engine import run_piston_engine

import numpy as np


def match_power(data, meta_model, power_req, core_flow, surrogate_status):

    error = False

    # input to surrogate
    pin = data[0]
    Tin = data[1]
    bore = data[8]
    cr = data[7]
    p_ratio = data[2]
    v_mean = data[10]

    # print(pin, Tin)
    # fuel type
    fuel_type = data[32]
    far_s, LHV = thermo_outdated.fuel_props(fuel_type)

    def find_match(x):
        # change fuel air ratio and bore to match power and turbine inlet temperature

        far34 = x[0]  # far is varied to match target power

        # THESE LIMITS ARE FOR H2
        if far34 < 0.002923 or far34 > 0.02923:
            residual = np.array([1e6, 1e6])
            return residual

        if surrogate_status:
            # get the output of the surrogate
            piston_input = np.atleast_2d(
                np.array([pin, Tin, cr, bore, far34, p_ratio, v_mean])
            )
            air_flow = meta_model[2].predict_values(piston_input)[0][0]
            induced_power = meta_model[5].predict_values(piston_input)[0][0] * 1e3
            p_tdc = meta_model[7].predict_values(piston_input)[0][0] * 1e5
            # T34 = meta_model[0].predict_values(piston_input)[0][0]
        else:
            flags = ["sweep"]
            data[30] = x[0]  # fuel air ratio
            (
                T34,
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

        # pressurise circumventing flow
        m_circumvent = core_flow - air_flow
        if m_circumvent < 0:
            return np.array([1e6, 1e6])
        pressure_circ, T_circumv, P_circumv = components.compressor(
            Tin, pin / 0.99, m_circumvent, 0.85, p_ratio * 0.99 * 0.99
        )

        # mix circumventing flow
        # equ34 = far34 / far_s
        # m34 = air_flow * (1 + far34)  # outflow of piston engine (air + fuel)
        # m35 = m34 + m_circumvent  # flow after mixing
        # T35, equ35 = components.mix(m34, T34, equ34, m_circumvent, T_circumv, equ2=0, fuel_type=fuel_type)
        # far35 = equ35 * far_s

        # power needed to pressurise the fuel
        fuel_flow = (
            air_flow * far34
        )  # far_given is the same as far in the engine (at least it is supposed to be)
        P_fuel_pump = components.fuel_pump(p_tdc, fuel_type, fuel_flow)

        # things needed for aux and friction losses
        bsr = data[9]
        stroke = bore / bsr
        lv_max = bore * 0.1
        cylinders = data[31]
        # v_mean = data[10]
        rpm = v_mean / (2 * stroke) * 60
        rps = rpm / 60
        Vd_tot = stroke * bore**2 / 4 * np.pi * cylinders
        cycle = data[3]
        if cycle == "4T":
            n_r = 2
        else:
            n_r = 1

        # auxiliary losses and friction losses. these do not depend on the trapped fuel air ratio
        fmep, fmep_aux, fmep_pe_loss = post_processing.friction_patton(
            bore, rpm, stroke, v_mean, pin, cr, cylinders, lv_max
        )
        friction_loss = (
            fmep_pe_loss * Vd_tot * rps / n_r
        )  # friction losses for total engine all cylinders
        aux_loss = fmep_aux * Vd_tot * rps / n_r  # auxiliary losses

        shaft_power = induced_power - aux_loss - friction_loss - P_circumv - P_fuel_pump

        residual = np.array([shaft_power - power_req])
        if not surrogate_status:
            print(f"{residual[0]:.16f}, {far34:.16f}")

        return residual

    try:
        start = timer()
        x0 = np.array([0.027])  # p_ratio, far
        x_match, infodict, ier, mesg = fsolve(find_match, x0, full_output=True)
        # print(infodict)
        if ier == 1:
            sol = infodict["fvec"]
            # print(f'Fsolve worked. {sol}, {x_match[0]}')
        elif ier == 1:
            cost = 5e-6 + np.abs(infodict["fvec"]) * 1e-5
            error = True
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, error, 0
        end = timer()
        # print(f'find matching power: {end - start}, {x_match}')
    except ValueError:
        # print('problem with matching power')
        error = True
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, error, 0

    # This last simulation run could probably be done without, just taking the values from find_match
    # input to surrogate
    far34_final = x_match[0]  # far that matched the power
    p_ratio_final = p_ratio

    if surrogate_status:
        # get the output of the surrogate
        piston_input_final = np.atleast_2d(
            np.array([pin, Tin, cr, bore, far34_final, p_ratio_final, v_mean])
        )
        T34_final = meta_model[0].predict_values(piston_input_final)[0][0]
        # eta_th_final = meta_model[1].predict_values(piston_input_final)[0][0]
        air_flow_final = meta_model[2].predict_values(piston_input_final)[0][0]
        p_max_final = meta_model[3].predict_values(piston_input_final)[0][0]
        T_max_final = meta_model[4].predict_values(piston_input_final)[0][0]
        induced_power_final = (
            meta_model[5].predict_values(piston_input_final)[0][0] * 1e3
        )
        heat_loss_final = meta_model[6].predict_values(piston_input_final)[0][0] * 1e3
        p_tdc_final = meta_model[7].predict_values(piston_input_final)[0][0] * 1e5

    else:
        flags = ["sweep"]
        data[30] = far34_final  # fuel air ratio
        (
            T34_final,
            power_piston,
            eta_th,
            air_flow_final,
            p_max_final,
            T_max_final,
            far,
            equ_trapped,
            induced_power_final,
            friction_loss,
            aux_loss,
            heat_loss_final,
            p_tdc_final,
        ) = run_piston_engine(data, flags)

    # power needed to compress circumventing air
    m_circumvent_final = core_flow - air_flow_final
    # if m_circumvent_final < 0:
    #    print(f'Flow through piston engine is smaller than core flow with {m_circumvent_final} [kg/s].')
    p_circ_final, T_circ_final, P_circumv_final = components.compressor(
        Tin, pin / 0.99, m_circumvent_final, 0.85, p_ratio_final * 0.99 * 0.99
    )

    # mix circumventing flow
    equ34 = far34_final / far_s
    m34 = air_flow_final * (1 + far34_final)  # outflow of piston engine (air + fuel)
    m35 = m34 + m_circumvent_final  # flow after mixing
    T35_final, equ35 = components.mix(
        m34,
        T34_final,
        equ34,
        m_circumvent_final,
        T_circ_final,
        equ2=0,
        fuel_type=fuel_type,
    )
    far35 = equ35 * far_s

    p34 = pin * 0.99 * p_ratio_final * 0.99
    p35 = p34

    # power needed to pressurise the fuel
    fuel_flow_final = air_flow_final * far34_final
    P_fuel_pump_final = components.fuel_pump(p_tdc_final, fuel_type, fuel_flow_final)

    # things needed for aux and friction losses
    bsr = data[9]
    stroke = bore / bsr
    lv_max = bore * 0.1
    cylinders = data[31]
    # v_mean = data[10]
    rpm = v_mean / (2 * stroke) * 60
    rps = rpm / 60
    Vd_tot = stroke * bore**2 / 4 * np.pi * cylinders
    cycle = data[3]
    if cycle == "4T":
        n_r = 2
    else:
        n_r = 1

    # auxiliary losses and friction losses. these do not depend on the trapped fuel air ratio
    fmep, fmep_aux, fmep_pe_loss = post_processing.friction_patton(
        bore, rpm, stroke, v_mean, pin, cr, cylinders, lv_max
    )
    friction_loss = (
        fmep_pe_loss * Vd_tot * rps / n_r
    )  # friction losses for total engine all cylinders
    aux_loss = fmep_aux * Vd_tot * rps / n_r  # auxiliary losses

    # power out from engine after fuel pump and aux losses and friction and pressurising circumventing flow
    shaft_power_final = (
        induced_power_final
        - aux_loss
        - friction_loss
        - P_circumv_final
        - P_fuel_pump_final
    )

    # print(T35_final)
    # print(p_ratio_final)
    # print(f'Power matching N: {shaft_power_final - power_req}')

    return (
        T34_final,
        T35_final,
        p34,
        p35,
        m34,
        m35,
        far34_final,
        far35,
        shaft_power_final,
        air_flow_final,
        p_max_final,
        T_max_final,
        far34_final,
        induced_power_final,
        friction_loss,
        aux_loss,
        heat_loss_final,
        P_fuel_pump_final,
        bore,
        error,
        P_circumv_final,
    )
