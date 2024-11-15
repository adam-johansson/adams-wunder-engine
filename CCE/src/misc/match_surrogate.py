from piston_engine.src.misc import post_processing
from scipy.optimize import fsolve, brentq
from CCE.src import components
from CCE.src import thermo_outdated
from timeit import default_timer as timer

import numpy as np


def match_piston_surrogate(data, meta_model, power_req, core_flow):

    error = False

    # input to surrogate
    pin = data[0]
    Tin = data[1]
    cr = data[7]
    bore = data[8]
    p_ratio = data[2]

    # fuel type
    fuel_type = data[32]

    def find_bore(bore):
        # doesn't matter
        far_given = 0.01

        # get the output of the surrogate
        piston_input = np.atleast_2d(np.array([pin, Tin, cr, bore, far_given, p_ratio]))
        air_flow1 = meta_model[2].predict_values(piston_input)[0][0]

        return air_flow1 - core_flow

    def find_match(x):
        # change fuel air ratio in engine
        far_given = x  # far is varied to match power
        #print(far_given)


        # get the output of the surrogate
        piston_input = np.atleast_2d(np.array([pin, Tin, cr, bore, far_given, p_ratio]))
        air_flow = meta_model[2].predict_values(piston_input)[0][0]
        induced_power = meta_model[5].predict_values(piston_input)[0][0]*1e3
        p_tdc = meta_model[7].predict_values(piston_input)[0][0]*1e5
        #print(f'throttle: {x[0]}')

        # pressurise circumventing flow
        m_circumvent = core_flow - air_flow
        pressure_circ, T_circumv, P_circumv = \
            components.compressor(data[1], data[0] / 0.99, m_circumvent, 0.85, data[2] * 0.99 * 0.99)

        # power needed to pressurise the fuel
        fuel_flow = air_flow * far_given  # far_given is the same as far in the engine (at least it is supposed to be)
        P_fuel_pump = components.fuel_pump(p_tdc, fuel_type, fuel_flow)

        shaft_power = induced_power - aux_loss - friction_loss - P_circumv - P_fuel_pump
        # induced power - friction losses - auxiliaries - pressurise fuel - circumventing flow pressure rise
        #print(far_given)
        #print(f'Shaft power: {shaft_power * 1e-3} [kW]')
        #print(f'Required power: {power_req * 1e-3} [kW]')
        #print(shaft_power - power_req)
        return shaft_power - power_req

    #try:
    #    bore_match = brentq(find_bore, 0.08, 0.2)
    #except ValueError:
    #    print('problem with matching piston flow')
    #    error = True
    #    return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, error

    #bore = bore_match * 0.99

    # use bore that gives correct mass flow
    # things needed for aux and friction losses
    bsr = data[9]
    stroke = bore / bsr
    lv_max = bore * 0.1
    cylinders = data[31]
    v_mean = data[10]
    rpm = v_mean / (2 * stroke) * 60
    rps = rpm / 60
    Vd_tot = stroke * bore ** 2 / 4 * np.pi * cylinders
    cycle = data[3]
    if cycle == '4T':
        n_r = 2
    else:
        n_r = 1

    # auxiliary losses and friction losses. these do not depend on the trapped fuel air ratio
    fmep, fmep_aux, fmep_pe_loss = post_processing.friction_patton(bore, rpm, stroke, v_mean, pin, cr, cylinders,
                                                                   lv_max)
    friction_loss = fmep_pe_loss * Vd_tot * rps / n_r  # friction losses for total engine all cylinders
    aux_loss = fmep_aux * Vd_tot * rps / n_r  # auxiliary losses

    try:
        start = timer()
        # brentq is twice as fast as fsolve
        x_match = brentq(find_match, 0.002923, 0.02923)
        end = timer()
        print(f'find matching power: {end - start}, {x_match}')
    except ValueError:
        print('problem with matching power')
        error = True
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, error
    #print(f'Matching FAR: {x_match[0]} [-]')

    # This last simulation run could probably be done without, just taking the values from find_match
    # input to surrogate
    far_final = x_match  # far that matched the power

    # get the output of the surrogate
    piston_input_final = np.atleast_2d(np.array([pin, Tin, cr, bore, far_final, p_ratio]))
    T_out_final = meta_model[0].predict_values(piston_input_final)[0][0]
    eta_th_final = meta_model[1].predict_values(piston_input_final)[0][0]
    air_flow_final = meta_model[2].predict_values(piston_input_final)[0][0]
    p_max_final = meta_model[3].predict_values(piston_input_final)[0][0]
    T_max_final = meta_model[4].predict_values(piston_input_final)[0][0]
    induced_power_final = meta_model[5].predict_values(piston_input_final)[0][0]*1e3
    heat_loss_final = meta_model[6].predict_values(piston_input_final)[0][0]*1e3
    p_tdc_final = meta_model[7].predict_values(piston_input_final)[0][0]*1e5

    # power needed to compress circumventing air
    m_circumvent_final = core_flow - air_flow_final
    #if m_circumvent_final < 0:
    #    print(f'Flow through piston engine is smaller than core flow with {m_circumvent_final} [kg/s].')
    dummy, dummy, P_circumv_final = \
        components.compressor(data[1], data[0] / 0.99, m_circumvent_final, 0.85, data[2] * 0.99 * 0.99)

    # power needed to pressurise the fuel
    fuel_flow_final = air_flow_final * far_final
    P_fuel_pump_final = components.fuel_pump(p_tdc_final, fuel_type, fuel_flow_final)


    # power out from engine after fuel pump and aux losses and friction and pressurising circumventing flow
    shaft_power_final = induced_power_final - aux_loss - friction_loss - P_circumv_final -\
                        P_fuel_pump_final

    # fuel properties to get equivalence ratio
    far_s, LHV = thermo_outdated.fuel_props(fuel_type)
    equ_trapped = far_final / far_s

    return T_out_final, shaft_power_final, eta_th_final, air_flow_final, p_max_final, T_max_final, far_final,\
        equ_trapped, far_final, induced_power_final, \
        friction_loss, aux_loss, heat_loss_final, P_fuel_pump_final, bore, error




