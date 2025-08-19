from piston_engine.src.misc import post_processing
from scipy.optimize import fsolve, brentq
from CCE.src import components
from thermo import fuel_props
from timeit import default_timer as timer
from piston_engine.engine import run_piston_engine
from CCE.src.surrogate import nn_output
from piston_engine.src.misc.seiliger import seiliger

import numpy as np


def match_power_nn(input, meta_model, power_req, core_flow, surrogate_status, p_loss_in, p_loss_out):

    error = False
    outputs = 22

    # input to surrogate
    pin = input["p_in"]
    Tin = input["T_in"]
    bore = input["bore"]
    cr = input["cr"]
    p_ratio = input["p_ratio"]
    v_mean = input["v_mean"]
    T_fuel = input["T_fuel"]


    # needed to convert surrogate data (single cylinder)
    cylinders = input["cylinders"]

    # print(pin, Tin)
    # fuel type
    fuel_type = input["fuel"]
    far_s, LHV = fuel_props(fuel_type)

    if surrogate_status:
        if pin < 2e5 or pin > 30e5 or Tin < 250 or Tin > 1000:
            # LÄGG TILL SEILIGER MAX TRYCK
            print("input to surrogate outside limits")
            error = True
            listofzeros = [0] * outputs
            listofzeros[-1] = error
            return listofzeros
    def find_match(x):
        # change fuel air ratio and bore to match power and turbine inlet temperature

        far34 = x[0]  # far is varied to match target power

        if far34 < far_s / 3.0:
            residual = 1e8 + np.abs(far34 - far_s / 3.0) * 1e8
            return residual

        elif far34 > far_s / 1.2:
            residual = 1e8 + np.abs(far34 - far_s / 1.1) * 1e8
            return residual

        if surrogate_status:
            # get the output of the surrogate

            piston_input = np.atleast_2d(
                np.array([pin, Tin, p_ratio, cr, bore, v_mean, T_fuel, far34])
            )
            # T34, air_flow, p_max, T_max, induced_power, heat_loss, p_tdc = nn_output(piston_input, meta_model)

            # check if input is within the limits of the surrogate model
            pmax_seiliger = seiliger(pin, Tin, cr, far34, bore, fuel_type)

            # if predicted pressure over 600 bar
            if pmax_seiliger > 400 * 1e5:
                return 1e6

            # use meta model to get outputs from the piston egnine
            output = meta_model.inference(piston_input)[0]

            indicated_power = output[0] * cylinders
            air_flow = output[7] * cylinders
            p_tdc = output[3]


        else:
            flags = ["sweep"]
            input["far_goal"] = x[0]  # fuel air ratio
            (
                _,
                _,
                _,
                air_flow,
                _,
                _,
                _,
                _,
                indicated_power,
                _,
                _,
                _,
                p_tdc,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = run_piston_engine(input, flags)

        # pressurise circumventing flow
        m_circumvent = core_flow - air_flow
        if m_circumvent < 0:
            return 1e6

        # dont need compress negative pressure ratio
        if p_ratio <= 1:
            p_ratio_circ = 1
            # pressure_circ, T_circumv, P_circumv = \
            #    components.compressor(Tin, pin / 0.99, m_circumvent, 0.85, p_ratio_circ * 0.99 * 0.99)
            pressure_circ, T_circumv, P_circumv = 0.0, Tin, 0.0
        else:
            pressure_circ, T_circumv, P_circumv = components.compressor(
                Tin, pin / 0.99, m_circumvent, 0.85, p_ratio * (1 - p_loss_out)
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
        bsr = input["bsr"]
        stroke = bore / bsr
        lv_max = bore * 0.1
        # cylinders = data[31]
        # v_mean = data[10]
        rpm = v_mean / (2 * stroke) * 60
        rps = rpm / 60
        Vd_tot = stroke * bore**2 / 4 * np.pi * cylinders
        cycle = input["cycle"]

        # auxiliary losses and friction losses. these do not depend on the trapped fuel air ratio
        friction_loss, aux_loss, _ = post_processing.friction_patton(
            bore, rpm, stroke, v_mean, pin, cr, cylinders, lv_max, cycle
        )


        shaft_power = indicated_power - aux_loss - friction_loss - P_circumv - P_fuel_pump

        residual = np.array([shaft_power - power_req])
        if not surrogate_status:
            print(
                f"(Not Neural Network) Residual between power: {residual[0]:.16f}, fuel air ratio: {far34:.16f}"
            )

        return residual

    try:
        start = timer()
        x0 = np.array([far_s / 2.0])  # p_ratio, far
        x_match, infodict, ier, mesg = fsolve(find_match, x0, full_output=True)
        # print(infodict)
        # print(ier)
        # print(mesg)
        if ier == 1:
            sol = infodict["fvec"]
            # print(f'Fsolve worked. {sol}, {x_match[0]}')
        else:
            # THIS IS IF POWER IS NOT MATCHED
            # print('fsolve failed in match_power_NN')
            error = True
            listofzeros = [0] * outputs
            listofzeros[-1] = error
            return listofzeros

        end = timer()
        # print(f'find matching power: {end - start}, {x_match}')
    except ValueError as e:
        print("problem with matching power")
        print(e)
        listofzeros = [0] * outputs
        listofzeros[-1] = error
        return listofzeros

    # This last simulation run could probably be done without, just taking the values from find_match
    # input to surrogate
    far34_final = x_match[0]  # far that matched the power
    p_ratio_final = p_ratio

    if surrogate_status:
        # get the output of the surrogate

        piston_input_final = np.atleast_2d(
            np.array([pin, Tin, p_ratio_final, cr, bore, v_mean, T_fuel, far34_final])
        )

        # T34_final, air_flow_final, p_max_final, T_max_final, induced_power_final, heat_loss_final, p_tdc_final\
        #    = nn_output(piston_input_final, meta_model)

        # use meta model to get outputs from the piston egnine
        output_final = meta_model.inference(piston_input_final)[0]

        # DENNA SKALL RÄKNAS UT ISTÄLLET!!!!

        indicated_power_final = output_final[0] * cylinders
        heat_loss_final = output_final[1] * cylinders
        nox_ppm = output_final[2]
        p_tdc_final = output_final[3]
        p_max_final = output_final[4]
        T_max_final = output_final[5]
        T34_final = output_final[6]
        air_flow_final = output_final[7] * cylinders



    else:
        flags = ["sweep"]
        input["far_goal"] = far34_final  # fuel air ratio
        (
            T34_final,
            _,
            _,
            air_flow_final,
            p_max_final,
            T_max_final,
            _,
            _,
            indicated_power_final,
            _,
            _,
            heat_loss_final,
            p_tdc_final,
            _,
            nox_ppm,
            _,
            EI_nox,
            _,
            nox_spec,
        ) = run_piston_engine(input, flags)

    # power needed to compress circumventing air
    m_circumvent_final = core_flow - air_flow_final
    if m_circumvent_final < 0:
        error = True
        # print(f'Flow through piston engine is smaller than core flow with {m_circumvent_final} [kg/s].')
        listofzeros = [0] * outputs
        listofzeros[-1] = error
        return listofzeros

    # negative pressure ratio dont need to compress
    if p_ratio_final <= 1:
        p_ratio_circ_final = 1
        # p_circ_final, T_circ_final, P_circumv_final = \
        #    components.compressor(Tin, pin / 0.99, m_circumvent_final, 0.85, p_ratio_circ_final * 0.99 * 0.99)
        p_circ_final, T_circ_final, P_circumv_final = 0, Tin, 0
    else:
        p_circ_final, T_circ_final, P_circumv_final = components.compressor(
            Tin, pin / 0.99, m_circumvent_final, 0.85, p_ratio_final * (1 - p_loss_out)
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

    p34 = pin * p_ratio_final * (1-p_loss_out)
    p35 = p34

    # power needed to pressurise the fuel
    fuel_flow_final = air_flow_final * far34_final
    P_fuel_pump_final = components.fuel_pump(p_tdc_final, fuel_type, fuel_flow_final)

    # things needed for aux and friction losses
    bsr = input["bsr"]
    stroke = bore / bsr
    lv_max = bore * 0.1
    # cylinders = data[31]
    # v_mean = data[10]
    rpm = v_mean / (2 * stroke) * 60
    rps = rpm / 60
    Vd_tot = stroke * bore**2 / 4 * np.pi * cylinders
    cycle = input["cycle"]
    if cycle == "4T":
        n_r = 2
    else:
        n_r = 1

    # auxiliary losses and friction losses. these do not depend on the trapped fuel air ratio
    friction_loss_final, aux_loss_final, _ = post_processing.friction_patton(
        bore, rpm, stroke, v_mean, pin, cr, cylinders, lv_max, cycle
    )


    # power out from engine after fuel pump and aux losses and friction and pressurising circumventing flow
    shaft_power_final = (
        indicated_power_final
        - aux_loss_final
        - friction_loss_final
        - P_circumv_final
        - P_fuel_pump_final
    )

    # print(T35_final)
    # print(p_ratio_final)
    # print(f'Power matching N: {shaft_power_final - power_req}')

    # calculate mass of NOx (ppm is based on m34) (and convert to fraction from ppm)
    m_nox = nox_ppm * m34 * 1e-6

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
        indicated_power_final,
        friction_loss_final,
        aux_loss_final,
        heat_loss_final,
        P_fuel_pump_final,
        bore,
        P_circumv_final,
        m_nox,
        error,
    )
