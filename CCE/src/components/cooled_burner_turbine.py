from thermo import fuel_props
from CCE.src import components
import numpy as np


def burner_turbine(input_dictionary):

    # m31 is core flow before piston engine
    # m32 is mass flow INTO the piston engine
    # m34 is mass flow OUT of the piston engine

    m31 = input_dictionary["m31"]
    m32 = input_dictionary["m32"]
    m34 = input_dictionary["m34"]
    T_cooling = input_dictionary["T_cooling"]
    T34 = input_dictionary["T34"]
    T4_req = input_dictionary["T4_req"]
    far34 = input_dictionary["far34"]
    fuel_type = input_dictionary["fuel_type"]
    T_fuel = input_dictionary["T_fuel"]
    dP_comb = input_dictionary["dP_comb"]
    eta_b = input_dictionary["eta_b"]
    p34 = input_dictionary["p34"]
    power_req = input_dictionary["power_req"]
    eta_s_lpt = input_dictionary["eta_s_lpt"]
    second_burner = input_dictionary["second burner"]

    # stoichiometric fuel air ratio
    far_s, _ = fuel_props(fuel_type)

    # no error as default
    error = False


    # material limits
    T_ngv = 1350
    T_rotor = 1250

    #flow bypassing the piston engine
    m_bypass = m31 - m32
    T_bypass = T_cooling

    # equivalence ratio at piston engine exhaust
    equ34 = far34 / far_s

    # if T4 is lower than rotor limit, no cooling is needed
    if T4_req < T_rotor:
        cooling = False
        m_ngv = 0.0
        m_rotor = 0.0
        q_ngv = 0.0
        m_cooling = 0.0

        # mass flow before piston engine
        # m31

        # mass flow into piston engine
        # m32

        # mass flow out of piston engine
        # m34

        # mass flow bypassing the engine
        # m_bypass

        # mix all bypass flow directly after piston engine
        T35, equ35 = components.mix(
            m34,
            T34,
            equ34,
            m_bypass,
            T_bypass,
            equ2=0,
            fuel_type=fuel_type,
        )

        far35 = equ35 * far_s
        p35 = p34

        m35 = m34 + m_bypass

        # second burner
        if second_burner:

            # when using a second burner, T4 must be higher than T35
            if T35 > T4_req:
                error = True
                # print('Prob too high power demand on LPT')
                output_dict = {
                    "sfc": 999,
                    "error": error,
                    "error_type": "second burner temp"

                }
                return output_dict


            # Second burner
            p4, T4, far_4 = components.burner(
                p35, T35, equ35, T4_req, dP_comb, eta_b, fuel_type, t_fuel=T_fuel
            )

            # pure air going into second burner
            m_air_burner = (m32 + m_bypass)

            # fuel air ratio of added fuel
            m_fuel_burner = (far_4 - far35) * m_air_burner

            # mass flow after burner
            m4 = m35 + m_fuel_burner

            # fuel air ratio after burner and piston engine
            equ4 = far_4 / far_s

        else:
            # skipping second burner
            p4 = p35
            T4 = T35
            m4 = m35
            m_fuel_burner = 0
            far4 = far35
            equ4 = far4 / far_s

        # Low pressure turbine, powering fan and IPC
        p5, T5, m5, equ5, error = components.turbine(
            T4,
            p4,
            m4,
            equ4,
            power_req,
            eta_s_lpt,
            fuel_type,
            cooling=cooling,
        )

        T46 = T4
        T47 = T5
        m46 = m4
        equ46 = equ5

        if error:
            # print('Prob too high power demand on LPT')
            output_dict = {
                "sfc": 999,
                "error": error,
                "error_type": "LPT"

            }
            return output_dict


    else:
        # THIS IS FOR TURBINE COOLING
        # we will only reach these high temperatures with second combustor

        # that means: cooling will only happen when T4 = T4_req

        #c_cool = 0.052
        c_cool = 0.128
        # T4 higher than rotor material limit

        # starting guess for cooling mass flow
        m_cooling = 1.0
        lim = 1e-6
        while True:
            #print(m_cooling)

            # mass flow before piston engine
            # m31

            # mass flow into piston engine
            # m32

            # mass flow out of piston engine
            # m34

            # mass flow bypassing the engine
            # m_bypass

            # mass flow mixed into the hot air before burner (bypass - cooling)
            m_mix_burner = (m_bypass - m_cooling)

            # mix bypass air (- cooling air) with hot piston exhaust
            T35, equ35 = components.mix(
                m34,
                T34,
                equ34,
                m_mix_burner,
                T_cooling,
                equ2=0,
                fuel_type=fuel_type,
            )

            # mass flow into the second burner
            m35 = m34 + m_mix_burner

            # pure air going into second burner
            m_air_burner = (m32 + m_mix_burner)

            p35 = p34
            far35 = equ35 * far_s

            # Second burner
            p4, T4, far4 = components.burner(
                p35, T35, equ35, T4_req, dP_comb, eta_b, fuel_type, t_fuel=T_fuel
            )

            # fuel flow of the second burner
            m_fuel_burner = (far4 - far35) * m_air_burner
            equ4 = far4 / far_s

            # mass flow out from the second burner and into the ngv
            m4 = m35 + m_fuel_burner

            if T4 < T_ngv:
                # ngv does not need cooling
                m_ngv = 0.0

                m41 = m4 + m_ngv

                T41 = T4



            else:
                # ngv needs cooling
                eta_cool_ngv = (T4 - T_ngv) / (T4 - T_cooling)

                # fraction of ngv cooling air to main flow
                cool_frac_ngv = c_cool * (eta_cool_ngv / (1 - eta_cool_ngv))

                m_ngv = cool_frac_ngv * m4

                # mix main flow with ngv cooling flow
                T41, equ41 = components.mix(
                    m4,
                    T4,
                    equ4,
                    m_ngv,
                    T_cooling,
                    equ2=0,
                    fuel_type=fuel_type,
                )

                m41 = m4 + m_ngv


            # cooling the rotor

            # ngv needs cooling
            eta_cool_rotor = (T41 - T_rotor) / (T41 - T_cooling)

            # fraction of ngv cooling air to main flow
            cool_frac_rotor = c_cool * (eta_cool_rotor / (1 - eta_cool_rotor))

            m_rotor = cool_frac_rotor * m41

            # we dont need to calculate the temperature after cooling since we do that in turbine power calculation

            # total cooling flow
            m_cooling_new = m_ngv + m_rotor

            # check for convergence
            residual = np.abs(m_cooling_new - m_cooling)/np.abs(m_cooling)
            #print(residual)

            if residual < lim:
                break

            # use new cooling mass flow for next iteration
            m_cooling = m_cooling_new

        m_cooling = m_cooling_new
        # fraction of cooling air used for ngv cooling
        q_ngv = m_ngv / m_cooling

        # here calculate the turbine..
        # Low pressure turbine, powering fan and IPC
        p5, T46, T47, T5, m46, m5, equ46, equ5, error = components.turbine(
            T4,
            p4,
            m4,
            equ4,
            power_req,
            eta_s_lpt,
            fuel_type,
            cooling=True,
            t_cool=T_cooling,
            m1_cool=m_cooling,
            q_ngv=q_ngv,
        )


        if error:
            #print('Prob too high power demand on LPT')
            output_dict = {
                "sfc": 999,
                "error": error,
                "error_type": "LPT"

            }
            return output_dict



    # Turbine exhaust duct pressure loss
    p6 = p5 * 0.99

    output_dictionary = {
        "fuel_flow_burner": m_fuel_burner,
        "equ35": equ35,
        "equ4": equ4,
        "equ46": equ46,
        "equ5": equ5,
        "T35": T35,
        "T4": T4,
        "T46": T46,
        "T47": T47,
        "T5": T5,
        "p4": p4,
        "p5": p5,
        "p6": p6,
        "m35": m35,
        "m4": m4,
        "m46": m46,
        "m5": m5,
        "m_ngv": m_ngv,
        "m_rotor": m_rotor,
        "m_cool": m_cooling,
        "q_ngv": q_ngv,
        "error": error,

    }

    return output_dictionary










