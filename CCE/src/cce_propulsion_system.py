import numpy as np
import importlib
from timeit import default_timer as timer

from CCE.src import components, compressible, misc
from CCE.src.gas_props.air_properties import isa

from thermo import fuel_props


def run_cce(indata, data_piston, flags, meta_model):
    [
        Fn,
        disa,
        bpr,
        T4_req,
        fpr_outer,
        Fs_req,
        dp_in,
        dp_bypass,
        Mach,
        eta_fan,
        eta_p_lpc,
        eta_p_hpc,
        eta_b,
        dPcomb,
        eta_s,
        eta_g,
        q_ngv,
        bpr_c,
        eta_s_lpt,
        cfg_core,
        cfg_bypass,
        cd_nozzle,
        alt,
        fuel_type,
        pi_pe,
        surrogate,
        cr,
        OPR,
        PR,
        bore,
        second_burner,
        t_fuel,
        t_tank,
        offtake,
    ] = indata

    error = False
    minor_error_mass = False
    minor_error_pressure = False

    # number of output variables from function
    outputs = 13

    # calculate mass flow
    m0 = Fn / Fs_req

    # calculate pressure ratios based on OPR and pressure ratio split
    fpr_inner = 0.913 * fpr_outer
    OPR_c = OPR / fpr_inner  # OPR of the compressor, excluding fan
    OPR_c = OPR_c / 0.9885 #account for the pressure loss after LPC
    pi_ipc = OPR_c**PR
    pi_hpc = OPR_c / pi_ipc

    # fuel props
    far_s, LHV = fuel_props(fuel_type)

    # ISA table
    pa, Ta, a = isa(
        alt, disa, False
    )  # static pressure, static temperature and speed of sound

    # calculate optimum fpr from Gouya paper (Unsure if this works for me)
    # fpr_outer = misc.fpr_opt(Mach, bpr, Fs_req, Ta, eta_s_lpt, eta_outer_fan, cfg_bypass, cd_nozzle, dp_bypass)

    # Total properties
    p0, T0 = compressible.stagnation(pa, Ta, Mach)

    # Inlet
    p2, T2 = components.inlet(p0, T0, dp_in)
    m2 = m0

    # Split core and bypass mass flow
    m21 = m2 / (1 + bpr)  # core flow
    m13 = m2 * bpr / (1 + bpr)  # bypass stream

    # Convert polytropic efficiency to isentropic for the fan


    # Inner fan
    p21, T21, P_inner_fan = components.compressor_isentropic(T2, p2, m21, eta_fan, fpr_inner)

    # Outer fan
    p13, T13, P_outer_fan = components.compressor_isentropic(T2, p2, m13, eta_fan, fpr_outer)

    # Compressor intake loss
    p22 = 0.99 * p21


    # LPC
    p25, T25, P_lpc = components.compressor(T21, p22, m21, eta_p_lpc, pi_ipc)
    m25 = m21

    # Inter compressor loss
    p26 = 0.9885 * p25


    # HPC
    p3, T3, P_hpc = components.compressor(T25, p26, m25, eta_p_hpc, pi_hpc)
    m3 = m25

    # Remove cooling flow
    m_cool = m3 * bpr_c
    m31 = m3 - m_cool  # core flow after cooling air is removed
    p31 = p3  # pressure of main flow
    T31 = T3  # temperature of main flow after removal of cooling flow

    p_loss_piston_in = 0.03
    p_loss_piston_out = 0.03


    if second_burner:
        p4 = p31 * (1-p_loss_piston_in) * pi_pe * (1-p_loss_piston_out) * (1 - dPcomb)
    else:
        p4 = p31 * (1-p_loss_piston_in) * pi_pe * (1-p_loss_piston_out)


    # Compress cooling flow
    if p4 > p3:
        pi_cool = p4 / p3
        p_cool, T_cool, P_cooling = components.compressor(
            T3, p3, m_cool * q_ngv, 0.85, pi_cool
        )
    else:
        P_cooling = 0.0
        T_cool = T3
        p_cool = p3


    # Piston intake duct pressure loss
    p32 = (1-p_loss_piston_in) * p31
    T32 = T31




    # Piston engine
    data_piston[0] = p32
    data_piston[1] = T32
    data_piston[2] = pi_pe
    data_piston[7] = cr
    data_piston[8] = bore
    lv_max = 0.1 * bore
    data_piston[16] = lv_max
    data_piston[25] = t_fuel

    # Number of V12 engines
    nr_engines = 2

    power_req = P_hpc / eta_g + P_cooling + offtake
    # Piston engine powers the HPC + Cooling+ gearbox for hpc ( no shaft efficiency now) (circumv flow is within
    # the piston model)
    power_req_single = power_req / nr_engines
    # flow_req_single = m32 / nr_engines  # flow in each piston engine

    (
        T34,
        T35,
        p34,
        p35,
        m34_single,
        m35_single,
        far34,
        far35,
        power_piston,
        air_flow,
        p_max,
        T_max,
        far_piston,
        induced_power,
        friction_loss,
        aux_loss,
        heat_loss,
        P_fuel_pump,
        bore_match,
        P_circumv,
        m_nox,
        error,
    ) = misc.match_power_nn(
        data_piston,
        meta_model,
        power_req_single,
        core_flow=m31 / nr_engines,
        surrogate_status=surrogate,
        p_loss_in=p_loss_piston_in,
        p_loss_out=p_loss_piston_out,
    )

    # if engine was not able to match power requirements or negative air flow, return error
    if error:
        cost = 999
        listofzeros = [0] * outputs
        listofzeros[0] = cost
        listofzeros[-1] = error
        return listofzeros

    # get total power from both V12 engines
    # power_tot is power after friction, aux, fuel_pump and circumventing compressor
    power_tot = nr_engines * power_piston  # break power on the piston engine shaft
    air_flow_tot = nr_engines * air_flow
    m34 = m34_single * nr_engines
    m35 = m35_single * nr_engines
    P_circumv_tot = P_circumv * nr_engines
    fuel_flow_piston = air_flow_tot * far_piston
    induced_power_tot = nr_engines * induced_power
    friction_loss_tot = friction_loss * nr_engines
    aux_loss_tot = aux_loss * nr_engines
    heat_loss_tot = heat_loss * nr_engines
    P_fuel_pump_tot = P_fuel_pump * nr_engines
    m_nox_tot = m_nox * nr_engines

    m_circumv = m31 - air_flow_tot



    # checking if flow is matching
    if m_circumv < 0:
        minor_error_mass = True
        error = True
        # print('Larger flow in piston engine than core')
        cost = 999 - m_circumv * 100

        listofzeros = [0] * outputs
        listofzeros[-1] = error
        listofzeros[0] = cost
        return listofzeros

    # checking max pressure
    if p_max > 250 * 1e5:
        error = True
        minor_error_pressure = True
        #print(f'Maximum pressure reached: {p_max*1e-5} bar')
        cost = 999 + (p_max - 250 * 1e5) * 100
        #return cost, 0, 0, 0, error, 0, 0, 0, 0, 0, 0, 0
        listofzeros = [0] * outputs
        listofzeros[-1] = error
        listofzeros[0] = cost
        return listofzeros

    bpr_piston = (
        m_circumv / m31
    )  # fraction of air led around engine (based on m31, after cooling flow is removed)
    # mass flow going into the piston engines
    m32 = m31 - m_circumv

    nr_of_cylinders_per_engine = data_piston[31]
    V_d = bore_match * bore_match**2 / 4 * np.pi
    V_d_tot = V_d * nr_of_cylinders_per_engine * nr_engines  # total displacement


    equ35 = far35 / far_s
    if second_burner:
        # Second burner
        p4, T4, far_4 = components.burner(
            p35, T35, equ35, T4_req, dPcomb, eta_b, fuel_type, t_fuel=t_fuel
        )

        # fuel air ratio of added fuel
        far_burner = far_4 - far35

        # m31 is pure air. After cooling flow removed but before piston.
        m4 = (
            m31 + fuel_flow_piston + far_burner * m31
        )  # flow after burner. air + fuel piston + fuel burner

        # burner fuel flow
        fuel_flow_burner = far_burner * m31

        # fuel air ratio after burner and piston engine
        equ4 = far_4 / far_s

    else:
        # skipping second burner
        p4 = p35
        T4 = T35
        m4 = m35
        fuel_flow_burner = 0
        far_burner = 0
        far4 = far35

        equ4 = far4 / far_s




    # Low pressure turbine, powering fan and IPC
    # IPC + inner and outer fan (with gearbox efficiency) + everything shaft efficiency
    power_lpt = (P_lpc + (P_inner_fan + P_outer_fan) / eta_g) / eta_s
    p5, T46, T47, T5, m46, m5, equ46, equ5, error = components.turbine(
        T4,
        p4,
        m4,
        equ4,
        power_lpt,
        eta_s_lpt,
        fuel_type,
        cooling=True,
        t_cool=T_cool,
        m1_cool=m_cool,
        q_ngv=q_ngv,
    )

    if error:
        # print('Prob too high power demand on LPT')
        cost = 999
        listofzeros = [0] * outputs
        listofzeros[-1] = error
        listofzeros[0] = cost
        return listofzeros

# Turbine exhaust duct pressure loss
    p6 = p5 * 0.99
    T6 = T5
    m6 = m5


    # Hot nozzle
    equ6 = equ5
    F8, v8_id, v8, error = components.nozzle(
        p6, T6, pa, equ6, m6, cfg_core, cd_nozzle, fuel_type
    )

    if error:
        #print(f'Prob too low pressure and temperature in hot nozzle. p, T: {p6, T6}')
        cost = 999 + (pa - p6)
        listofzeros = [0] * outputs
        listofzeros[-1] = error
        listofzeros[0] = cost
        return listofzeros

    # Heating the fuel
    if fuel_type == "H2":
        heating_fuel, oil_temp_1 = components.fuel_oil_hx(
            300e5, t_tank, 0.08, t_fuel, fuel_flow_piston + fuel_flow_burner
        )
    else:
        heating_fuel = 0.0
        oil_temp_1 = 400


    # Bypass stream
    # Split stream into cooling air and not cooling air
    # total amount of heat produced by piston engine
    cooling_req = (
        heat_loss_tot + friction_loss_tot
    )

    # total amount of heat needed to be rejected (some heat used to heat fuel)
    heating_bypass = (
        cooling_req - heating_fuel
    )


    # Oil to air heat exchanger in the bypass
    p15, T15, m15, m_oil = components.hx_NASA(
        p13, T13, heating_bypass, oil_temp_1
    )

    print(f"Mass flow of engine oil: {m_oil} kg/s")

    m14 = m13 - m15  # mass flow not going through heat exchanger
    p14 = (1 - dp_bypass) * p13
    T14 = T13  # adiabatic bypass

    # Cold nozzle ( no heat exchanger)
    F18, v18_id, v18, error = components.nozzle(
        p14, T14, pa, equ=0, m=m14, cfg=cfg_bypass, cd=cd_nozzle, fuel_type=fuel_type
    )
    if error:
        #print('Prob too low pressure and temperature in cold nozzle')
        cost = 999 + (pa - p14)
        listofzeros = [0] * outputs
        listofzeros[-1] = error
        listofzeros[0] = cost
        return listofzeros

    # Cooling flow nozzle
    F17, v17_id, v17, error = components.nozzle(
        p15, T15, pa, equ=0, m=m15, cfg=cfg_bypass, cd=cd_nozzle, fuel_type=fuel_type
    )

    if error:
        #print('Prob too low pressure and temperature in cooling nozzle')
        cost = 999 + (pa - p15)
        listofzeros = [0] * outputs
        listofzeros[-1] = error
        listofzeros[0] = cost
        return listofzeros

    # Thrust
    v_0 = a * Mach  # air speed
    F = F8 + F18 + F17 - v_0 * m2  # net thrust

    # Total fuel flow
    mdot_fuel = fuel_flow_piston + fuel_flow_burner

    # NOx emission index
    EI_nox = m_nox_tot / (mdot_fuel) * 1e3

    # Ideal jet velocity ratio NOT VALID ANYMORE
    vel_ratio = v18_id / v8_id

    power_hpc = (
        P_hpc / eta_g
    )  # power required by the hpc divided by mechanical efficiency (from piston engine/HPT)

    # Calculating the work potential left after powering lpc and inner fan
    p_wp, _, _, T_wp, _, m_wp, _, equ_wp, error = components.turbine(
        T4,
        p4,
        m4,
        equ4,
        (P_lpc + P_inner_fan / eta_g) / eta_s,
        eta_s_lpt,
        fuel_type,
        cooling=True,
        t_cool=T_cool,
        m1_cool=m_cool,
        q_ngv=q_ngv,
    )

    # Efficiencies
    sfc, eta_core, eta_transmission, eta_th, eta_p, eta_o, Fs = misc.calc_efficiencies_cce(
        F,
        mdot_fuel,
        m14,
        v18_id,
        m15,
        v17_id,
        m6,
        v8_id,
        m0,
        v_0,
        p_wp,
        T_wp,
        equ_wp,
        fuel_type,
        pa,
        LHV,
    )

    if "print_output" in flags:

        # Creating array for output
        p_array = (
            np.array(
                [
                    p2,
                    p13,
                    p14,
                    p15,
                    p21,
                    p25,
                    p3,
                    p31,
                    p32,
                    p34,
                    p35,
                    p4,
                    p4,
                    p5,
                    p5,
                    p6,
                ]
            )
            * 1e-3
        )
        T_array = (
            T2,
            T13,
            T14,
            T15,
            T21,
            T25,
            T3,
            T31,
            T32,
            T34,
            T35,
            T4,
            T4,
            T46,
            T5,
            T6,
        )
        m_array = (
            m2,
            m13,
            m14,
            m15,
            m21,
            m25,
            m3,
            m31,
            m32,
            m34,
            m35,
            m4,
            m4,
            m46,
            m5,
            m6,
        )

        far_array = (
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    far34,
                    far35,
                    equ4 * far_s,
                    equ46 * far_s,
                    equ46 * far_s,
                    equ5 * far_s,
                    equ5 * far_s,
                    equ6 * far_s,
                ]
            )
        )

        s_array = misc.entropy_array(p_array, T_array, far_array, fuel_type)

        misc.print_output(
            air_flow_tot,
            sfc,
            F,
            m0,
            m21,
            power_tot,
            power_hpc,
            offtake,
            power_lpt,
            p3,
            T3,
            p35,
            T34,
            T4,
            p_max,
            T_max,
            far34 / far_s,
            far_s,
            induced_power_tot,
            fuel_flow_piston,
            fuel_flow_burner,
            V_d_tot,
            friction_loss_tot,
            aux_loss_tot,
            heat_loss_tot,
            bpr_piston,
            m_nox_tot,
        )

        misc.print_efficiencies(eta_o, eta_p, eta_th, eta_transmission, eta_core, Fs)

        misc.plot_stations_cce(p_array, T_array)

        misc.csv_output_cce(p_array, T_array, m_array, far_array, s_array)
        # power lost in the compressor gearbox
        # assuming offtake is taken before gearbox
        piston_gearbox = (
            induced_power_tot
            - P_fuel_pump_tot
            - P_circumv_tot
            - friction_loss_tot
            - aux_loss_tot
            - P_cooling
            - offtake
        ) * (1 - eta_g)
        misc.power_balance(
            induced_power_tot,
            friction_loss_tot,
            aux_loss_tot,
            P_fuel_pump_tot,
            P_circumv_tot,
            P_cooling,
            piston_gearbox,
            offtake,
            P_hpc,
            power_lpt,
            power_lpt * (1 - eta_s),
            P_lpc,
            (P_inner_fan + P_outer_fan) * (1 - eta_g) / eta_g,
            P_inner_fan + P_outer_fan,
            heat_loss_tot,
            heating_fuel,
            heating_bypass,
        )

        misc.energy_flow_fuel(
            LHV,
            fuel_flow_piston,
            fuel_flow_burner,
            t_tank,
            fuel_type,
            t_fuel,
            induced_power_tot,
            heat_loss_tot,
            T32,
            T34,
            p32,
            p34,
            m32,
            m34,
            far34 / far_s,
            T35,
            equ35,
            m35,
            T4,
            equ4,
            m4,
            pa,
        )

        # print(f'ideal jet velocity ratio: {vel_ratio}')

        # cost function if minor errors NOT USED FOR NOW
        if minor_error_mass:
            punish_factor = 1e-4
            sfc = sfc - punish_factor * m_circumv

        if minor_error_pressure:
            punish_factor = 1e-4
            print(p_max)
            sfc = sfc + punish_factor * (p_max - 300)
    # print(f'FAR: {far_piston}, T_before: {T32}, T_after: {T34}, bore: {bore_match} ')
    return sfc, vel_ratio, F, m0, p_max, T_max, T32, T34, T4, far_piston, T35, EI_nox, error
