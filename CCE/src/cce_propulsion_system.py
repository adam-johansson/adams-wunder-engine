import numpy as np
import importlib
from timeit import default_timer as timer

from CCE.src import components, thermo, compressible, misc
from CCE.src.gas_props.air_properties import isa


def run_cce(indata, data_piston, flags, meta_model):
    [Fn, disa, bpr, T35_req, fpr_outer, Fs_req, dp_in, dp_bypass, Mach, eta_inner_fan,
     eta_outer_fan, pi_hpc, eta_p_hpc, pi_ipc, eta_p_ipc, eta_b, dPcomb, eta_s, eta_g, q_ngv, bpr_c,
     eta_s_lpt, cfg_core, cfg_bypass, cd_nozzle, alt, fuel_type, pi_pe, surrogate, m0, cr, OPR, PR, bore, second_burner] = indata

    error = False
    minor_error_mass = False
    minor_error_pressure = False

    # calculate mass flow
    m0 = Fn / Fs_req

    # calculate pressure ratios based on OPR and pressure ratio split
    fpr_inner = 0.91 * fpr_outer
    OPR_c = OPR / fpr_inner  # OPR of the compressor, excluding fan
    pi_ipc = OPR_c ** PR
    pi_hpc = OPR_c / pi_ipc

    # fuel props
    far_s, LHV = thermo.fuel_props(fuel_type)

    # ISA table
    pa, Ta, a = isa(alt, disa, False)  # static pressure, static temperature and speed of sound

    # calculate optimum fpr from Gouya paper (Unsure if this works for me)
    #fpr_outer = misc.fpr_opt(Mach, bpr, Fs_req, Ta, eta_s_lpt, eta_outer_fan, cfg_bypass, cd_nozzle, dp_bypass)

    # Total properties
    p0, T0 = compressible.stagnation(pa, Ta, Mach)

    # Inlet
    p2, T2 = components.inlet(p0, T0, dp_in)
    m2 = m0

    # Split core and bypass mass flow
    m21 = m2 / (1 + bpr)  # core flow
    m13 = m2 * bpr / (1 + bpr)  # bypass stream

    # Inner fan
    p21, T21, P21 = components.compressor(T2, p2, m21, eta_inner_fan, fpr_inner)

    # Outer fan
    p13, T13, P13 = components.compressor(T2, p2, m13, eta_outer_fan, fpr_outer)

    # Compressor intake loss
    p22 = 0.99 * p21

    # IPC
    p25, T25, P25 = components.compressor(T21, p22, m21, eta_p_ipc, pi_ipc)
    m25 = m21

    # Inter compressor loss
    p26 = 0.99 * p25

    # HPC
    p3, T3, P3 = components.compressor(T25, p26, m25, eta_p_hpc, pi_hpc)
    m3 = m25

    # Remove cooling flow
    m_cool = m3 * bpr_c
    m31 = m3 - m_cool  # core flow after cooling air is removed
    p31 = p3  # pressure of main flow
    T31 = T3  # temperature of main flow after removal of cooling flow

    if second_burner:
        p4 = p31 * 0.99 * pi_pe * 0.99 * (1 - dPcomb)
    else:
        p4 = p31 * 0.99 * pi_pe * 0.99

    # Compress cooling flow
    pi_cool = p4 / p3
    p_cool, T_cool, P_cooling = components.compressor(T3, p3, m_cool * q_ngv, 0.85, pi_cool)

    # Piston intake duct pressure loss
    p32 = 0.99 * p31
    T32 = T31

    # Piston engine
    data_piston[0] = p32
    data_piston[1] = T32
    data_piston[2] = pi_pe
    data_piston[7] = cr
    data_piston[8] = bore

    # flags: plot, output, validation, sweep
    flags_piston = ["sweep"]

    # Number of V12 engines
    nr_engines = 2

    power_req = P3 / eta_g + P_cooling
    # Piston engine powers the HPC + Cooling+ gearbox for hpc ( no shaft efficiency now) (circumv flow is within
    # the piston model)
    power_req_single = power_req / nr_engines
    # flow_req_single = m32 / nr_engines  # flow in each piston engine

    # matching throttle and diameter
    # T34, power_piston, eta_th, air_flow, p_max, T_max, far_piston, bore_match, equ_trapped, throttle_match, \
    #    induced_power, friction_loss, aux_loss, heat_loss \
    #    = misc.match_piston_engine(data_piston, flags, match_status, power_req_single, flow_req_single)


    # ADD SOME CHECKS HERE TO SEE IF INPUT IS WITHIN VALID RANGE OF SURROGATE
    T34, T35, p34, p35, m34_single, m35_single, far34, far35, power_piston, air_flow, p_max, T_max, far_piston, \
        induced_power, friction_loss, aux_loss, heat_loss, P_fuel_pump, bore_match, error, P_circumv \
        = misc.match_power_nn(data_piston, meta_model, power_req_single, core_flow=m31/nr_engines, surrogate_status=surrogate)

    # if engine was not able to match power requirements or negative air flow, return error
    if error:
        return 999, 0, 0, 0, error, 0, 0, 0, 0, 0, 0

    power_tot = nr_engines * power_piston  # break power on the piston engine shaft
    air_flow_tot = nr_engines * air_flow
    m34 = m34_single * nr_engines
    m35 = m35_single * nr_engines

    m_circumv = m31 - air_flow_tot

    # checking if flow is matching
    if m_circumv < 0:
        minor_error_mass = True
        error = True
        #print('Larger flow in piston engine than core')
        cost = 999 - m_circumv*100
        return cost, 0, 0, 0, error, 0, 0, 0, 0, 0, 0

    # checking max pressure
    if p_max > 250:
        error = True
        minor_error_pressure = True
        #print(f'Maximum pressure reached: {p_max}')
        cost = 999 + (p_max - 250)*100
        return cost, 0, 0, 0, error, 0, 0, 0, 0, 0, 0

    bpr_piston = m_circumv / m3  # fraction of air led around engine (based on m3)
    m32 = m31 - m_circumv

    nr_of_cylinders_per_engine = data_piston[31]
    V_d = bore_match * bore_match ** 2 / 4 * np.pi
    V_d_tot = V_d * nr_of_cylinders_per_engine * nr_engines  # total displacement

    # get total power from both V12 engines
    fuel_flow_piston = air_flow_tot * far_piston
    induced_power_tot = nr_engines * induced_power
    friction_loss_tot = friction_loss * nr_engines
    aux_loss_tot = aux_loss * nr_engines
    heat_loss_tot = heat_loss * nr_engines
    P_fuel_pump_tot = P_fuel_pump * nr_engines

    if second_burner:
        # Second burner
        p4, T4, far_burner = components.burner(p35, T35, far35 / far_s, T35_req, dPcomb, eta_b, fuel_type)
        m4 = m35 * (1 + far_burner)  # flow after burner

        # burner fuel flow
        fuel_flow_burner = far_burner * m35

        # fuel air ratio after burner and piston engine
        far_tot = far_burner + far_piston  # is this correct?
        equ4 = far_tot / far_s

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
    p43, T41, T43, T5, m41, m5, equ5, error = components.turbine(T4, p4, m4, equ4,
                                                          (P25 + (P21 + P13) / eta_g) / eta_s,
                                                          eta_s_lpt, fuel_type, cooling=True, t_cool=T_cool,
                                                          m1_cool=m_cool, q_ngv=q_ngv)

    if error:
        #print('Prob too high power demand on LPT')
        return 999, 0, 0, m0, error, 0, 0, 0, 0, 0, 0
    p41 = p4  # after NGV
    m43 = m41  # after rotor
    p5 = p43  # after stator

    # Turbine exhaust duct pressure loss
    p6 = p5 * 0.99
    T6 = T5
    m6 = m5

    # Hot nozzle
    equ6 = equ5
    F8, v8_id, v8, error = components.nozzle(p6, T6, pa, equ6, m6, cfg_core, cd_nozzle, fuel_type)

    if error:
        #print(f'Prob too low pressure and temperature in hot nozzle. p, T: {p6, T6}')
        cost = 999 + (pa - p6)
        return cost, 0, 0, m0, error, 0, 0, 0, 0, 0, 0

    # Heating the fuel
    # TODO: Make this work for jetA as well
    heating_h2, oil_temp_1 = components.fuel_oil_hx(300e5, 20, 0.08, 450, fuel_flow_piston + fuel_flow_burner)

    # Bypass stream
    # Split stream into cooling air and not cooling air
    bypass_cooling = 0.9
    m14 = bypass_cooling * m13  # mass flow not going through heat exchanger
    p14 = (1 - dp_bypass) * p13
    T14 = T13  # adiabatic bypass

    # Cold nozzle ( no heat exchanger)
    F18, v18_id, v18, error = components.nozzle(p14, T14, pa, equ=0, m=m14, cfg=cfg_bypass, cd=cd_nozzle,
                                                fuel_type=fuel_type)
    if error:
        #print('Prob too low pressure and temperature in cold nozzle')
        cost = 999 + (pa - p14)
        return cost, 0, 0, m0, error, 0, 0, 0, 0, 0, 0

    # Cooling bypass flow
    m15 = (1 - bypass_cooling) * m13  # mass flow through heat exchanger
    cooling_req = heat_loss_tot + friction_loss_tot  # total amount of heat produced by piston engine
    heating_bypass = cooling_req - heating_h2  # amount of heat needed to be cooled to the bypass
    #dp_hx = 5.7 / 100  # pressure loss over hx
    p15, T15 = components.hx_NASA(p13, T13, heating_bypass, m15, oil_temp_1)  # going through air-oil hx

    # Cooling flow nozzle
    F17, v17_id, v17, error = components.nozzle(p15, T15, pa, equ=0, m=m15, cfg=cfg_bypass, cd=cd_nozzle,
                                                fuel_type=fuel_type)

    if error:
        # print('Prob too low pressure and temperature in cooling nozzle')
        cost = 999 + (pa - p15)
        return cost, 0, 0, m0, error, 0, 0, 0, 0, 0, 0

    # Thrust
    v_0 = a * Mach  # air speed
    F = F8 + F18 + F17 - v_0 * m2  # net thrust

    # Total fuel flow
    mdot_fuel = fuel_flow_piston + fuel_flow_burner

    # Ideal jet velocity ratio NOT VALID ANYMORE
    vel_ratio = v18_id / v8_id

    # Power
    power_lpt = (P25 / eta_s) + (P21 + P13) / (eta_s * eta_g)

    # power required by the fan divided by mechanical efficiency and gearbox + ipc only shaft efficiency
    power_hpc = P3 / eta_s  # power required by the hpc divided by mechanical efficiency (from piston engine/HPT)

    # Calculating the work potential left after powering ipc and inner fan
    p_wp, dummy, dummy, T_wp, dummy, m_wp, equ_wp, error = components.turbine(T4, p4, m4, equ4,
                                                          (P25 + P21 / eta_g) / eta_s,
                                                          eta_s_lpt, fuel_type, cooling=True, t_cool=T_cool,
                                                          m1_cool=m_cool, q_ngv=q_ngv)

    # Efficiencies
    sfc, eta_core, eta_transmission, eta_th, eta_p, eta_o, Fs = misc.calc_efficiencies(F, mdot_fuel, m14, v18_id,
                                                                                       m15, v17_id, m6, v8_id, m0,
                                                                                       v_0, p_wp, T_wp,
                                                                                       equ_wp, fuel_type, pa, LHV)

    if 'print_output' in flags:

        # Creating array for output
        p_array = np.array([p2, p13, p14, p15, p21, p25, p3, p31, p32, p34, p35, p4, p41, p43, p5, p6]) * 1e-3
        T_array = (T2, T13, T14, T15, T21, T25, T3, T31, T32, T34, T35, T4, T41, T43, T5, T6)
        m_array = (m2, m13, m14, m15, m21, m25, m3, m31, m32, m34, m35, m4, m41, m43, m5, m6)
        misc.print_output(air_flow_tot, sfc, F, m0, m21, power_tot * 1e-3, power_hpc, power_lpt,
                          p3, T3, p35, T34, T4, p_max, T_max, far34 / far_s, far_s, induced_power_tot,
                          fuel_flow_piston, fuel_flow_burner, V_d_tot, friction_loss_tot, aux_loss_tot, heat_loss_tot,
                          bpr_piston)

        misc.print_efficiencies(eta_o, eta_p, eta_th, eta_transmission, eta_core, Fs)

        misc.plot_stations(p_array, T_array)

        misc.csv_output(p_array, T_array, m_array)
        piston_gearbox = (induced_power_tot - friction_loss_tot - aux_loss_tot) * (1 - eta_g)
        misc.energy_output(induced_power_tot, friction_loss_tot, aux_loss_tot, P_fuel_pump_tot, P_circumv, P_cooling,
                           piston_gearbox, P3, power_lpt, power_lpt * (1 - eta_s), P25,
                           (P21 + P13) * (1 - eta_g) / eta_g, P21 + P13, heat_loss_tot,
                           heating_h2, heating_bypass)

        #print(f'ideal jet velocity ratio: {vel_ratio}')

        # cost function if minor errors NOT USED FOR NOW
        if minor_error_mass:
            punish_factor = 1e-4
            sfc = sfc - punish_factor * m_circumv

        if minor_error_pressure:
            punish_factor = 1e-4
            print(p_max)
            sfc = sfc + punish_factor * (p_max - 300)
    #print(f'FAR: {far_piston}, T_before: {T32}, T_after: {T34}, bore: {bore_match} ')
    return sfc, vel_ratio, F, m0, error, p_max, T_max, T32, T34, T4, far_piston
