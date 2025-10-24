import numpy as np
import importlib
from timeit import default_timer as timer

from CCE.src import components, compressible, misc
from CCE.src.gas_props.air_properties import isa

from thermo import fuel_props


def run_turbofan(input, flags):
    Fn = input['Fn']
    dTisa = input['dTisa']
    bpr = input['bpr']
    T4_req = input['T4']
    fpr_outer = input['fpr_outer']
    Fs_req = input['Fs_req']
    dp_intake = input['dp_intake']
    dp_bypass = input['dp_bypass']
    Mach = input['M']
    eta_fan = input['eta_fan']
    eta_p_ipc = input['eta_p_lpc']
    eta_p_hpc = input['eta_p_hpc']
    eta_b = input['eta_b']
    dPcomb = input['dPcomb']
    eta_s = input['eta_s']
    eta_g = input['eta_g']
    eta_hpt = input['eta_hpt']
    eta_lpt = input['eta_lpt']
    cfg_core = input['cfg_core']
    cfg_bypass = input['cfg_bypass']
    cd_nozzle = input['cd_nozzle']
    alt = input['alt']
    fuel_type = input['fuel']
    OPR = input['OPR']
    PR = input['PR']
    t_fuel = input['t_fuel']

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
        alt, dTisa, False
    )  # static pressure, static temperature and speed of sound

    # calculate optimum fpr from Gouya paper (Unsure if this works for me)
    # fpr_outer = misc.fpr_opt(Mach, bpr, Fs_req, Ta, eta_s_lpt, eta_outer_fan, cfg_bypass, cd_nozzle, dp_bypass)

    # Total properties
    p0, T0 = compressible.stagnation(pa, Ta, Mach)

    # Inlet
    p2, T2 = components.inlet(p0, T0, dp_intake)
    m2 = m0

    # Split core and bypass mass flow

    # Bypass stream
    m12 = m2 * bpr / (1 + bpr)

    # Outer fan
    p12, T12, P_outer_fan = components.compressor_isentropic(T2, p2, m12, eta_fan, fpr_outer)

    # Core stream
    m22 = m2 / (1 + bpr)

    # Inner fan
    p22, T22, P_inner_fan = components.compressor_isentropic(T2, p2, m22, eta_fan, fpr_inner)

    # IPC
    p24, T24, P_lpc = components.compressor(T22, p22, m22, eta_p_ipc, pi_ipc)
    m24 = m22

    # Inter compressor loss
    p25 = 0.9885 * p24
    T25 = T24
    m25 = m24

    # HPC
    p3, T3, P_hpc = components.compressor(T25, p25, m25, eta_p_hpc, pi_hpc)
    m3 = m25

    T31 = T3
    p31 = p3

    input_burner_turbine = {
        "m31": m3,
        "m32": 0.0,
        "m34": 0.0,
        "T_cooling": T3,
        "T34": T3,
        "T4_req": T4_req,
        "far34": 0.0,
        "fuel_type": fuel_type,
        "T_fuel": t_fuel,
        "dP_comb": dPcomb,
        "eta_b": eta_b,
        "p34": p3,
        "power_req": P_hpc / eta_s,
        "eta_s_lpt": eta_hpt,
        "second burner": True,
    }

    output_burner_turbine = components.burner_turbine(input_burner_turbine)

    # if error in burner_turbine
    if output_burner_turbine["error"]:
        error = True
        # print("problem with piston engine matching")
        output_dict = {
            "sfc": 999,
            "error": error,
            "error_type": "PISTON"

        }
        return output_dict

    # number 6 is just after 1% pressure loss after turbine duct
    m31 = output_burner_turbine["m35"]
    m4 = output_burner_turbine["m4"]
    m41 = output_burner_turbine["m46"]
    m45 = output_burner_turbine["m5"]
    p4 = output_burner_turbine["p4"]
    p41 = output_burner_turbine["p5"]
    p45 = output_burner_turbine["p6"]
    T4 = output_burner_turbine["T4"]
    # T46 is after ngv cooling
    T41 = output_burner_turbine["T46"]
    # T47 is after power extraction
    T42 = output_burner_turbine["T47"]
    # T5 is after rotor cooling
    T45 = output_burner_turbine["T5"]
    equ4 = output_burner_turbine["equ4"]
    equ41 = output_burner_turbine["equ46"]
    equ45 = output_burner_turbine["equ5"]
    fuel_flow_burner = output_burner_turbine["fuel_flow_burner"]
    m_cool = output_burner_turbine["m_cool"]
    m_cool_ngv = output_burner_turbine["m_ngv"]
    m_cool_rotor = output_burner_turbine["m_rotor"]
    q_ngv = output_burner_turbine["q_ngv"]


    print(f"Cooling ratio: {m_cool / m3}")

    far_burner = equ4 * far_s


    # Low pressure turbine, powering fan and LPC
    # IPC + inner and outer fan (with gearbox efficiency) + everything shaft efficiency
    power_lpt = (P_lpc + (P_inner_fan + P_outer_fan) / eta_g) / eta_s
    p5, T5, m5, equ5, error = components.turbine(
        T45,
        p45,
        m45,
        equ45,
        power_lpt,
        eta_lpt,
        fuel_type,
        cooling=False,
    )


    # Hot nozzle
    m6 = m5
    p6 = p5
    T6 = T5

    # Hot nozzle (no recuperation)
    equ6 = equ5
    F8, v8_id, v8, error = components.nozzle(
        p6, T6, pa, equ6, m6, cfg_core, cd_nozzle, fuel_type
    )

    # Bypass stream

    # Cold nozzle
    F18, v18_id, v18, error = components.nozzle(
        p12, T12, pa, equ=0, m=m12, cfg=cfg_bypass, cd=cd_nozzle, fuel_type=fuel_type
    )

    # Thrust
    v_0 = a * Mach  # air speed

    # Net thrust
    F = F8 + F18 - v_0 * m2

    # Ideal jet velocity ratio NOT VALID ANYMORE
    vel_ratio = v18_id / v8_id

    # Calculating the work potential left after powering LPC and inner fan
    p_wp, T_wp, m_wp, equ_wp, error = components.turbine(
        T45,
        p45,
        m45,
        equ45,
        (P_lpc + P_inner_fan / eta_g) / eta_s,
        eta_lpt,
        fuel_type,
        cooling=False,
    )

    # Efficiencies
    sfc, eta_core, eta_transmission, eta_th, eta_p, eta_o, Fs = misc.calc_efficiencies_jetA_geared(
        F,
        fuel_flow_burner,
        m12,
        v18_id,
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

        p_array = (
                np.array(
                    [
                        pa,
                        p0,
                        p2,
                        p22,
                        p25,
                        p3,
                        p31,
                        p4,
                        p4,
                        p45,
                        p45,
                        p5,
                        p6,
                        p12,
                    ]
                )
                * 1e-5
        )
        T_array = (
            np.array(
                [
                    Ta,
                    T0,
                    T2,
                    T22,
                    T25,
                    T3,
                    T31,
                    T4,
                    T41,
                    T42,
                    T45,
                    T5,
                    T6,
                    T12,
                ]
            )
        )
        m_array = (
            np.array(
                [
                    m0,
                    m0,
                    m2,
                    m22,
                    m25,
                    m3,
                    m31,
                    m4,
                    m41,
                    m41,
                    m45,
                    m5,
                    m6,
                    m12,
                ]
            )
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
                    far_burner,
                    far_s * equ41,
                    far_s * equ41,
                    far_s * equ45,
                    far_s * equ5,
                    far_s * equ6,
                    0.0,
                ]
            )
        )

        s_array = misc.entropy_array(p_array, T_array, far_array, fuel_type)

        misc.print_efficiencies(eta_o, eta_p, eta_th, eta_transmission, eta_core, Fs)

        misc.plot_stations_jetA_geared(p_array, T_array)

        misc.csv_output_jetA_geared(p_array, T_array, m_array, far_array, s_array)

    return sfc, vel_ratio, F, m0
