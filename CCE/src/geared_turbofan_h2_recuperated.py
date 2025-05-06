import numpy as np
import importlib
from timeit import default_timer as timer

from CCE.src import components, compressible, misc
from CCE.src.gas_props.air_properties import isa

from thermo import fuel_props


def run_turbofan(indata, flags):
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
        eta_p_hpc,
        eta_p_ipc,
        eta_b,
        dPcomb,
        eta_s,
        eta_g,
        q_ngv,
        bpr_c,
        eta_hpt,
        eta_lpt,
        cfg_core,
        cfg_bypass,
        cd_nozzle,
        alt,
        fuel_type,
        OPR,
        PR,
        t_fuel,
        t_tank,
        offtake,
        dp_rec,
        dT_rec,
    ] = indata

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

    # Bypass stream
    m12 = m2 * bpr / (1 + bpr)

    # Outer fan
    p12, T12, P_outer_fan = components.compressor_isentropic(T2, p2, m12, eta_fan, fpr_outer)

    # Core stream
    m22 = m2 / (1 + bpr)

    # Inner fan
    p22, T22, P_inner_fan = components.compressor_isentropic(T2, p2, m22, eta_fan, fpr_inner)

    # LPC
    p24, T24, P_lpc = components.compressor(T22, p22, m22, eta_p_ipc, pi_ipc)
    m24 = m22

    # Inter compressor loss
    p25 = 0.9885 * p24
    T25 = T24
    m25 = m24

    # HPC
    p3, T3, P_hpc = components.compressor(T25, p25, m25, eta_p_hpc, pi_hpc)
    m3 = m25

    # Remove cooling flow
    m_cool = m3 * bpr_c
    m31 = m3 - m_cool  # core flow after cooling air is removed
    p31 = p3  # pressure of main flow
    T31 = T3  # temperature of main flow after removal of cooling flow

    T_cool = T3
    p_cool = p3

    # Burner
    p4, T4, far_burner = components.burner(
        p31, T31, 0.0, T4_req, dPcomb, eta_b, fuel_type, t_fuel=t_fuel
    )

    # fuel added to the mass flow
    fuel_flow_burner = far_burner * m31
    m4 = m31 + fuel_flow_burner

    # fuel air equivalence ratio
    equ4 = far_burner / far_s

    # HPT
    power_hpt = P_hpc / eta_s + offtake
    p45, T41, T42, T45, m41, m45, equ41, equ45, error = components.turbine(
        T4,
        p4,
        m4,
        equ4,
        power_hpt,
        eta_hpt,
        fuel_type,
        cooling=True,
        t_cool=T_cool,
        m1_cool=m_cool,
        q_ngv=q_ngv,
    )


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

    # Flow is split into recuperator and straight to nozzle
    # Hot nozzle (no recuperation)
    recuperation_split = 0.5
    m6 = recuperation_split * m5
    p6 = p5
    T6 = T5

    # Hot nozzle (no recuperation)
    equ6 = equ5
    F8, v8_id, v8, error = components.nozzle(
        p6, T6, pa, equ6, m6, cfg_core, cd_nozzle, fuel_type
    )

    # fuel heat recuperator
    m7 = (1 - recuperation_split) * m5
    #T7, p7 = components.hx_NASA(
    #        p13, T13, heating_bypass, m15, oil_temp_1)
    T7 = T6 - dT_rec
    p7 = p6 * (1 - dp_rec)
    equ7 = equ6

    # Recuperated nozzle
    F75, v75_id, v75, error = components.nozzle(
        p7, T7, pa, equ7, m7, cfg_core, cd_nozzle, fuel_type
    )

    # Bypass stream

    # Cold nozzle
    F18, v18_id, v18, error = components.nozzle(
        p12, T12, pa, equ=0, m=m12, cfg=cfg_bypass, cd=cd_nozzle, fuel_type=fuel_type
    )

    # Thrust
    v_0 = a * Mach  # air speed

    # Net thrust
    F = F8 + F18 + F75 - v_0 * m2

    # Ideal jet velocity ratio NOT VALID ANYMORE
    vhot_avg = (v8_id + v75_id)/2
    vel_ratio = v18_id / vhot_avg

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
    sfc, eta_core, eta_transmission, eta_th, eta_p, eta_o, Fs = misc.calc_efficiencies_recuperated_h2_geared(
        F,
        fuel_flow_burner,
        m12,
        v18_id,
        m6,
        v8_id,
        m7,
        v75_id,
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
                        p7,
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
                    T7,
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
                    m7,
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
                    far_s * equ7,
                ]
            )
        )

        s_array = misc.entropy_array(p_array, T_array, far_array, fuel_type)

        misc.print_efficiencies(eta_o, eta_p, eta_th, eta_transmission, eta_core, Fs)

        misc.plot_stations_rec_h2_geared(p_array, T_array)

        misc.csv_output_rec_h2_geared(p_array, T_array, m_array, far_array, s_array)

    return sfc, vel_ratio, F, m0
