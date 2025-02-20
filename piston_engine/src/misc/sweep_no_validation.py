import matplotlib.pyplot as plt
import numpy as np
import os

from piston_engine.engine import run_piston_engine


def sweep_no_diesel_greek_validation(d, flags):

    # validate against the Rakolpoulos paper
    # the three different load cases (-20 degrees injection timing)
    fuel_air_ratios = np.array([0.028, 0.0345, 0.041])

    nitrogen_oxides_early = []
    peak_pressures_early = []
    indicated_effs_early = []
    IMEPs_early = []

    for far_goal in fuel_air_ratios:
        data = [d.p_in, d.T_in, d.p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, d.cr, d.d, d.bsr,
                d.v_mean, d.lms, d.Twalls, d.ch,
                d.valve_timings, d.n_valve, d.lv_max, d.cd, d.eta_c, d.mf_tot, d.wa,
                d.wm, d.m_wiebe, d.phi_sc, d.phi_cd, d.T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type, far_goal,
                d.cylinders, d.fuel, d.c1, d.c4, d.c5]

        if far_goal > 0.04:
            flags.append('validate_nox_diesel')

        T4, work_piston, eta_th, air_flow, p_max, T_max, far, equ_trapped, induced_power, friction_loss, aux_loss,\
            heat_loss, p_tdc, outflow, no, imep = run_piston_engine(data, flags)


        nitrogen_oxides_early.append(no)
        IMEPs_early.append(imep * 1e-5)
        peak_pressures_early.append(p_max * 1e-5)
        indicated_effs_early.append(eta_th * 1e2)

    flags.remove('validate_nox_diesel')
    nitrogen_oxides_late = []
    peak_pressures_late = []
    indicated_effs_late = []
    IMEPs_late = []

    # the three different load cases (-15 degrees injection timing)
    fuel_air_ratios = np.array([0.028, 0.0345, 0.041])

    phi_sc = (350.0 / 180) * np.pi  # angle at combustion start

    for far_goal in fuel_air_ratios:
        data = [d.p_in, d.T_in, d.p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, d.cr, d.d, d.bsr,
                d.v_mean, d.lms, d.Twalls, d.ch,
                d.valve_timings, d.n_valve, d.lv_max, d.cd, d.eta_c, d.mf_tot, d.wa,
                d.wm, d.m_wiebe, phi_sc, d.phi_cd, d.T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type, far_goal,
                d.cylinders, d.fuel, d.c1, d.c4, d.c5]


        T4, work_piston, eta_th, air_flow, p_max, T_max, far, equ_trapped, induced_power, friction_loss, aux_loss,\
            heat_loss, p_tdc, outflow, no, imep = run_piston_engine(data, flags)


        nitrogen_oxides_late.append(no)
        IMEPs_late.append(imep * 1e-5)
        peak_pressures_late.append(p_max * 1e-5)
        indicated_effs_late.append(eta_th * 1e2)

    # load data from Rakopoulos

    dirname = os.path.dirname(__file__)
    print(dirname)
    filename_no_early = os.path.join(dirname, '../../validation_output_data/NO_diesel/IMEP.txt')
    filename_pressure_early = os.path.join(dirname, '../../validation_output_data/NO_diesel/peak_pressure_high.txt')
    filename_eff_early = os.path.join(dirname, '../../validation_output_data/NO_diesel/eff_high.txt')
    filename_no_late = os.path.join(dirname, '../../validation_output_data/NO_diesel/no_low.txt')
    filename_pressure_late = os.path.join(dirname, '../../validation_output_data/NO_diesel/peak_pressure_low.txt')
    filename_eff_late = os.path.join(dirname, '../../validation_output_data/NO_diesel/eff_low.txt')

    no_early_val = np.loadtxt(filename_no_early, delimiter=",")
    no_late_val = np.loadtxt(filename_no_late, delimiter=",")
    pp_early_val = np.loadtxt(filename_pressure_early, delimiter=",")
    pp_late_val = np.loadtxt(filename_pressure_late, delimiter=",")
    eff_early_val = np.loadtxt(filename_eff_early, delimiter=",")
    eff_late_val = np.loadtxt(filename_eff_late, delimiter=",")
    fig, ax5 = plt.subplots()

    fs = 18
    ax5.plot(IMEPs_early, nitrogen_oxides_early, marker='o', label="Simulation")
    ax5.plot(IMEPs_late, nitrogen_oxides_late, marker='o', label="Simulation")
    ax5.plot(no_early_val[:, 0], no_early_val[:, 1], marker='x', label="Validation")
    ax5.plot(no_late_val[:, 0], no_late_val[:, 1], marker='x', label="Validation")
    ax5.set_xlabel(r'Indicated mean effective pressure (IMEP) [$bar$]', fontsize=fs)
    ax5.set_ylabel(r'NO concentration (ppm)', fontsize=fs)
    ax5.legend(loc='best', fontsize='small', frameon=False)


    fig, ax6 = plt.subplots()
    ax6.plot(IMEPs_early, peak_pressures_early, marker='o', label="Simulation")
    ax6.plot(IMEPs_late, peak_pressures_late, marker='o', label="Simulation")
    ax6.plot(pp_early_val[:, 0], pp_early_val[:, 1], marker='x', label="Validation")
    ax6.plot(pp_late_val[:, 0], pp_late_val[:, 1], marker='x', label="Validation")
    ax6.set_xlabel(r'Indicated mean effective pressure (IMEP) [$bar$]', fontsize=fs)
    ax6.set_ylabel(r'Peak pressure (bar)', fontsize=fs)
    ax6.legend(loc='best', fontsize='small', frameon=False)

    fig, ax7 = plt.subplots()
    ax7.plot(IMEPs_early, indicated_effs_early, marker='o', label="Simulation")
    ax7.plot(IMEPs_late, indicated_effs_late, marker='o', label="Simulation")
    ax7.plot(eff_early_val[:, 0], eff_early_val[:, 1], marker='x', label="Validation")
    ax7.plot(eff_late_val[:, 0], eff_late_val[:, 1], marker='x', label="Validation")
    ax7.set_xlabel(r'Indicated mean effective pressure (IMEP) [$bar$]', fontsize=fs)
    ax7.set_ylabel(r'Indicated efficiency (%)', fontsize=fs)
    ax7.legend(loc='best', fontsize='small', frameon=False)



    plt.show()

    return


