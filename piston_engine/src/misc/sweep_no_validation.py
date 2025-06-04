import matplotlib.pyplot as plt
import numpy as np
import os

from piston_engine.engine import run_piston_engine


def sweep_no_diesel_greek_validation(d, flags):

    num = 3
    # validate against the Rakolpoulos paper
    # the three different load cases (-20 degrees injection timing)

    # design point far 0.0425
    far_dp = 0.03975
    fuel_air_ratios = np.linspace(0.0267, far_dp, num)
    # cds = np.linspace(35, 41.72, num) * np.pi / 180
    # m_wiebes = np.linspace(2.20,2.21,num)

    phi_sc = (352.0 / 180) * np.pi  # angle at combustion start
    phi_cd = (40.0 / 180) * np.pi
    m_wiebe = 1.1

    nitrogen_oxides_early = []
    EI_early = []
    peak_pressures_early = []
    indicated_effs_early = []
    IMEPs_early = []

    for far_goal in fuel_air_ratios:

        # from combustion book
        phi_cd_adjusted = phi_cd * (far_goal / far_dp) ** 0.6

        data = [
            d.p_in,
            d.T_in,
            d.p_ratio,
            d.cycle,
            d.thermo,
            d.cooling,
            d.opposed,
            d.cr,
            d.d,
            d.bsr,
            d.v_mean,
            d.lms,
            d.Twalls,
            d.ch,
            d.valve_timings,
            d.n_valve,
            d.lv_max,
            d.cd,
            d.eta_c,
            d.mf_tot,
            d.wa,
            d.wm,
            m_wiebe,
            phi_sc,
            phi_cd_adjusted,
            d.T_fuel,
            d.p_fuel,
            d.it,
            d.wiebe_type,
            d.valve_type,
            far_goal,
            d.cylinders,
            d.fuel,
            d.c1,
            d.c4,
            d.c5,
            d.premixed,
        ]

        if far_goal > 0.039:
            flags.append("validate_nox_diesel_early")

        (
            T4,
            work_piston,
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
            outflow,
            no,
            imep,
            EI_nox,
            volume_eff,
            nox_spec,
        ) = run_piston_engine(data, flags)

        # print(f"Peak pressure: {p_max * 1e-5}, peak temp: {T_max}, NO: {no}")
        nitrogen_oxides_early.append(no)
        IMEPs_early.append(imep * 1e-5)
        peak_pressures_early.append(p_max * 1e-5)
        indicated_effs_early.append(eta_th * 1e2)
        EI_early.append(EI_nox)

    flags.remove("validate_nox_diesel_early")
    nitrogen_oxides_late = []
    peak_pressures_late = []
    indicated_effs_late = []
    IMEPs_late = []
    EI_late = []

    # the three different load cases (-15 degrees injection timing)
    far_dp = 0.0401
    fuel_air_ratios = np.linspace(0.0267, far_dp, num)
    # matches Woschni
    # fuel_air_ratios = np.array([0.031, 0.038, 0.0465])

    phi_sc = (355.0 / 180) * np.pi  # angle at combustion start
    phi_cd = (40.0 / 180) * np.pi
    m_wiebe = 1.5

    # m_wiebes = np.linspace(2.28, 2.28, num)
    # cds = np.linspace(35,39.96, num) * np.pi/180

    for far_goal in fuel_air_ratios:

        # from combustion book
        phi_cd_adjusted = phi_cd * (far_goal / far_dp) ** 0.6

        data = [
            d.p_in,
            d.T_in,
            d.p_ratio,
            d.cycle,
            d.thermo,
            d.cooling,
            d.opposed,
            d.cr,
            d.d,
            d.bsr,
            d.v_mean,
            d.lms,
            d.Twalls,
            d.ch,
            d.valve_timings,
            d.n_valve,
            d.lv_max,
            d.cd,
            d.eta_c,
            d.mf_tot,
            d.wa,
            d.wm,
            m_wiebe,
            phi_sc,
            phi_cd_adjusted,
            d.T_fuel,
            d.p_fuel,
            d.it,
            d.wiebe_type,
            d.valve_type,
            far_goal,
            d.cylinders,
            d.fuel,
            d.c1,
            d.c4,
            d.c5,
            d.premixed,
        ]

        if far_goal > 0.04:
            flags.append("validate_nox_diesel_late")

        (
            T4,
            work_piston,
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
            outflow,
            no,
            imep,
            EI_nox,
            volume_eff,
            nox_spec,
        ) = run_piston_engine(data, flags)

        # print(f"Peak pressure: {p_max * 1e-5}, peak temp: {T_max}, NO: {no}")
        nitrogen_oxides_late.append(no)
        IMEPs_late.append(imep * 1e-5)
        peak_pressures_late.append(p_max * 1e-5)
        indicated_effs_late.append(eta_th * 1e2)
        EI_late.append(EI_nox)

    # load data from Rakopoulos

    dirname = os.path.dirname(__file__)
    print(dirname)
    filename_no_early = os.path.join(
        dirname, "../../validation_output_data/NO_diesel/IMEP.txt"
    )
    filename_pressure_early = os.path.join(
        dirname, "../../validation_output_data/NO_diesel/peak_pressure_high.txt"
    )
    filename_eff_early = os.path.join(
        dirname, "../../validation_output_data/NO_diesel/eff_high.txt"
    )
    filename_no_late = os.path.join(
        dirname, "../../validation_output_data/NO_diesel/no_low.txt"
    )
    filename_pressure_late = os.path.join(
        dirname, "../../validation_output_data/NO_diesel/peak_pressure_low.txt"
    )
    filename_eff_late = os.path.join(
        dirname, "../../validation_output_data/NO_diesel/eff_low.txt"
    )

    no_early_val = np.loadtxt(filename_no_early, delimiter=",")
    no_late_val = np.loadtxt(filename_no_late, delimiter=",")
    pp_early_val = np.loadtxt(filename_pressure_early, delimiter=",")
    pp_late_val = np.loadtxt(filename_pressure_late, delimiter=",")
    eff_early_val = np.loadtxt(filename_eff_early, delimiter=",")
    eff_late_val = np.loadtxt(filename_eff_late, delimiter=",")
    fig, ax5 = plt.subplots()

    fs = 18
    ax5.plot(IMEPs_early, nitrogen_oxides_early, marker="o", label="Simulation early")
    ax5.plot(IMEPs_late, nitrogen_oxides_late, marker="o", label="Simulation late")
    ax5.plot(
        no_early_val[:, 0], no_early_val[:, 1], marker="x", label="Validation early"
    )
    ax5.plot(no_late_val[:, 0], no_late_val[:, 1], marker="x", label="Validation late")
    ax5.set_xlabel(r"Indicated mean effective pressure (IMEP) [$bar$]", fontsize=fs)
    ax5.set_ylabel(r"NO concentration (ppm)", fontsize=fs)
    ax5.legend(loc="best", fontsize="small", frameon=False)
    ax5.grid()

    fig, ax6 = plt.subplots()
    ax6.plot(IMEPs_early, peak_pressures_early, marker="o", label="Simulation")
    ax6.plot(IMEPs_late, peak_pressures_late, marker="o", label="Simulation")
    ax6.plot(pp_early_val[:, 0], pp_early_val[:, 1], marker="x", label="Validation")
    ax6.plot(pp_late_val[:, 0], pp_late_val[:, 1], marker="x", label="Validation")
    ax6.set_xlabel(r"Indicated mean effective pressure (IMEP) [$bar$]", fontsize=fs)
    ax6.set_ylabel(r"Peak pressure (bar)", fontsize=fs)
    ax6.legend(loc="best", fontsize="small", frameon=False)

    fig, ax7 = plt.subplots()
    ax7.plot(IMEPs_early, indicated_effs_early, marker="o", label="Simulation early")
    ax7.plot(IMEPs_late, indicated_effs_late, marker="o", label="Simulation late")
    ax7.plot(
        eff_early_val[:, 0], eff_early_val[:, 1], marker="x", label="Validation early"
    )
    ax7.plot(
        eff_late_val[:, 0], eff_late_val[:, 1], marker="x", label="Validation late"
    )
    ax7.set_xlabel(r"Indicated mean effective pressure (IMEP) [$bar$]", fontsize=fs)
    ax7.set_ylabel(r"Indicated efficiency (%)", fontsize=fs)
    ax7.legend(loc="best", fontsize="small", frameon=False)

    fig, ax8 = plt.subplots()
    ax8.plot(IMEPs_early, EI_early, marker="o", label="Simulation early")
    ax8.plot(IMEPs_late, EI_late, marker="o", label="Simulation late")
    ax8.set_xlabel(r"Indicated mean effective pressure (IMEP) [$bar$]", fontsize=fs)
    ax8.set_ylabel(r"NO emission index (g/kg)", fontsize=fs)
    ax8.legend(loc="best", fontsize="small", frameon=False)

    plt.show()

    return
