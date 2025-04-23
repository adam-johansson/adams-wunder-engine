import matplotlib.pyplot as plt
import numpy as np
import os

from piston_engine.engine import run_piston_engine


def sweep_no_diesel_kth_validation(d, flags):

    comp_eff = 0.8
    T_amb = 298
    p_atm = 1.01325e5
    p_ins = np.array([p_atm, p_atm + 0.4e5, p_atm + 0.75e5])

    # fuel air ratio at point where heat release curve is calibrated
    # far_dp = 0.0069
    # fuel_air_ratios = np.array([0.011, 0.0081, far_dp])

    # variable heat walls
    far_dp = 0.0073
    fuel_air_ratios = np.array([0.0103, 0.0084, far_dp])

    # pressure and temperature at intake valve closing for the design point
    T_dp = T_amb * ((p_atm + 0.75e5) / p_atm) ** ((1.4 - 1) / (1.4 * comp_eff))
    p_dp = p_atm + 0.75e5

    nitrogen_oxides_low = []
    EI_low = []
    # 25 and 0.15
    m_wiebe = 0.25
    phi_sc = (358.0 / 180) * np.pi  # angle at combustion start
    phi_cd = (20 / 180) * np.pi  # angle related to combustion duration 43

    for far_goal, p_in in zip(fuel_air_ratios, p_ins):

        T_in = T_amb * (p_in / p_atm) ** ((1.4 - 1) / (1.4 * comp_eff))
        # p_ratio = p_atm / p_in
        p_ratio = 0.9

        # Twalls = np.array([400, 400, 400])
        Twalls = np.array([1, 1, 1]) * ((p_in - p_atm) / 0.75e5) * 100 + 400

        phi_cd_adjusted = phi_cd * (far_goal / far_dp) ** 0.6

        m_wiebe_adjusted = m_wiebe * ((p_in * T_dp) / (p_dp * T_in))

        data = [
            p_in,
            T_in,
            p_ratio,
            d.cycle,
            d.thermo,
            d.cooling,
            d.opposed,
            d.cr,
            d.d,
            d.bsr,
            d.v_mean,
            d.lms,
            Twalls,
            d.ch,
            d.valve_timings,
            d.n_valve,
            d.lv_max,
            d.cd,
            d.eta_c,
            d.mf_tot,
            d.wa,
            d.wm,
            m_wiebe_adjusted,
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

        if p_in > 1.7e5:
            flags.append("validate_scania_lowload")

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
        ) = run_piston_engine(data, flags)

        print(f"Peak pressure: {p_max * 1e-5}, peak temp: {T_max}, NO: {no}")
        # print(f"NOx: {no}")
        nitrogen_oxides_low.append(no)
        EI_low.append(EI_nox)

    flags.remove("validate_scania_lowload")

    # fuel air ratio at point where heat release curve is calibrated
    # for walls 400
    far_dp = 0.0405
    fuel_air_ratios = np.array([0.06, 0.047, far_dp])

    # variable heat walls
    far_dp = 0.0428
    fuel_air_ratios = np.array([0.0605, 0.049, far_dp])

    # HIGH LOAD PARAMETERS
    m_wiebe = 0.9
    phi_sc = (357.0 / 180) * np.pi  # angle at combustion start
    phi_cd = (43 / 180) * np.pi  # angle related to combustion duration 43

    nitrogen_oxides_high = []
    EI_high = []
    for far_goal, p_in in zip(fuel_air_ratios, p_ins):

        T_in = T_amb * (p_in / p_atm) ** ((1.4 - 1) / (1.4 * comp_eff))
        # p_ratio = p_atm / p_in
        p_ratio = 0.9
        Twalls = np.array([1, 1, 1]) * ((p_in - p_atm) / 0.75e5) * 100 + 400
        # Twalls = np.array([400, 400, 400])

        phi_cd_adjusted = phi_cd * (far_goal / far_dp) ** 0.6
        # estimate
        m_wiebe_adjusted = m_wiebe * ((p_in * T_dp) / (p_dp * T_in))

        data = [
            p_in,
            T_in,
            p_ratio,
            d.cycle,
            d.thermo,
            d.cooling,
            d.opposed,
            d.cr,
            d.d,
            d.bsr,
            d.v_mean,
            d.lms,
            Twalls,
            d.ch,
            d.valve_timings,
            d.n_valve,
            d.lv_max,
            d.cd,
            d.eta_c,
            d.mf_tot,
            d.wa,
            d.wm,
            m_wiebe_adjusted,
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

        if p_in > 1.7e5:
            flags.append("validate_scania_highload")

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
        ) = run_piston_engine(data, flags)

        print(f"Peak pressure: {p_max * 1e-5}, peak temp: {T_max}, NO: {no}")
        # print(f"NOx: {no}")
        nitrogen_oxides_high.append(no)
        EI_high.append(EI_nox)

    # load data from KTH Msc thesis
    dirname = os.path.dirname(__file__)
    nox_val_exp_file = os.path.join(
        dirname, "../../validation_output_data/Scania/nox_exp.txt"
    )
    nox_val_sim_file = os.path.join(
        dirname, "../../validation_output_data/Scania/nox_sim.txt"
    )

    nox_val_exp = np.loadtxt(nox_val_exp_file, delimiter=",")
    nox_val_sim = np.loadtxt(nox_val_sim_file, delimiter=",")

    fig, ax5 = plt.subplots()
    fs = 18
    ax5.plot(p_ins * 1e-5, nitrogen_oxides_high, label="Simulation, 135 mg", marker="o")
    ax5.plot(p_ins * 1e-5, nitrogen_oxides_low, label="Simulation, 23 mg", marker="o")
    ax5.plot(p_ins * 1e-5, nox_val_exp[3:], label="Experimental, 135 mg", marker="x")
    ax5.plot(p_ins * 1e-5, nox_val_exp[:3], label="Experimental, 23 mg", marker="x")
    ax5.plot(p_ins * 1e-5, nox_val_sim[:3], label="Validation sim, 23 mg", marker="D")
    ax5.plot(p_ins * 1e-5, nox_val_sim[3:], label="Validation sim, 135 mg", marker="D")
    ax5.set_xlabel(r"Inlet pressure [$bar$]", fontsize=fs)
    ax5.set_ylabel(r"NO concentration (ppm)", fontsize=fs)
    ax5.legend(loc="best", fontsize="small", frameon=False)
    ax5.set_xticks([1, 1.4, 1.75])

    fig, ax6 = plt.subplots()
    ax6.plot(p_ins * 1e-5, EI_high, label="Simulation, 135 mg", marker="o")
    ax6.plot(p_ins * 1e-5, EI_low, label="Simulation, 23 mg", marker="o")
    ax6.set_xlabel(r"Inlet pressure [$bar$]", fontsize=fs)
    ax6.set_ylabel(r"NO emission index (g/kg)", fontsize=fs)
    ax6.legend(loc="best", fontsize="small", frameon=False)
    ax6.set_xticks([1, 1.4, 1.75])

    plt.show()

    return
