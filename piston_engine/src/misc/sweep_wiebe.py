import matplotlib.pyplot as plt
import numpy as np
import os

from piston_engine.engine import run_piston_engine


def sweep_wiebe(d, flags):

    # variable heat walls
    far_dp = 0.0428
    phi_scs = np.linspace(357 / 180 * np.pi, 363 / 180 * np.pi, 20)
    phi_cds = np.linspace(10 / 180 * np.pi, 60 / 180 * np.pi, 10)
    m_wiebes = np.linspace(0.1, 2.0, 10)

    # HIGH LOAD PARAMETERS
    m_wiebe = 0.9
    phi_sc = (357.0 / 180) * np.pi  # angle at combustion start
    phi_cd = (43 / 180) * np.pi  # angle related to combustion duration 43

    nitrogen_oxides = []
    EI = []
    eff = []

    for phi_sc in phi_scs:

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
            phi_cd,
            d.T_fuel,
            d.p_fuel,
            d.it,
            d.wiebe_type,
            d.valve_type,
            far_dp,
            d.cylinders,
            d.fuel,
            d.c1,
            d.c4,
            d.c5,
        ]

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
        ) = run_piston_engine(data, flags)

        print(f"Peak pressure: {p_max * 1e-5}, peak temp: {T_max}, NO: {no}")

        nitrogen_oxides.append(no)
        EI.append(EI_nox)
        eff.append(eta_th)

    fs = 18

    fig, ax1 = plt.subplots()
    ax1.plot(phi_scs * 180 / np.pi, nitrogen_oxides, marker="o")
    # ax1.plot(m_wiebes, nitrogen_oxides, marker="o")
    ax1.set_xlabel(r"Start of combustion $\theta_{SOC}$ [$^{\circ}$]", fontsize=fs)
    # ax1.set_xlabel(r'Combustion duration $\theta_{CD}$ [$^{\circ}$]', fontsize=fs)
    # ax1.set_xlabel(r'Wiebe shape $m_w$ [-]', fontsize=fs)
    ax1.set_ylabel(r"NO exhaust concentration (ppm)", fontsize=fs)
    # ax1.legend(loc='best', fontsize='small', frameon=False)

    fig, ax2 = plt.subplots()
    ax2.plot(phi_scs * 180 / np.pi, EI, marker="o")
    # ax2.plot(m_wiebes, EI, marker="o")
    ax2.set_xlabel(r"Start of combustion $\theta_{SOC}$ [$^{\circ}$]", fontsize=fs)
    # ax2.set_xlabel(r'Combustion duration $\theta_{CD}$ [$^{\circ}$]', fontsize=fs)
    # ax2.set_xlabel(r'Wiebe shape $m_w$ [-]', fontsize=fs)
    ax2.set_ylabel(r"NO emission index (g/kg)", fontsize=fs)
    # ax2.legend(loc='best', fontsize='small', frameon=False)

    fig, ax3 = plt.subplots()
    ax3.plot(phi_scs * 180 / np.pi, eff, marker="o")
    # ax3.plot(m_wiebes, eff, marker="o")
    ax3.set_xlabel(r"Start of combustion $\theta_{SOC}$ [$^{\circ}$]", fontsize=fs)
    # ax3.set_xlabel(r'Combustion duration $\theta_{CD}$ [$^{\circ}$]', fontsize=fs)
    # ax3.set_xlabel(r'Wiebe shape $m_w$ [-]', fontsize=fs)
    ax3.set_ylabel(r"Eff (%)", fontsize=fs)
    # ax2.legend(loc='best', fontsize='small', frameon=False)

    plt.show()

    return
