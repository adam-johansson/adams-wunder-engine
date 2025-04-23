import matplotlib.pyplot as plt
import numpy as np
import os

from piston_engine.engine import run_piston_engine
from thermo import fuel_props, polynomials, mixture
from piston_engine.src.misc import post_processing


def sweep_equ(d, flags):

    num = 10

    equs = np.linspace(0.4, 0.99, num)
    phi_cds = np.linspace(63 * np.pi / 180, 19 * np.pi / 180, num)
    # phi_cds = np.linspace(63 * np.pi / 180, 19 * np.pi / 180, 10)

    imeps = []
    effs = []
    NOx = []

    far_s, LHV = fuel_props("H2")

    phi_cd = (20 / 180) * np.pi  # given
    far_dp = far_s

    for equ in equs:

        far_goal = equ * far_s

        phi_cd_adjusted = phi_cd * (far_goal / far_dp) ** 0.6
        print(phi_cd_adjusted * 180 / np.pi)

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
            d.m_wiebe,
            d.phi_sc,
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

        (
            T4,
            brake_power,
            eta_th,
            air_flow,
            p_max,
            T_max,
            far,
            equ_trapped,
            indicated_power,
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

        print(
            f"Peak pressure: {p_max * 1e-5}, peak temp: {T_max}, NO: {no}, imep: {imep*1e-5} bar"
        )

        fuel_power = air_flow * far_goal * LHV
        brake_eff = brake_power / fuel_power

        imeps.append(imep * 1e-5)
        effs.append(brake_eff * 1e2)
        NOx.append(no)

    import os

    dirname = os.path.dirname(__file__)
    filename_nox = os.path.join(
        dirname, "../../validation_output_data/H2_water/nox.txt"
    )
    nox_val = np.loadtxt(filename_nox, delimiter=",")

    fs = 18

    fig, ax3 = plt.subplots()
    ax3.plot(imeps, NOx, label="Simulation", marker="o")
    # ax3.plot(nox_val[:, 0], nox_val[:, 1], label="Validation", marker="x", color='b')
    ax3.set_xlabel(r"IMEP (bar)", fontsize=fs)
    ax3.set_ylabel(r"Nitric oxide (ppm)", fontsize=fs)
    ax3.legend(loc="best", fontsize="small", frameon=False)
    # ax3.set_ylim([0, 10000])

    plt.show()

    return
