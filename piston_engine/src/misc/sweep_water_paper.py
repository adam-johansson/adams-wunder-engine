import matplotlib.pyplot as plt
import numpy as np
import os

from piston_engine.engine import run_piston_engine
from thermo import fuel_props
from piston_engine.src.misc import post_processing

def sweep_equ(d, flags):

    equs = np.linspace(0.25, 0.95, 10)

    powers = []
    effs = []
    T_outs = []
    NOx = []

    for equ in equs:

        far_s, LHV = fuel_props("H2")

        far_goal = equ * far_s

        data = [d.p_in, d.T_in, d.p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, d.cr, d.d, d.bsr,
                d.v_mean, d.lms, d.Twalls, d.ch,
                d.valve_timings, d.n_valve, d.lv_max, d.cd, d.eta_c, d.mf_tot, d.wa,
                d.wm, d.m_wiebe, d.phi_sc, d.phi_cd, d.T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type, far_goal,
                d.cylinders, d.fuel, d.c1, d.c4, d.c5, d.premixed]


        T4, brake_power, eta_th, air_flow, p_max, T_max, far, equ_trapped, indicated_power, friction_loss, aux_loss, \
            heat_loss, p_tdc, outflow, no, imep, EI_nox = run_piston_engine(data, flags)

        print(f"Peak pressure: {p_max * 1e-5}, peak temp: {T_max}, NO: {no}")

        fuel_power = air_flow * far_goal * LHV
        brake_eff = brake_power / fuel_power

        powers.append(brake_power * 1e-3)
        effs.append(brake_eff * 1e2)
        T_outs.append(T4)
        NOx.append(no)


    fs = 18

    fig, ax6 = plt.subplots()
    ax6.plot(equs, powers, label="Simulation", marker="o")
    ax6.set_xlabel(r'Equivalence ratio $\phi$ ()', fontsize=fs)
    ax6.set_ylabel(r'Brake power (kW)', fontsize=fs)
    ax6.legend(loc='best', fontsize='small', frameon=False)

    fig, ax5 = plt.subplots()
    ax5.plot(equs, effs, label="Simulation", marker="o")
    ax5.set_xlabel(r'Equivalence ratio $\phi$ ()', fontsize=fs)
    ax5.set_ylabel(r'Brake thermal efficiency ($\%$)', fontsize=fs)
    ax5.legend(loc='best', fontsize='small', frameon=False)

    fig, ax4 = plt.subplots()
    ax4.plot(equs, T_outs, label="Simulation", marker="o")
    ax4.set_xlabel(r'Equivalence ratio $\phi$ ()', fontsize=fs)
    ax4.set_ylabel(r'Exhaust gas temperature ($K$)', fontsize=fs)
    ax4.legend(loc='best', fontsize='small', frameon=False)

    fig, ax3 = plt.subplots()
    ax3.plot(equs, NOx, label="Simulation", marker="o")
    ax3.set_xlabel(r'Equivalence ratio $\phi$ ()', fontsize=fs)
    ax3.set_ylabel(r'Nitric oxide (ppm)', fontsize=fs)
    ax3.legend(loc='best', fontsize='small', frameon=False)



    plt.show()

    return


