import matplotlib.pyplot as plt
import numpy as np
import os

from piston_engine.engine import run_piston_engine
from thermo import fuel_props, polynomials, mixture
from piston_engine.src.misc import post_processing

def sweep_equ(d, flags):

    num = 10

    lambdas = np.linspace(3, 6, num)
    #phi_cds = np.linspace(63 * np.pi / 180 , 19 * np.pi / 180, num)
    #phi_cds = np.linspace(63 * np.pi / 180, 19 * np.pi / 180, 10)

    imeps = []
    NOx = []
    brake_effs = []
    indicated_effs = []
    heat_percents = []
    friction_percents = []

    far_s, LHV = fuel_props("H2")

    # Universal gas constant from NASA polynomials pdf
    R = 8.314510  # J mol^-1 K^-1
    M_h2 = 2.0158800e-3  # kg/mol
    R_h2 = R / M_h2
    rho_h2 = d.p_in / (R_h2 * 293)


    for lamba in lambdas:

        equ = 1 / lamba
        far_goal = equ * far_s

        data = [d.p_in, d.T_in, d.p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, d.cr, d.d, d.bsr,
                d.v_mean, d.lms, d.Twalls, d.ch,
                d.valve_timings, d.n_valve, d.lv_max, d.cd, d.eta_c, d.mf_tot, d.wa,
                d.wm, d.m_wiebe, d.phi_sc, d.phi_cd, d.T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type, far_goal,
                d.cylinders, d.fuel, d.c1, d.c4, d.c5, d.premixed]


        T4, brake_power, eta_th, air_flow, p_max, T_max, far, equ_trapped, indicated_power, friction_loss, aux_loss, \
            heat_loss, p_tdc, outflow, no, imep, EI_nox, volume_eff, nox_spec = run_piston_engine(data, flags)

        print(f"Peak pressure: {p_max * 1e-5}, peak temp: {T_max}, NO: {no} (ppm),"
              f" imep: {imep*1e-5} bar, indicated power:{indicated_power * 1e-3} kW")

        fuel_power = air_flow * far_goal * LHV
        brake_eff = brake_power / fuel_power
        indicated_eff = indicated_power / fuel_power
        heat_percent = heat_loss / fuel_power
        friction_percent = friction_loss / fuel_power

        print(f"Fuel flow: {air_flow * far_goal * 60 * 1000 / rho_h2} dm^3 / min")

        imeps.append(imep * 1e-5)
        brake_effs.append(brake_eff * 1e2)
        indicated_effs.append(indicated_eff * 1e2)
        heat_percents.append(heat_percent * 1e2)
        friction_percents.append(friction_percent * 1e2)



        NOx.append(nox_spec)


    import os
    dirname = os.path.dirname(__file__)
    filename_nox = os.path.join(dirname, '../../validation_output_data/H2_water/nox.txt')
    nox_val = np.loadtxt(filename_nox, delimiter=",")


    fs = 18

    fig, ax6 = plt.subplots()
    ax6.plot(lambdas, brake_effs, label="Simulation", marker="o")
    #ax6.plot(power_val[:, 0], power_val[:, 1], label="Validation", marker="x", color='b')
    ax6.set_xlabel(r'Excess air ratio $\lambda$ ()', fontsize=fs)
    ax6.set_ylabel(r'Brake thermal efficiency [$\%$]', fontsize=fs)
    ax6.legend(loc='best', fontsize='small', frameon=False)
    #ax6.set_ylim([0, 10])

    fig, ax5 = plt.subplots()
    ax5.plot(lambdas, indicated_effs, label="Simulation", marker="o")
    #ax6.plot(power_val[:, 0], power_val[:, 1], label="Validation", marker="x", color='b')
    ax5.set_xlabel(r'Excess air ratio $\lambda$ ()', fontsize=fs)
    ax5.set_ylabel(r'Indicated thermal efficiency [$\%$]', fontsize=fs)
    ax5.legend(loc='best', fontsize='small', frameon=False)
    #ax6.set_ylim([0, 10])

    fig, ax4 = plt.subplots()
    ax4.plot(lambdas, heat_percents, label="Simulation", marker="o")
    #ax6.plot(power_val[:, 0], power_val[:, 1], label="Validation", marker="x", color='b')
    ax4.set_xlabel(r'Excess air ratio $\lambda$ ()', fontsize=fs)
    ax4.set_ylabel(r'Heat loss [$\%$]', fontsize=fs)
    ax4.legend(loc='best', fontsize='small', frameon=False)
    #ax6.set_ylim([0, 10])

    fig, ax2 = plt.subplots()
    ax2.plot(lambdas, friction_percents, label="Simulation", marker="o")
    #ax6.plot(power_val[:, 0], power_val[:, 1], label="Validation", marker="x", color='b')
    ax2.set_xlabel(r'Excess air ratio $\lambda$ ()', fontsize=fs)
    ax2.set_ylabel(r'Friction loss [$\%$]', fontsize=fs)
    ax2.legend(loc='best', fontsize='small', frameon=False)
    #ax6.set_ylim([0, 10])


    fig, ax3 = plt.subplots()
    ax3.plot(lambdas, NOx, label="Simulation", marker="o")
    #ax3.plot(nox_val[:, 0], nox_val[:, 1], label="Validation", marker="x", color='b')
    ax3.set_xlabel(r'Excess air ratio $\lambda$ ()', fontsize=fs)
    ax3.set_ylabel(r'Nitric oxide [g/kWh]', fontsize=fs)
    ax3.legend(loc='best', fontsize='small', frameon=False)
    #ax3.set_ylim([0, 10000])



    plt.show()

    return


