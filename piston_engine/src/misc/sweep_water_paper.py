import matplotlib.pyplot as plt
import numpy as np
import os

from piston_engine.engine import run_piston_engine
from thermo import fuel_props, polynomials, mixture
from piston_engine.src.misc import post_processing

def sweep_equ(d, flags):

    num = 30

    equs = np.linspace(0.26, 0.99, num)
    phi_cds = np.linspace(63 * np.pi / 180 , 19 * np.pi / 180, num)
    #phi_cds = np.linspace(63 * np.pi / 180, 19 * np.pi / 180, 10)

    powers = []
    effs = []
    T_outs = []
    NOx = []
    volume_effs = []

    far_s, LHV = fuel_props("H2")

    #phi_cd_adjusted = phi_cd * (far_goal / far_dp) ** 0.6

    for equ, phi_cd in zip(equs, phi_cds):


        far_goal = equ * far_s

        m_wiebe = 2.5  # 4.0

        phi_sc = (359.0 / 180) * np.pi  # 359
        #phi_cd = (23.0 / 180) * np.pi  # 19.0

        data = [d.p_in, d.T_in, d.p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, d.cr, d.d, d.bsr,
                d.v_mean, d.lms, d.Twalls, d.ch,
                d.valve_timings, d.n_valve, d.lv_max, d.cd, d.eta_c, d.mf_tot, d.wa,
                d.wm, m_wiebe, phi_sc, phi_cd, d.T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type, far_goal,
                d.cylinders, d.fuel, d.c1, d.c4, d.c5, d.premixed]


        T4, brake_power, eta_th, air_flow, p_max, T_max, far, equ_trapped, indicated_power, friction_loss, aux_loss, \
            heat_loss, p_tdc, outflow, no, imep, EI_nox, volume_eff, nox_spec = run_piston_engine(data, flags)

        print(f"Peak pressure: {p_max * 1e-5}, peak temp: {T_max}, NO: {no}")

        fuel_power = air_flow * far_goal * LHV
        brake_eff = brake_power / fuel_power

        powers.append(brake_power * 1e-3)
        effs.append(brake_eff * 1e2)
        T_outs.append(T4)
        NOx.append(no)
        volume_effs.append(volume_eff * 1e2)


    import os
    dirname = os.path.dirname(__file__)
    filename_egt = os.path.join(dirname, '../../validation_output_data/H2_water/egt.txt')
    filename_volume = os.path.join(dirname, '../../validation_output_data/H2_water/volume_eff.txt')
    filename_thermal = os.path.join(dirname, '../../validation_output_data/H2_water/thermal_eff.txt')
    filename_power = os.path.join(dirname, '../../validation_output_data/H2_water/power.txt')
    filename_nox = os.path.join(dirname, '../../validation_output_data/H2_water/nox.txt')
    egt_val = np.loadtxt(filename_egt, delimiter=",")
    volume_val = np.loadtxt(filename_volume, delimiter=",")
    power_val = np.loadtxt(filename_power, delimiter=",")
    thermal_val = np.loadtxt(filename_thermal, delimiter=",")
    nox_val = np.loadtxt(filename_nox, delimiter=",")

    # convert validation exhaust gas temperature (egt) to Kelvin from Celsius
    egt_val[:, 1] = egt_val[:, 1] + 273.15

    # analytical expression for volumetric efficiency
    equ_analytic = np.linspace(0.25,0.9,100)
    # fuel air mass ratio
    far_analytic_mass = equ_analytic * far_s

    _, _, _, _, M_H2 = polynomials.H2(298, 1e5)
    _, _, _, _, _, _, _, M_air = mixture(298, 1e5)

    # fuel air volume ratio (directly proportional to density and hence molar mass)
    far_analytic_volume = far_analytic_mass * (M_air / M_H2)

    # volumetric efficiency (volume of air in the fuel-air mixture)
    volume_eff_analytic = 1 / (1 + far_analytic_volume)


    fs = 18

    fig, ax6 = plt.subplots()
    ax6.plot(equs, powers, label="Simulation", marker="o")
    ax6.plot(power_val[:, 0], power_val[:, 1], label="Validation", marker="x", color='b')
    ax6.set_xlabel(r'Equivalence ratio $\phi$ ()', fontsize=fs)
    ax6.set_ylabel(r'Brake power (kW)', fontsize=fs)
    ax6.legend(loc='best', fontsize='small', frameon=False)
    ax6.set_ylim([0, 10])

    fig, ax5 = plt.subplots()
    ax5.plot(equs, effs, label="Simulation", marker="o")
    ax5.plot(thermal_val[:, 0], thermal_val[:, 1], label="Validation", marker="x", color='b')
    ax5.set_xlabel(r'Equivalence ratio $\phi$ ()', fontsize=fs)
    ax5.set_ylabel(r'Brake thermal efficiency ($\%$)', fontsize=fs)
    ax5.legend(loc='best', fontsize='small', frameon=False)
    ax5.set_ylim([0, 30])

    fig, ax4 = plt.subplots()
    ax4.plot(equs, T_outs, label="Simulation", marker="o")
    ax4.plot(egt_val[:, 0], egt_val[:, 1], label="Validation", marker="x", color='b')
    ax4.set_xlabel(r'Equivalence ratio $\phi$ ()', fontsize=fs)
    ax4.set_ylabel(r'Exhaust gas temperature ($K$)', fontsize=fs)
    ax4.legend(loc='best', fontsize='small', frameon=False)
    ax4.set_ylim([0, 1200])

    fig, ax3 = plt.subplots()
    ax3.plot(equs, NOx, label="Simulation", marker="o")
    ax3.plot(nox_val[:, 0], nox_val[:, 1], label="Validation", marker="x", color='b')
    ax3.set_xlabel(r'Equivalence ratio $\phi$ ()', fontsize=fs)
    ax3.set_ylabel(r'Nitric oxide (ppm)', fontsize=fs)
    ax3.legend(loc='best', fontsize='small', frameon=False)
    #ax3.set_ylim([0, 10000])


    fig, ax2 = plt.subplots()
    ax2.plot(equs, volume_effs, label="Simulation", marker="o")
    ax2.plot(volume_val[:, 0], volume_val[:, 1], label="Validation", marker="x", color='b')
    ax2.plot(equ_analytic, volume_eff_analytic * 1e2, label="Analytical")
    ax2.set_xlabel(r'Equivalence ratio $\phi$ ()', fontsize=fs)
    ax2.set_ylabel(r'Volumetric efficiency ($\%$)', fontsize=fs)
    ax2.legend(loc='best', fontsize='small', frameon=False)
    ax2.set_ylim([0, 100])


    plt.show()

    return


