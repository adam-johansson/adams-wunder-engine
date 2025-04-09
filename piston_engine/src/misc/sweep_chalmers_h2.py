import matplotlib.pyplot as plt
import numpy as np
import os

from piston_engine.engine import run_piston_engine
from thermo import fuel_props, polynomials, mixture
from piston_engine.src.misc import post_processing

def sweep_chalmers(d, flags):

    num = 10

    p_ins = np.linspace(1.6e5, 1e5, num)
    air_fuels = np.linspace(1.5,2.5,num)

    imeps = []
    effs = []
    NOx = []

    far_s, LHV = fuel_props("H2")


    for p_in, air_fuel in zip(p_ins, air_fuels):

        equ = 1 / air_fuel

        far_goal = equ * far_s

        data = [p_in, d.T_in, d.p_ratio, d.cycle, d.thermo, d.cooling, d.opposed, d.cr, d.d, d.bsr,
                d.v_mean, d.lms, d.Twalls, d.ch,
                d.valve_timings, d.n_valve, d.lv_max, d.cd, d.eta_c, d.mf_tot, d.wa,
                d.wm, d.m_wiebe, d.phi_sc, d.phi_cd, d.T_fuel, d.p_fuel, d.it, d.wiebe_type, d.valve_type, far_goal,
                d.cylinders, d.fuel, d.c1, d.c4, d.c5, d.premixed]


        T4, brake_power, eta_th, air_flow, p_max, T_max, far, equ_trapped, indicated_power, friction_loss, aux_loss, \
            heat_loss, p_tdc, outflow, no, imep, EI_nox, volume_eff, nox_spec  = run_piston_engine(data, flags)

        print(f"Peak pressure: {p_max * 1e-5}, peak temp: {T_max}, NO: {nox_spec} g/kWh, imep: {imep} bar")

        imeps.append(imep * 1e-5)
        effs.append(eta_th * 1e2)
        NOx.append(nox_spec)


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


    fs = 18

    fig, ax6 = plt.subplots()
    ax6.plot(p_ins * 1e-5, imeps, label="Simulation", marker="o")
    #ax6.plot(power_val[:, 0], power_val[:, 1], label="Validation", marker="x", color='b')
    ax6.set_xlabel(r'P_in $\phi$ ()', fontsize=fs)
    ax6.set_ylabel(r'IMEP (bar)', fontsize=fs)
    ax6.legend(loc='best', fontsize='small', frameon=False)
    #ax6.set_ylim([0, 10])

    fig, ax5 = plt.subplots()
    ax5.plot(p_ins * 1e-5, effs, label="Simulation", marker="o")
    #ax5.plot(thermal_val[:, 0], thermal_val[:, 1], label="Validation", marker="x", color='b')
    ax5.set_xlabel(r'P_in ()', fontsize=fs)
    ax5.set_ylabel(r'Indicated thermal efficiency ($\%$)', fontsize=fs)
    ax5.legend(loc='best', fontsize='small', frameon=False)
    #ax5.set_ylim([0, 30])

    fig, ax4 = plt.subplots()
    ax4.plot(air_fuels, effs, label="Simulation", marker="o")
    #ax5.plot(thermal_val[:, 0], thermal_val[:, 1], label="Validation", marker="x", color='b')
    ax4.set_xlabel(r'Lambda ()', fontsize=fs)
    ax4.set_ylabel(r'Indicated thermal efficiency ($\%$)', fontsize=fs)
    ax4.legend(loc='best', fontsize='small', frameon=False)
    #ax5.set_ylim([0, 30])

    fig, ax2 = plt.subplots()
    ax2.plot(air_fuels, imeps, label="Simulation", marker="o")
    #ax5.plot(thermal_val[:, 0], thermal_val[:, 1], label="Validation", marker="x", color='b')
    ax2.set_xlabel(r'Lambda ()', fontsize=fs)
    ax2.set_ylabel(r'IMEP (bar)', fontsize=fs)
    ax2.legend(loc='best', fontsize='small', frameon=False)
    #ax5.set_ylim([0, 30])


    fig, ax3 = plt.subplots()
    ax3.plot(p_ins * 1e-5, NOx, label="Simulation", marker="o")
    #ax3.plot(nox_val[:, 0], nox_val[:, 1], label="Validation", marker="x", color='b')
    ax3.set_xlabel(r'P_in', fontsize=fs)
    ax3.set_ylabel(r'Nitric oxide (g/kWh)', fontsize=fs)
    ax3.legend(loc='best', fontsize='small', frameon=False)
    #ax3.set_ylim([0, 10000])

    plt.show()

    return


